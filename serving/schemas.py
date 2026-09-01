"""
Request/response contracts for the FastAPI scoring endpoint
(serving/app.py). Two raw wire shapes in (SS7Transaction / SMPPTransaction -
one per protocol, field names matching each protocol's own raw CDR
columns, same ones ingestion/ss7.py and ingestion/smpp.py already map from
for batch training), one shared ScoreResponse out - see serving/app.py's
module docstring for why there are two request parsers but one scoring
path.

SCOPE: rule_pattern_score (serving/scoring.py) drives fraud_results/
recommended_action, per the external response contract this build has to
match. anomaly_score (serving/anomaly_scoring.py) is scored alongside it
and surfaced on ScoreResponse.anomaly_score for visibility - per
CLAUDE.md's "Keep rule_pattern_score and anomaly_score separate. Do not
average them. Disagreements between the two scores are valuable and
should remain visible", it does NOT feed the FRAUD/NOT_FRAUD decision or
recommended_action yet; that's a deliberate, disclosed follow-up (same
staged-rollout status as LIME/SHAP - see serving/app.py's module
docstring on _confidence/_reason_codes being disclosed heuristics, not
final).
"""
from typing import Annotated, Literal

from pydantic import BaseModel, Field


class SS7Transaction(BaseModel):
    """Field names match ingestion/ss7.py's raw column names directly
    (calling_gt, called_gt, dcs, imsi, pid, tpdu_length, message_type,
    sarref, msg_part, msg_parts) - this is the live single-message
    equivalent of one row of the raw SS7 CDR ingestion/ss7.py already
    parses for batch training."""

    reference: str
    time_stamp: int  # epoch milliseconds, per the sample payload
    message_type: int
    calling_gt: str
    called_gt: str
    a_number: str | None = None
    b_number: str | None = None
    content: str  # hex bytes, space-separated (e.g. "d9 77 5d ...")
    dcs: str | int
    pid: str | int | None = None
    imsi: str | None = None
    service_centre_address: str | None = None
    service_centre_time_stamp: str | None = None
    tpdu_length: int | None = None
    sarref: int | None = None
    msg_part: int | None = None
    msg_parts: int | None = None


class SMPPTransaction(BaseModel):
    """Field names match the live SMPP wire shape - a_number/b_number
    (+ oa_ton/oa_npi/da_ton/da_npi) rather than ingestion/smpp.py's raw
    `oa`/`da` batch-CDR column names, since that's what the actual
    request payload carries; mapped to the same `originator`/
    `destination` canonical columns either way (see serving/canonical.py)."""

    reference: str
    time_stamp: str  # ISO 8601, per the sample payload
    smpp_operation: str | int
    system_id: str | None = None
    source_ip: str | None = None
    a_number: str
    oa_ton: str | int | None = None
    oa_npi: str | int | None = None
    b_number: str
    da_ton: str | int | None = None
    da_npi: str | int | None = None
    content: str  # hex bytes, space-separated
    dcs: str | int
    esm_class: str | int | None = None  # NOTE: the live wire payload uses
    # `esm_class`, not ingestion/smpp.py's batch-CDR column name
    # `esme_class` (see that module's FEATURE_MAP comment - dropped there
    # too, not required) - renamed to match what requests actually send.
    # Unused by serving/canonical.py's mapping either way.
    msg_type: str | int | None = None
    gsm_features: str | int | None = None
    messaging_mode: str | int | None = None
    sar_ref: int | None = None
    sar_msg_parts: int | None = None
    sar_msg_part: int | None = None
    app_dest_port: int | None = None
    app_src_port: int | None = None


class SS7ScoreRequest(BaseModel):
    # Omitted/empty means "evaluate all supported fraud types" per the API
    # spec, NOT "evaluate nothing" - serving/app.py's score() resolves
    # that (an empty/omitted list here still needs to be turned into "all
    # supported types" before use, not read as-is).
    fraud_types: list[str] = Field(default_factory=list)
    # Required (spec: mandatory) - controls whether evaluation stops at
    # the first FRAUD prediction (False) or continues through every
    # requested fraud_type regardless (True). Currently a no-op in
    # practice: this build only ever evaluates one fraud_type (SPAM_SMS -
    # see FraudPredictionResult's docstring), so there's nothing further
    # to stop early from yet. Still required/validated now so requests
    # match the spec, and serving/app.py's evaluation loop is already
    # structured to honor it once a second fraud_type is modeled.
    deep_scan: bool
    protocol: Literal["SS7"]
    transaction: SS7Transaction


class SMPPScoreRequest(BaseModel):
    fraud_types: list[str] = Field(default_factory=list)
    deep_scan: bool
    protocol: Literal["SMPP"]
    transaction: SMPPTransaction


# Pydantic discriminated union - `protocol` on each member selects which
# transaction shape applies, so FastAPI validates straight into the right
# model instead of this module hand-rolling a dispatch-by-dict-shape step.
ScoreRequest = Annotated[
    SS7ScoreRequest | SMPPScoreRequest, Field(discriminator="protocol")
]


class FraudPredictionResult(BaseModel):
    """One evaluated fraud_type's result. ONLY fraud_types this build
    actually models are ever included in ScoreResponse.fraud_results - see
    that field's docstring. Today that's SPAM_SMS only: rule_pattern_score
    is trained specifically on fraud_type=="spam" (labels/rule_labels.py's
    build_rule_labels()), there is no SMISHING-specific score yet - a
    requested SMISHING is honestly omitted rather than reusing SPAM_SMS's
    number under a different label."""

    fraud_type: str
    prediction: Literal["FRAUD", "NOT_FRAUD"]
    risk_score: int = Field(ge=0, le=100)
    confidence: int = Field(ge=0, le=100)
    reason_codes: list[str] = Field(default_factory=list)


class ScoreResponse(BaseModel):
    """Matches the external API response contract exactly (reference/
    status/error_message/recommended_action/processing_time_ms/
    model_version/fraud_results) - no extra top-level fields, so this
    stays a drop-in match for callers built against that contract."""

    reference: str
    status: Literal["SUCCESS", "FAILURE"]
    error_message: str | None = None
    recommended_action: Literal["BLOCK", "PASS"] | None = None
    processing_time_ms: int | None = None
    model_version: str | None = None
    # Mandatory (non-None) when status == SUCCESS; None when status ==
    # FAILURE. NOT one entry per requested fraud_type - see
    # FraudPredictionResult's docstring: only fraud_types this build
    # actually models (SPAM_SMS) ever appear, even if the request asked
    # for others too (e.g. SMISHING).
    fraud_results: list[FraudPredictionResult] | None = None
    # ADDITIVE field, not part of the external response contract's
    # original spec - see module docstring's SCOPE note. None whenever
    # anomaly scoring itself fails (no promoted champion yet, missing
    # corpus for this source, etc.) - a rule_pattern_score success is
    # never downgraded to FAILURE just because anomaly_score couldn't be
    # computed, see serving/app.py's score().
    anomaly_score: float | None = None
