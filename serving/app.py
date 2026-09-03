"""
FastAPI service - CLAUDE.md's "Next: 1. FastAPI service combining both
scores" step. Both scores are computed: rule_pattern_score
(serving/scoring.py) drives fraud_results/recommended_action per the
external response contract; anomaly_score (serving/anomaly_scoring.py) is
computed alongside it and surfaced on ScoreResponse.anomaly_score for
visibility only - see serving/schemas.py's module docstring for why it
doesn't gate the FRAUD/NOT_FRAUD decision yet.

TWO endpoints (/v1/score/smpp, /v1/score/ss7), ONE scoring path: SMPP and
SS7 carry genuinely different raw wire fields (serving/schemas.py's
SMPPTransaction/SS7Transaction) and now get separate routes/request
models too - but CLAUDE.md is still explicit that both must map to the
same canonical schema before touching any model ("SMPP and SS7 must map
to the same canonical schema before shared ML", "Start with one shared
model; split by source only if segment evaluation justifies it") - so
protocol-specific parsing lives in serving/canonical.py, and everything
from there on (Feast lookup, scoring, response-building) is shared via
`_score()`, `source` carried through as a feature rather than a routing
key for the MODEL side (rule_pattern_score is separately source-suffixed
per model, see serving/scoring.py - that split is independent of this
one, which is purely about having two request shapes/routes instead of a
discriminated union on one).

RESPONSE CONTRACT is an external spec (reference/status/error_message/
recommended_action/processing_time_ms/model_version/fraud_results) - see
serving/schemas.py's ScoreResponse docstring. HTTP 200 is returned for
every syntactically valid request this handler reaches, success or failure
- `status`/`error_message` carry outcome, not the HTTP status line
(FastAPI's own 422 still applies one layer up, for a request that doesn't
even match SS7ScoreRequest/SMPPScoreRequest's shape).

fraud_results ONLY EVER CONTAINS SPAM_SMS (see
schemas.FraudPredictionResult's docstring) - rule_pattern_score is trained
specifically on fraud_type=="spam" (labels/rule_labels.py), there is no
SMISHING-specific score in this build. A request that asks only for
SMISHING gets back an empty fraud_results array, not a fabricated number.

risk_score/confidence below remain explicit, DISCLOSED heuristics, not a
calibrated business threshold - _FRAUD_THRESHOLD=0.5 is still an
unvalidated starting point.

reason_codes/feature_contributions are now REAL: _reason_codes() derives
its codes from serving.scoring.explain_rule_pattern()'s actual per-request
shap.TreeExplainer contributions (CLAUDE.md's "Next: 1. Wire real
SHAP/LIME output into /v1/score's reason_codes" - done for SHAP; LIME is
deliberately NOT wired in here, see explain_rule_pattern()'s docstring for
why it isn't safe to run inline). The reason-code STRINGS are still our
own placeholder vocabulary (external spec's Table 8-4 enum unknown as of
writing) - only which codes fire is now evidence-based, not the strings
themselves. Explanation is only computed for a FRAUD prediction (cost-
control: TreeExplainer is cheap but still real work, no reason to pay it
for every NOT_FRAUD request) and is treated as best-effort, same as
anomaly_score below - a failure here degrades reason_codes/
feature_contributions, it never turns a successful score into FAILURE.
"""
import time

from fastapi import FastAPI

from serving.anomaly_scoring import score_anomaly
from serving.anomaly_scoring import ChampionUnavailableError as AnomalyChampionUnavailableError
from serving.anomaly_scoring import CorpusUnavailableError
from serving.canonical import CanonicalRow, map_smpp_transaction, map_ss7_transaction
from serving.feature_lookup import get_imsi_features, get_sender_features
from serving.schemas import (
    FeatureContribution,
    FraudPredictionResult,
    ScoreResponse,
    SMPPScoreRequest,
    SS7ScoreRequest,
)
from serving.scoring import (
    BEHAVIORAL_COLS,
    ChampionUnavailableError,
    ChampionUnsupportedError,
    explain_rule_pattern,
    score_rule_pattern,
)

app = FastAPI(title="Spam Detection Scoring API")

# Only fraud_type this build actually models - see module docstring.
# A tuple, not a set: deep_scan's early-stop semantics (score.py's
# evaluation loop) depend on evaluating requested types in a fixed,
# deterministic order once a second type exists here.
_SUPPORTED_FRAUD_TYPES = ("SPAM_SMS",)

# Probability cutoff for FRAUD vs NOT_FRAUD - an unvalidated starting
# point (0.5), same status as models/anomaly/train.py's un-tuned
# `contamination="auto"`: the real precision/recall-calibrated cutoff is
# a later, business-driven decision, not baked in here.
_FRAUD_THRESHOLD = 0.5

# How many top-|contribution| features feed feature_contributions and
# get checked for reason-code membership below - not a tuned value, just
# small enough to stay a "top reasons" list rather than dumping every
# feature the model used.
_TOP_K_CONTRIBUTIONS = 5


def _confidence(cold_start: bool, text_decode_failed: bool) -> int:
    """Feature-completeness-based confidence, NOT a calibrated
    probability - see architecture plan's "confidence: independent
    measure of how much to trust [probability]... cold-start senders =
    lower confidence" (docs/sms_spam_technical_architecture_plan.md).
    A cold-start sender has no behavioral history at all (the 4
    BEHAVIORAL_COLS this model actually uses) - undecodable content means
    the model saw an empty `text_length`/failure flag instead of the real
    message. Both lower trust in the score; neither invalidates it."""
    confidence = 90
    if cold_start:
        confidence -= 40
    if text_decode_failed:
        confidence -= 20
    return max(confidence, 10)


def _reason_codes(
    row: dict, cold_start: bool, contributions: list[tuple[str, float, float]],
) -> list[str]:
    """Reason codes for a FRAUD prediction, membership now driven by REAL
    per-request SHAP contributions (serving.scoring.explain_rule_pattern),
    not fixed value thresholds like before: a feature only earns its code
    if it's among the top _TOP_K_CONTRIBUTIONS by |contribution| AND its
    contribution is positive (pushed THIS row toward FRAUD, not just
    present). `contributions` is `[]` whenever explanation failed/was
    unavailable (see score()'s try/except) - degrades to the
    cold_start/text_decode_failed codes plus the base code, same as
    before explainability existed.

    TODO: these 5 codes are still OUR OWN placeholder set, not validated
    against the external spec's Table 8-4 enum (unknown as of writing -
    the one example response seen so far used "PROMOTIONAL_CONTENT",
    which isn't in this list). Revisit once Table 8-4 is available: align
    these strings to the real enum values and consider a typed Enum here
    instead of list[str], so an out-of-spec code can't silently go out."""
    positive_top = {
        feature for feature, _value, shap_value in contributions[:_TOP_K_CONTRIBUTIONS]
        if shap_value > 0
    }
    codes = ["KNOWN_SPAM_PATTERN"]
    if "sender_repeat_content_ratio_1hr" in positive_top:
        codes.append("REPEATED_CONTENT")
    if {"sender_msgs_last_1hr", "sender_msgs_last_5min"} & positive_top:
        codes.append("HIGH_SENDER_VELOCITY")
    if cold_start:
        codes.append("NEW_SENDER_LOW_HISTORY")
    if row.get("text_decode_failed"):
        codes.append("UNDECODABLE_CONTENT")
    return codes


@app.post("/v1/score/smpp", response_model=ScoreResponse)
def score_smpp(request: SMPPScoreRequest) -> ScoreResponse:
    return _score(request, map_smpp_transaction(request.transaction))


@app.post("/v1/score/ss7", response_model=ScoreResponse)
def score_ss7(request: SS7ScoreRequest) -> ScoreResponse:
    return _score(request, map_ss7_transaction(request.transaction))


def _score(request: SMPPScoreRequest | SS7ScoreRequest, canonical: CanonicalRow) -> ScoreResponse:
    """Shared body behind both routes above - request-shape/canonical-
    mapping is the only thing that differs per protocol (see module
    docstring); everything from here on (Feast lookup, both scores,
    response-building) is identical regardless of which endpoint was
    hit."""
    start = time.perf_counter()
    reference = request.transaction.reference

    try:
        behavioral = get_sender_features(canonical.sender_id, canonical.text)
        # SS7-only, keyed on imsi not sender_id (serving/feature_lookup.py's
        # get_imsi_features() docstring) - merged into the same dict since
        # serving/scoring.py's build_rule_pattern_row() reads every feature
        # from one `behavioral` mapping regardless of which entity it came
        # from. canonical.imsi is None for SMPP / an SS7 request with no
        # imsi - get_imsi_features() short-circuits that case itself.
        behavioral = {**behavioral, **get_imsi_features(canonical.imsi)}
    except Exception as e:  # Feast store missing/unreachable, etc. - a
        # real operational failure, not a modeling one.
        return ScoreResponse(
            reference=reference, status="FAILURE",
            error_message=f"behavioral feature lookup failed: {e}",
        )
    cold_start = all(behavioral.get(c) is None for c in BEHAVIORAL_COLS)

    try:
        probability, model_version, row = score_rule_pattern(canonical, behavioral)
    except (ChampionUnavailableError, ChampionUnsupportedError) as e:
        return ScoreResponse(reference=reference, status="FAILURE", error_message=str(e))
    except Exception as e:
        return ScoreResponse(reference=reference, status="FAILURE", error_message=f"scoring failed: {e}")

    # Empty/omitted fraud_types means "evaluate every supported type", per
    # the API spec - NOT "evaluate nothing" (see SS7ScoreRequest's
    # docstring; the old `if "SPAM_SMS" in request.fraud_types` check got
    # this backwards for the empty-list case).
    requested = set(request.fraud_types) if request.fraud_types else set(_SUPPORTED_FRAUD_TYPES)
    to_evaluate = [ft for ft in _SUPPORTED_FRAUD_TYPES if ft in requested]

    # deep_scan=False: stop at the first FRAUD prediction rather than
    # evaluating every remaining requested type (spec: "the AI/ML engine
    # MAY stop the evaluation" - a real short-circuit, not just a hint).
    # Only one fraud_type exists today (SPAM_SMS), so this loop never
    # actually has a second iteration to skip yet - structured as a loop
    # now so a future second fraud_type only needs its own scoring branch
    # added below, not a rewrite of this control flow.
    fraud_results: list[FraudPredictionResult] = []
    for fraud_type in to_evaluate:
        if fraud_type == "SPAM_SMS":
            prediction = "FRAUD" if probability >= _FRAUD_THRESHOLD else "NOT_FRAUD"

            reason_codes: list[str] = []
            feature_contributions: list[FeatureContribution] = []
            if prediction == "FRAUD":
                # Best-effort, same convention as anomaly_score below: a
                # missing champion/explainer degrades explainability, it
                # never turns a successful rule_pattern_score into
                # FAILURE. Only paid for on FRAUD - see module docstring.
                try:
                    contributions = explain_rule_pattern(canonical, row)
                except Exception as e:
                    print(f"reason-code explanation unavailable for {reference!r}: {e}")
                    contributions = []
                reason_codes = _reason_codes(row, cold_start, contributions)
                feature_contributions = [
                    FeatureContribution(feature=f, value=v, contribution=c)
                    for f, v, c in contributions[:_TOP_K_CONTRIBUTIONS]
                ]

            fraud_results.append(
                FraudPredictionResult(
                    fraud_type="SPAM_SMS",
                    prediction=prediction,
                    risk_score=round(probability * 100),
                    confidence=_confidence(cold_start, canonical.text_decode_failed),
                    reason_codes=reason_codes,
                    feature_contributions=feature_contributions,
                )
            )
        if not request.deep_scan and fraud_results and fraud_results[-1].prediction == "FRAUD":
            break

    recommended_action = "BLOCK" if any(r.prediction == "FRAUD" for r in fraud_results) else "PASS"

    # Independent of rule_pattern_score above - a failure here (no
    # promoted anomaly champion yet, no corpus for this source, etc.)
    # must NOT turn a successful rule_pattern_score result into
    # status=FAILURE; anomaly_score is surfaced for visibility, not a
    # required part of this response (see schemas.ScoreResponse's
    # anomaly_score docstring).
    try:
        anomaly_score, _, _ = score_anomaly(canonical, behavioral)
    except (AnomalyChampionUnavailableError, CorpusUnavailableError) as e:
        print(f"anomaly_score unavailable for {reference!r}: {e}")
        anomaly_score = None
    except Exception as e:
        print(f"anomaly_score failed for {reference!r}: {e}")
        anomaly_score = None

    processing_time_ms = int((time.perf_counter() - start) * 1000)

    return ScoreResponse(
        reference=reference,
        status="SUCCESS",
        recommended_action=recommended_action,
        processing_time_ms=processing_time_ms,
        model_version=model_version,
        fraud_results=fraud_results,
        anomaly_score=anomaly_score,
    )
