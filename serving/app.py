"""
FastAPI service - CLAUDE.md's "Next: 1. FastAPI service combining both
scores" step. Three scores are computed: rule_pattern_score
(serving/scoring.py), anomaly_score (serving/anomaly_scoring.py), and -
when a decision-fusion champion is available for this source -
fusion_score (serving/fusion_scoring.py), a small trained model over the
first two. See serving/schemas.py's module docstring for the SCOPE note.

DECISION FUSION, how it gates the decision without violating CLAUDE.md's
"no averaging, keep disagreement visible":
  1. rule_pattern_score and anomaly_score are always both computed and
     always both surfaced unchanged (risk_score still derives from
     rule_pattern_score; anomaly_score is its own field) - fusion never
     overwrites either.
  2. anomaly_score is computed before the fraud_type loop now (fusion
     needs it for the decision, not just display) - same best-effort
     try/except as before.
  3. If a fusion champion exists for canonical.source and anomaly_score
     succeeded, fusion_score drives `prediction`/`recommended_action`
     (same _FRAUD_THRESHOLD). Otherwise prediction falls back to
     rule_pattern_score alone - identical to pre-fusion behavior.
  4. When fusion is the reason a row is FRAUD (rule_pattern_score alone
     was below threshold, anomaly_score pushed it over), `_reason_codes()`
     adds ANOMALY_SIGNAL_ESCALATION - see that function's docstring for
     why the opposite direction isn't reason-coded.

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
import logging
import time

from fastapi import FastAPI

from serving.anomaly_scoring import score_anomaly
from serving.anomaly_scoring import ChampionUnavailableError as AnomalyChampionUnavailableError
from serving.anomaly_scoring import CorpusUnavailableError
from serving.canonical import CanonicalRow, map_smpp_transaction, map_ss7_transaction
from serving.feature_lookup import get_imsi_features, get_sender_features
from serving.fusion_scoring import score_fusion
from serving.fusion_scoring import ChampionUnavailableError as FusionChampionUnavailableError
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

# One basicConfig call, here rather than per-module - this is the process
# entrypoint (uvicorn imports this module once). Stage-by-stage logs below
# exist so a running server's terminal shows which algorithm/stage is
# executing per request, not just print()-only failure paths (the prior
# convention). Level is INFO by default - override via
# `logging.getLogger("serving").setLevel(...)` or the LOG_LEVEL env var if
# ever wired through config/settings.py.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

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
    fusion_delta: str | None = None,
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

    `fusion_delta`: "escalated" when fusion_score is the reason this row is
    FRAUD at all (rule_pattern_score alone was below threshold, anomaly_score
    pushed it over). None on agreement, or when no fusion champion exists
    yet. The opposite direction (fusion suppressing a would-be FRAUD call)
    isn't reason-coded - reason_codes are FRAUD-only, same cost-control
    convention as this function's other codes; still visible via the raw
    risk_score/anomaly_score/fusion_score fields.

    TODO: these codes are still OUR OWN placeholder set, not validated
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
    if fusion_delta == "escalated":
        codes.append("ANOMALY_SIGNAL_ESCALATION")
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
    logger.info("[%s] score request received (source=%s)", reference, canonical.source)

    stage_start = time.perf_counter()
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
        logger.error("[%s] Feast behavioral lookup failed: %s", reference, e)
        return ScoreResponse(
            reference=reference, status="FAILURE",
            error_message=f"behavioral feature lookup failed: {e}",
        )
    cold_start = all(behavioral.get(c) is None for c in BEHAVIORAL_COLS)
    logger.info(
        "[%s] Feast lookup done in %dms (cold_start=%s)",
        reference, int((time.perf_counter() - stage_start) * 1000), cold_start,
    )

    stage_start = time.perf_counter()
    try:
        probability, model_version, row = score_rule_pattern(canonical, behavioral)
    except (ChampionUnavailableError, ChampionUnsupportedError) as e:
        logger.error("[%s] rule_pattern_score (LightGBM) unavailable: %s", reference, e)
        return ScoreResponse(reference=reference, status="FAILURE", error_message=str(e))
    except Exception as e:
        logger.error("[%s] rule_pattern_score (LightGBM) failed: %s", reference, e)
        return ScoreResponse(reference=reference, status="FAILURE", error_message=f"scoring failed: {e}")
    logger.info(
        "[%s] rule_pattern_score (LightGBM v%s) = %.4f in %dms",
        reference, model_version, probability, int((time.perf_counter() - stage_start) * 1000),
    )

    # Computed ahead of the fraud_type loop now - fusion needs it to make
    # the decision, not just display it. Best-effort, same as before.
    stage_start = time.perf_counter()
    try:
        anomaly_score, _, _ = score_anomaly(canonical, behavioral)
        logger.info(
            "[%s] anomaly_score (IsolationForest + FAISS near-dup) = %.4f in %dms",
            reference, anomaly_score, int((time.perf_counter() - stage_start) * 1000),
        )
    except (AnomalyChampionUnavailableError, CorpusUnavailableError) as e:
        logger.warning("[%s] anomaly_score unavailable: %s", reference, e)
        anomaly_score = None
    except Exception as e:
        logger.warning("[%s] anomaly_score failed: %s", reference, e)
        anomaly_score = None

    # Only runs if anomaly_score succeeded and a fusion champion exists for
    # this source. `decision_score` drives prediction below; `fusion_score`
    # (possibly None) is surfaced on the response.
    fusion_score: float | None = None
    if anomaly_score is not None:
        stage_start = time.perf_counter()
        try:
            fusion_score, fusion_version = score_fusion(canonical.source, probability, anomaly_score)
            logger.info(
                "[%s] fusion_score (decision_fusion v%s) = %.4f in %dms",
                reference, fusion_version, fusion_score, int((time.perf_counter() - stage_start) * 1000),
            )
        except FusionChampionUnavailableError as e:
            logger.info("[%s] fusion_score unavailable, falling back to rule_pattern_score alone: %s", reference, e)
        except Exception as e:
            logger.warning("[%s] fusion_score failed, falling back to rule_pattern_score alone: %s", reference, e)
    decision_score = fusion_score if fusion_score is not None else probability
    fusion_escalated = (
        fusion_score is not None
        and decision_score >= _FRAUD_THRESHOLD
        and probability < _FRAUD_THRESHOLD
    )

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
            prediction = "FRAUD" if decision_score >= _FRAUD_THRESHOLD else "NOT_FRAUD"

            reason_codes: list[str] = []
            feature_contributions: list[FeatureContribution] = []
            if prediction == "FRAUD":
                # Best-effort, same convention as anomaly_score below: a
                # missing champion/explainer degrades explainability, it
                # never turns a successful rule_pattern_score into
                # FAILURE. Only paid for on FRAUD - see module docstring.
                explain_start = time.perf_counter()
                try:
                    contributions = explain_rule_pattern(canonical, row)
                    logger.info(
                        "[%s] explain_rule_pattern (shap.TreeExplainer) done in %dms, top feature=%s",
                        reference, int((time.perf_counter() - explain_start) * 1000),
                        contributions[0][0] if contributions else None,
                    )
                except Exception as e:
                    logger.warning("[%s] reason-code explanation unavailable: %s", reference, e)
                    contributions = []
                reason_codes = _reason_codes(
                    row, cold_start, contributions,
                    fusion_delta="escalated" if fusion_escalated else None,
                )
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

    processing_time_ms = int((time.perf_counter() - start) * 1000)
    logger.info(
        "[%s] scored in %dms -> prediction=%s recommended_action=%s",
        reference, processing_time_ms,
        fraud_results[0].prediction if fraud_results else None, recommended_action,
    )

    return ScoreResponse(
        reference=reference,
        status="SUCCESS",
        recommended_action=recommended_action,
        processing_time_ms=processing_time_ms,
        model_version=model_version,
        fraud_results=fraud_results,
        anomaly_score=anomaly_score,
        fusion_score=fusion_score,
    )
