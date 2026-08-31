"""
FastAPI service - CLAUDE.md's "Next: 1. FastAPI service combining both
scores" step, THIS PASS covers rule_pattern_score only (see
serving/schemas.py's module docstring for why anomaly_score is a
deliberate follow-up).

ONE endpoint, TWO request parsers, ONE scoring path: SMPP and SS7 carry
genuinely different raw wire fields (serving/schemas.py's
SMPPTransaction/SS7Transaction), but CLAUDE.md is explicit that both must
map to the same canonical schema before touching any model ("SMPP and SS7
must map to the same canonical schema before shared ML", "Start with one
shared model; split by source only if segment evaluation justifies it") -
so protocol-specific parsing lives in serving/canonical.py, and everything
from there on (Feast lookup, scoring, response-building) is shared,
`source` carried through as a feature rather than a routing key.

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

risk_score/confidence/reason_codes below are explicit, DISCLOSED heuristics,
not a calibrated business threshold or a real explainability pass - LIME/
SHAP are CLAUDE.md's still-pending next steps (Current status: "Next: ...
2. LIME integration 3. SHAP integration/evaluation"). Revisit
_FRAUD_THRESHOLD/_confidence/_reason_codes once those land, rather than
treating these numbers as final.
"""
import time

from fastapi import FastAPI

from serving.canonical import map_smpp_transaction, map_ss7_transaction
from serving.feature_lookup import get_sender_features
from serving.schemas import (
    FraudPredictionResult,
    ScoreRequest,
    ScoreResponse,
    SMPPScoreRequest,
)
from serving.scoring import (
    BEHAVIORAL_COLS,
    ChampionUnavailableError,
    ChampionUnsupportedError,
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


def _reason_codes(row: dict, cold_start: bool) -> list[str]:
    """Cheap, rule-based stand-ins for real feature-contribution reason
    codes (architecture plan Section 7: "native feature importance /
    cheap contribution scores... fast enough for the inline path") -
    thresholds below are illustrative, not fit to data. Only populated
    for a FRAUD prediction, matching the response contract's own example
    (NOT_FRAUD -> reason_codes: [])."""
    codes = ["KNOWN_SPAM_PATTERN"]
    if row.get("sender_repeat_content_ratio_1hr", 0) >= 0.5:
        codes.append("REPEATED_CONTENT")
    if row.get("sender_msgs_last_1hr", 0) >= 50:
        codes.append("HIGH_SENDER_VELOCITY")
    if cold_start:
        codes.append("NEW_SENDER_LOW_HISTORY")
    if row.get("text_decode_failed"):
        codes.append("UNDECODABLE_CONTENT")
    return codes


@app.post("/v1/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    start = time.perf_counter()
    reference = request.transaction.reference

    canonical = (
        map_smpp_transaction(request.transaction)
        if isinstance(request, SMPPScoreRequest)
        else map_ss7_transaction(request.transaction)
    )

    try:
        behavioral = get_sender_features(canonical.sender_id, canonical.text)
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
            fraud_results.append(
                FraudPredictionResult(
                    fraud_type="SPAM_SMS",
                    prediction=prediction,
                    risk_score=round(probability * 100),
                    confidence=_confidence(cold_start, canonical.text_decode_failed),
                    reason_codes=_reason_codes(row, cold_start) if prediction == "FRAUD" else [],
                )
            )
        if not request.deep_scan and fraud_results and fraud_results[-1].prediction == "FRAUD":
            break

    recommended_action = "BLOCK" if any(r.prediction == "FRAUD" for r in fraud_results) else "PASS"
    processing_time_ms = int((time.perf_counter() - start) * 1000)

    return ScoreResponse(
        reference=reference,
        status="SUCCESS",
        recommended_action=recommended_action,
        processing_time_ms=processing_time_ms,
        model_version=model_version,
        fraud_results=fraud_results,
    )
