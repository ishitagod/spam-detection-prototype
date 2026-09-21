"""
pytest suite for serving/app.py's POST /v1/score/smpp and /v1/score/ss7
endpoints - the response-contract shaping (status/recommended_action/
fraud_results/reason_codes), not real model/Feast integration. Both routes
share the same _score() body, so most cases below exercise it via whichever
route matches the payload's protocol; only the request-shape/routing checks
near the bottom care about hitting the "wrong" route on purpose.
serving.app.get_sender_features,
serving.app.get_imsi_features, and serving.app.score_rule_pattern are all
monkeypatched so this runs without a promoted MLflow champion or an
applied Feast store - both are exercised separately
(tests/test_serving_scoring.py, serving/feature_lookup.py's own manual
script).

Run:
    pytest tests/test_serving_app.py -v
"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import serving.app as app_module
from serving.scoring import ChampionUnavailableError

SS7_PAYLOAD = {
    "fraud_types": ["SPAM_SMS", "SMISHING"],
    "deep_scan": True,
    "protocol": "SS7",
    "transaction": {
        "reference": "34993520154",
        "time_stamp": 1683038843499,
        "message_type": 3,
        "calling_gt": "60944123210002",
        "called_gt": "911234500002",
        "a_number": "99645634751",
        "b_number": "222",
        "content": "d9 77 5d 0e 7a 52 a1 a0 f4 1c 54 a3 e1 64 b1 19",
        "dcs": "0",
        "pid": "0",
        "imsi": "1040023547528",
        "service_centre_address": "944123210003",
        "service_centre_time_stamp": "23-05-02 20:17:23",
        "tpdu_length": 43,
        "sarref": 115,
        "msg_part": 1,
        "msg_parts": 2,
    },
}

SMPP_PAYLOAD = {
    "fraud_types": ["SPAM_SMS", "SMISHING"],
    "deep_scan": True,
    "protocol": "SMPP",
    "transaction": {
        "reference": "34993520433",
        "time_stamp": "2026-08-20T10:47:38.000Z",
        "smpp_operation": "4",
        "system_id": "ESME_SYS_01",
        "source_ip": "192.168.1.10",
        "a_number": "60123456789",
        "oa_ton": "1", "oa_npi": "1",
        "b_number": "60198765432",
        "da_ton": "1", "da_npi": "1",
        "content": "48 65 6c 6c 6f 20 57 6f 72 6c 64",
        "dcs": "0",
        "esm_class": "0", "msg_type": "0", "gsm_features": "0", "messaging_mode": "0",
        "sar_ref": None, "sar_msg_parts": None, "sar_msg_part": None,
        "app_dest_port": None, "app_src_port": None,
    },
}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        app_module, "get_sender_features",
        lambda sender_id, candidate_text: {
            "sender_msgs_last_5min": 1, "sender_msgs_last_1hr": 5,
            "sender_unique_destinations_1hr": 3, "sender_repeat_content_ratio_1hr": 0.1,
        },
    )
    monkeypatch.setattr(
        app_module, "get_imsi_features",
        lambda imsi: {"imsi_distinct_originators_1hr": 2 if imsi else None},
    )
    return TestClient(app_module.app)


def _mock_score(probability: float):
    def fake(canonical, behavioral):
        return probability, "rule_pattern_score_model/v3", {
            "sender_msgs_last_1hr": behavioral["sender_msgs_last_1hr"],
            "sender_repeat_content_ratio_1hr": behavioral["sender_repeat_content_ratio_1hr"],
            "text_decode_failed": int(canonical.text_decode_failed),
        }
    return fake


def test_high_probability_returns_fraud_block(client, monkeypatch):
    monkeypatch.setattr(app_module, "score_rule_pattern", _mock_score(0.95))
    resp = client.post("/v1/score/ss7", json=SS7_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()

    assert body["status"] == "SUCCESS"
    assert body["reference"] == "34993520154"
    assert body["recommended_action"] == "BLOCK"
    assert body["model_version"] == "rule_pattern_score_model/v3"
    assert len(body["fraud_results"]) == 1  # SMISHING requested but not modeled - omitted
    result = body["fraud_results"][0]
    assert result["fraud_type"] == "SPAM_SMS"
    assert result["prediction"] == "FRAUD"
    assert result["risk_score"] == 95
    assert "KNOWN_SPAM_PATTERN" in result["reason_codes"]


def test_low_probability_returns_not_fraud_pass(client, monkeypatch):
    monkeypatch.setattr(app_module, "score_rule_pattern", _mock_score(0.05))
    resp = client.post("/v1/score/smpp", json=SMPP_PAYLOAD)
    body = resp.json()

    assert body["recommended_action"] == "PASS"
    result = body["fraud_results"][0]
    assert result["prediction"] == "NOT_FRAUD"
    assert result["risk_score"] == 5
    assert result["reason_codes"] == []


def test_smishing_only_request_returns_empty_fraud_results(client, monkeypatch):
    monkeypatch.setattr(app_module, "score_rule_pattern", _mock_score(0.9))
    payload = {**SS7_PAYLOAD, "fraud_types": ["SMISHING"]}
    resp = client.post("/v1/score/ss7", json=payload)
    body = resp.json()

    assert body["status"] == "SUCCESS"
    assert body["fraud_results"] == []
    assert body["recommended_action"] == "PASS"  # no FRAUD result present


def test_empty_fraud_types_evaluates_all_supported_not_nothing(client, monkeypatch):
    """Spec: omitted/empty fraud_types means 'evaluate all supported
    types', NOT 'evaluate nothing' - a real bug this fixes (the old code
    read `"SPAM_SMS" in request.fraud_types`, which is False for an empty
    list)."""
    monkeypatch.setattr(app_module, "score_rule_pattern", _mock_score(0.9))
    payload = {**SS7_PAYLOAD, "fraud_types": []}
    resp = client.post("/v1/score/ss7", json=payload)
    body = resp.json()

    assert body["status"] == "SUCCESS"
    assert len(body["fraud_results"]) == 1
    assert body["fraud_results"][0]["fraud_type"] == "SPAM_SMS"


def test_fusion_escalates_low_rule_pattern_score_to_fraud(client, monkeypatch):
    """rule_pattern_score alone (0.1) is below _FRAUD_THRESHOLD, but a
    strongly agreeing anomaly_score pushes the FUSED decision over it -
    ANOMALY_SIGNAL_ESCALATION must appear, and recommended_action must be
    BLOCK even though rule_pattern_score alone would have said PASS."""
    monkeypatch.setattr(app_module, "score_rule_pattern", _mock_score(0.1))
    monkeypatch.setattr(
        app_module, "score_anomaly",
        lambda canonical, behavioral: (0.9, "anomaly_SS7/v1", {}),
    )
    monkeypatch.setattr(
        app_module, "score_fusion",
        lambda source, rule_pattern_score, anomaly_score: (0.8, "decision_fusion_model_SS7/v1"),
    )
    resp = client.post("/v1/score/ss7", json=SS7_PAYLOAD)
    body = resp.json()

    assert body["anomaly_score"] == pytest.approx(0.9)
    assert body["fusion_score"] == pytest.approx(0.8)
    assert body["recommended_action"] == "BLOCK"
    result = body["fraud_results"][0]
    assert result["prediction"] == "FRAUD"
    assert result["risk_score"] == 10  # unchanged - still rule_pattern_score-based
    assert "ANOMALY_SIGNAL_ESCALATION" in result["reason_codes"]


def test_fusion_agreement_does_not_add_escalation_code(client, monkeypatch):
    """Fusion available and agrees with rule_pattern_score (both already
    say FRAUD) - no escalation happened, so the code must not fire."""
    monkeypatch.setattr(app_module, "score_rule_pattern", _mock_score(0.95))
    monkeypatch.setattr(
        app_module, "score_anomaly",
        lambda canonical, behavioral: (0.9, "anomaly_SS7/v1", {}),
    )
    monkeypatch.setattr(
        app_module, "score_fusion",
        lambda source, rule_pattern_score, anomaly_score: (0.97, "decision_fusion_model_SS7/v1"),
    )
    resp = client.post("/v1/score/ss7", json=SS7_PAYLOAD)
    body = resp.json()

    result = body["fraud_results"][0]
    assert result["prediction"] == "FRAUD"
    assert "ANOMALY_SIGNAL_ESCALATION" not in result["reason_codes"]


def test_no_fusion_champion_falls_back_to_rule_pattern_score_alone(client, monkeypatch):
    """No fusion champion for this source yet - fusion_score is None on
    the response and the decision is identical to pre-fusion behavior."""
    from serving.fusion_scoring import ChampionUnavailableError as FusionChampionUnavailableError

    monkeypatch.setattr(app_module, "score_rule_pattern", _mock_score(0.05))
    monkeypatch.setattr(
        app_module, "score_anomaly",
        lambda canonical, behavioral: (0.99, "anomaly_SS7/v1", {}),
    )

    def raise_unavailable(source, rule_pattern_score, anomaly_score):
        raise FusionChampionUnavailableError("no fusion champion promoted yet")

    monkeypatch.setattr(app_module, "score_fusion", raise_unavailable)
    resp = client.post("/v1/score/ss7", json=SS7_PAYLOAD)
    body = resp.json()

    assert body["fusion_score"] is None
    assert body["anomaly_score"] == pytest.approx(0.99)  # still surfaced despite fusion failing
    result = body["fraud_results"][0]
    assert result["prediction"] == "NOT_FRAUD"  # rule_pattern_score (0.05) alone, unaffected


def test_champion_unavailable_returns_failure_status_not_http_error(client, monkeypatch):
    def raise_unavailable(canonical, behavioral):
        raise ChampionUnavailableError("no champion promoted yet")

    monkeypatch.setattr(app_module, "score_rule_pattern", raise_unavailable)
    resp = client.post("/v1/score/ss7", json=SS7_PAYLOAD)

    assert resp.status_code == 200  # status/error_message carry outcome, not the HTTP line
    body = resp.json()
    assert body["status"] == "FAILURE"
    assert "no champion promoted yet" in body["error_message"]
    assert body["fraud_results"] is None


def test_unknown_protocol_is_rejected_by_request_validation(client):
    bad_payload = {**SS7_PAYLOAD, "protocol": "XYZ"}
    resp = client.post("/v1/score/ss7", json=bad_payload)
    assert resp.status_code == 422


def test_smpp_payload_rejected_on_ss7_route(client):
    """The two routes are now separate request shapes, not one
    discriminated union - posting an SMPP transaction to /v1/score/ss7 (or
    vice versa) must fail request validation, not silently mis-route."""
    resp = client.post("/v1/score/ss7", json=SMPP_PAYLOAD)
    assert resp.status_code == 422


def test_ss7_payload_rejected_on_smpp_route(client):
    resp = client.post("/v1/score/smpp", json=SS7_PAYLOAD)
    assert resp.status_code == 422
