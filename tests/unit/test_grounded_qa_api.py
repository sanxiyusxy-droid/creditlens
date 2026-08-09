"""Grounded QA HTTP request validation tests."""

import uuid

import httpx
import pytest
from apps.api import main as api_main

VALID_IDEMPOTENCY_KEY = "qa-request-key-001"


async def _post_question(payload: dict) -> httpx.Response:
    case_id = uuid.uuid4()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_main.app),
        base_url="http://test",
    ) as client:
        return await client.post(f"/api/v1/cases/{case_id}/questions", json=payload)


@pytest.mark.parametrize(
    ("payload", "invalid_field"),
    [
        ({"question": "   ", "idempotency_key": VALID_IDEMPOTENCY_KEY}, "question"),
        (
            {
                "question": "valid question",
                "top_k": 0,
                "idempotency_key": VALID_IDEMPOTENCY_KEY,
            },
            "top_k",
        ),
        (
            {
                "question": "valid question",
                "top_k": 21,
                "idempotency_key": VALID_IDEMPOTENCY_KEY,
            },
            "top_k",
        ),
        (
            {
                "question": "valid question",
                "decision_cutoff_at": "2026-06-30T12:00:00",
                "idempotency_key": VALID_IDEMPOTENCY_KEY,
            },
            "decision_cutoff_at",
        ),
    ],
    ids=["blank-question", "top-k-too-small", "top-k-too-large", "naive-cutoff"],
)
async def test_question_request_rejects_invalid_input_with_422(payload, invalid_field):
    response = await _post_question(payload)

    assert response.status_code == 422
    assert any(error["loc"][-1] == invalid_field for error in response.json()["detail"])


async def test_question_request_rejects_missing_idempotency_key_with_422():
    response = await _post_question({"question": "valid question"})

    assert response.status_code == 422
    assert any(error["loc"][-1] == "idempotency_key" for error in response.json()["detail"])


async def test_question_request_rejects_too_short_idempotency_key_with_422():
    response = await _post_question({"question": "valid question", "idempotency_key": "short"})

    assert response.status_code == 422
    assert any(error["loc"][-1] == "idempotency_key" for error in response.json()["detail"])


async def test_valid_request_forwards_idempotency_key_unchanged(monkeypatch):
    captured: dict = {}

    class _FakeResult:
        @staticmethod
        def model_dump(*, mode: str):
            assert mode == "json"
            return {"accepted": True}

    class _FakeQAService:
        def __init__(self, **_kwargs):
            pass

        async def ask(self, **kwargs):
            captured.update(kwargs)
            return _FakeResult()

    monkeypatch.setattr(api_main, "QAService", _FakeQAService)
    idempotency_key = "CaseSensitive-Key_2026-001"

    response = await _post_question(
        {
            "question": "申请材料是否齐全？",
            "top_k": 6,
            "idempotency_key": idempotency_key,
        }
    )

    assert response.status_code == 200
    assert response.json() == {"accepted": True}
    assert captured["idempotency_key"] == idempotency_key
