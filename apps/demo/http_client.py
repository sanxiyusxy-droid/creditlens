"""Small fail-closed HTTP boundary for the local interview demo.

The UI must never render response bodies or raw request exceptions: either can
contain provider, proxy or local path details.  Callers receive only stable
error codes and an optional HTTP status.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True, slots=True)
class DemoHTTPError(Exception):
    code: str
    status_code: int | None = None


def _checked_response(response: requests.Response) -> requests.Response:
    if not 200 <= int(response.status_code) < 300:
        raise DemoHTTPError("API_HTTP_ERROR", int(response.status_code))
    return response


def get_json(url: str, *, params: dict[str, Any] | None = None, timeout: float = 120) -> dict:
    try:
        response = _checked_response(requests.get(url, params=params, timeout=timeout))
        payload = response.json()
    except DemoHTTPError:
        raise
    except requests.RequestException as exc:
        raise DemoHTTPError("API_UNREACHABLE") from exc
    except ValueError as exc:
        raise DemoHTTPError("API_RESPONSE_INVALID") from exc
    if not isinstance(payload, dict):
        raise DemoHTTPError("API_RESPONSE_INVALID")
    return payload


def post_json(url: str, *, payload: dict[str, Any], timeout: float = 120) -> dict:
    try:
        response = _checked_response(requests.post(url, json=payload, timeout=timeout))
        result = response.json()
    except DemoHTTPError:
        raise
    except requests.RequestException as exc:
        raise DemoHTTPError("API_UNREACHABLE") from exc
    except ValueError as exc:
        raise DemoHTTPError("API_RESPONSE_INVALID") from exc
    if not isinstance(result, dict):
        raise DemoHTTPError("API_RESPONSE_INVALID")
    return result


def get_binary(url: str, *, params: dict[str, Any], timeout: float = 60) -> bytes:
    try:
        return _checked_response(requests.get(url, params=params, timeout=timeout)).content
    except DemoHTTPError:
        raise
    except requests.RequestException as exc:
        raise DemoHTTPError("API_UNREACHABLE") from exc
