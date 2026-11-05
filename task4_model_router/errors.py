"""Standardised gateway error envelope.

Shape is OpenAI-compatible so existing SDK clients parse it natively::

    {"error": {"message": "...", "type": "rate_limit_error", "code": "tenant_token_limit_exceeded",
               "param": null, "request_id": "req_..."}}

Rules: messages are written by the gateway, never copied from upstream; upstream
bodies/headers, exception text and stack traces are logged server-side only.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("task4.errors")


class GatewayError(Exception):
    def __init__(
        self,
        status: int,
        err_type: str,
        code: str,
        message: str,
        *,
        param: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status, self.err_type, self.code, self.message, self.param = status, err_type, code, message, param
        self.headers = headers or {}


def error_body(err_type: str, code: str, message: str, request_id: str, param: str | None = None) -> dict:
    return {"error": {"message": message, "type": err_type, "code": code, "param": param, "request_id": request_id}}


def _rid(request: Request) -> str:
    return getattr(request.state, "request_id", None) or "req_unknown"


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(GatewayError)
    async def _gateway(request: Request, exc: GatewayError) -> JSONResponse:
        headers = {"x-request-id": _rid(request), **exc.headers}
        return JSONResponse(
            error_body(exc.err_type, exc.code, exc.message, _rid(request), exc.param), status_code=exc.status, headers=headers
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        loc = [str(p) for p in first.get("loc", []) if p != "body"]
        message = f"Invalid request: {'.'.join(loc) or 'body'}: {first.get('msg', 'invalid value')}"
        return JSONResponse(
            error_body("invalid_request_error", "invalid_request", message, _rid(request), ".".join(loc) or None),
            status_code=400,
            headers={"x-request-id": _rid(request)},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return JSONResponse(
            error_body("invalid_request_error", code, "Request could not be served", _rid(request)),
            status_code=exc.status_code,
            headers={"x-request-id": _rid(request)},
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error rid=%s", _rid(request))  # full traceback stays server-side
        return JSONResponse(
            error_body("internal_error", "internal_error", "An internal gateway error occurred", _rid(request)),
            status_code=500,
            headers={"x-request-id": _rid(request)},
        )
