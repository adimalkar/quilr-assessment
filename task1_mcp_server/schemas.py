"""Strict Pydantic input schemas for the Task 1 MCP tools.

These models are the single source of truth: the JSON Schema advertised in
``tools/list`` is generated from them, and the same models validate every
``tools/call`` before any handler logic runs.
"""

from __future__ import annotations

import unicodedata
from decimal import Decimal
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

# `CUST-` followed by exactly five digits. Pydantic compiles patterns with the
# Rust regex engine, where `$` is end-of-input (a trailing "\n" is rejected,
# unlike Python's `re` where `$` also matches before a final newline).
CUSTOMER_ID_PATTERN = r"^CUST-[0-9]{5}$"

CustomerId = Annotated[
    str,
    StringConstraints(pattern=CUSTOMER_ID_PATTERN, min_length=10, max_length=10),
    Field(description="Customer identifier formatted as CUST-XXXXX (five digits).", examples=["CUST-10001"]),
]

MAX_REFUND_AMOUNT = 1_000_000.0


class _StrictModel(BaseModel):
    # strict: no type coercion ("12.5" is not a float, 1 is not a str, True is not a number)
    # extra=forbid: unknown argument keys are a validation error, not silently dropped
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)


class GetCustomerRecordInput(_StrictModel):
    customer_id: CustomerId


class TriggerRefundInput(_StrictModel):
    customer_id: CustomerId
    amount: Annotated[
        float,
        Field(
            gt=0,
            le=MAX_REFUND_AMOUNT,
            description="Refund amount in account currency. Positive, at most 2 decimal places.",
            examples=[49.99],
        ),
    ]
    reason: Annotated[
        str,
        # Whitespace is stripped *before* the length check, so ten spaces is not a reason.
        StringConstraints(strip_whitespace=True, min_length=10, max_length=500),
        Field(description="Human-readable justification (10-500 characters).", examples=["Duplicate charge on invoice"]),
    ]

    @field_validator("amount")
    @classmethod
    def _at_most_two_decimals(cls, value: float) -> float:
        # Decimal(str(x)) gives the shortest round-tripping repr, so 0.1 -> "0.1" (not 0.1000000000000000055...).
        exponent = Decimal(str(value)).as_tuple().exponent
        if isinstance(exponent, int) and exponent < -2:
            raise ValueError("amount must have at most 2 decimal places")
        return value

    @field_validator("reason")
    @classmethod
    def _printable_reason(cls, value: str) -> str:
        # Zero-width (Cf) and control (Cc) characters would let "\u200b"*10 or "\x00"*10 pass the
        # length check while saying nothing; newlines/tabs are fine in a free-text reason.
        bad = [ch for ch in value if unicodedata.category(ch) in ("Cc", "Cf") and ch not in "\n\t"]
        if bad:
            raise ValueError("reason must not contain control or invisible formatting characters")
        if sum(not ch.isspace() for ch in value) < 10:
            raise ValueError("reason must contain at least 10 non-whitespace characters")
        return value


# ---------------------------------------------------------------- output schemas (advertised as outputSchema)
class CustomerRecord(BaseModel):
    customer_id: str
    name: str
    email: str
    tier: str
    lifetime_spend: float
    refunded_total: float
    refundable_balance: float


class RefundReceipt(BaseModel):
    refund_id: str
    customer_id: str
    amount: float
    reason: str
    status: str
    remaining_refundable_balance: float


TOOL_OUTPUT_MODELS: dict[str, type[BaseModel]] = {
    "get_customer_record": CustomerRecord,
    "trigger_refund": RefundReceipt,
}


TOOL_INPUT_MODELS: dict[str, type[_StrictModel]] = {
    "get_customer_record": GetCustomerRecordInput,
    "trigger_refund": TriggerRefundInput,
}
