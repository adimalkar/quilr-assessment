"""In-memory mock customer store used by the Task 1 server."""

from __future__ import annotations

import itertools
import threading
from dataclasses import asdict, dataclass, field


@dataclass
class Customer:
    customer_id: str
    name: str
    email: str
    tier: str
    lifetime_spend: float
    refunded_total: float = 0.0

    @property
    def refundable_balance(self) -> float:
        return round(self.lifetime_spend - self.refunded_total, 2)

    def to_public_dict(self) -> dict:
        data = asdict(self)
        data["refundable_balance"] = self.refundable_balance
        return data


@dataclass
class CustomerStore:
    customers: dict[str, Customer] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _refund_seq: itertools.count = field(default_factory=lambda: itertools.count(1), repr=False)
    # idempotency_key -> (fingerprint of the arguments, stored receipt dict)
    _idempotency: dict[str, tuple[tuple, dict]] = field(default_factory=dict, repr=False)

    def get(self, customer_id: str) -> Customer | None:
        return self.customers.get(customer_id)

    def idempotent(self, key: str, fingerprint: tuple, make_receipt) -> tuple[dict, bool]:
        """Run ``make_receipt`` at most once per key. Returns (receipt, replayed).
        Raises KeyError if the key was already used with different arguments."""
        with self._lock:  # re-entrant: refund() inside make_receipt takes it again
            if key in self._idempotency:
                seen_fp, receipt = self._idempotency[key]
                if seen_fp != fingerprint:
                    raise KeyError(key)
                return receipt, True
            receipt = make_receipt()  # check-and-issue is one critical section: no double refund
            self._idempotency[key] = (fingerprint, receipt)
            return receipt, False

    def refund(self, customer_id: str, amount: float) -> tuple[str, Customer]:
        """Apply a refund atomically. Raises LookupError / ValueError on business-rule violations."""
        with self._lock:
            customer = self.customers.get(customer_id)
            if customer is None:
                raise LookupError(f"Customer {customer_id} not found")
            if amount > customer.refundable_balance:
                raise ValueError(
                    f"Refund of {amount:.2f} exceeds refundable balance of {customer.refundable_balance:.2f}"
                )
            customer.refunded_total = round(customer.refunded_total + amount, 2)
            return f"RFD-{next(self._refund_seq):06d}", customer


def default_store() -> CustomerStore:
    return CustomerStore(
        customers={
            c.customer_id: c
            for c in [
                Customer("CUST-10001", "Ada Lovelace", "ada@example.com", "gold", 1250.00),
                Customer("CUST-10002", "Alan Turing", "alan@example.com", "silver", 310.50),
                Customer("CUST-10003", "Grace Hopper", "grace@example.com", "platinum", 9800.00),
            ]
        }
    )
