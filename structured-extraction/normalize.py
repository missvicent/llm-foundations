from __future__ import annotations
import datetime as dt
from rapidfuzz import fuzz

_DATE_FORMATS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%m-%d-%Y",
    "%B %d, %Y",
    "%d %B %Y",
)


def vendor_matches(a: str, b: str, threshold: int = 88) -> bool:
    """
    token_set_ratio ignores word order and extra tokens
    """
    a = (a or "").lower().strip()
    b = (b or "").lower().strip()
    return fuzz.token_set_ratio(a, b) >= threshold


def _parse_date(text: str) -> dt.date:
    text = (text or "").strip()
    for format in _DATE_FORMATS:
        try:
            return dt.datetime.strptime(text, format).date()
        except ValueError:
            continue
    raise ValueError(f"Could not parse date: {text}")


def date_matches(date: str, text: str, tolerance_days: int = 0) -> bool:
    """
    Canonicalize both to date objects, then compare.
    tolerance_days=0 → exact. Set to 1 if you want to accept off-by-one.
    """
    return abs((_parse_date(date) - _parse_date(text)).days) <= tolerance_days


def _parse_total(val: str | int | float) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    # Strip currency symbols and spaces
    s = "".join(c for c in s if c.isdigit() or c in ".,-")
    # EU-Style "1.234,56" → "1234.56"
    if "," in s and s.rfind(",") > s.rfind("."):
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", "")
    return float(s)


def total_matches(
    a: str, b: str, abs_tol: float = 0.01, rel_tol: float = 0.001
) -> bool:
    """
    Match if within 1 cent or 0.1% of the larger value
    Handles both small invoices and large ones.
    """
    a, b = _parse_total(a), _parse_total(b)
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b)))
