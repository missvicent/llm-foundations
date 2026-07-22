from __future__ import annotations

import datetime as dt
from email import message
import json
from pathlib import Path

import instructor
from openai import OpenAI
from decimal import Decimal
from pydantic import BaseModel, field_validator
from normalize import vendor_matches, date_matches, total_matches


class Invoice(BaseModel):
    vendor: str
    date: str
    total: Decimal

    @field_validator("date")
    @classmethod
    def iso_date(cls, v: str) -> str:
        dt.datetime.fromisoformat(v)
        return v


def make_client(local: bool):
    if local:
        return instructor.from_openai(
            OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",
            ),
            mode=instructor.Mode.JSON,
        )
    return instructor.from_openai(OpenAI())


def extract(client, text: str, model: str) -> Invoice:
    return client.chat.completions.create(
        response_model=Invoice,
        max_retries=2,
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"Extract vendor, ISO date (YYYY-MM-DD) and numeric total.\n\n{text}",
            }
        ],
    )


def score(local: bool) -> dict:
    rows = [
        json.loads(line)
        for line in Path("../data/invoices.jsonl").read_text().splitlines()
    ]
    per_field = {"vendor": 0, "date": 0, "total": 0}
    mismatches = []

    for r in rows:
        out = extract(make_client(local), r["text"], "8b")

        check = {
            "vendor": vendor_matches(out.vendor, r["vendor"]),
            "date": date_matches(out.date, r["date"]),
            "total": total_matches(out.total, r["total"]),
        }

        for field, match in check.items():
            if not match:
                per_field[field] += 1
                mismatches.append((r["text"], field, out[field], r[field]))

        total_fields = 3 * len(rows)

        return {
            "correct": total_fields - sum(per_field.values()),
            "total": total_fields,
            "per_field_errors": per_field,
            "mismatches": mismatches[:10],
        }


if __name__ == "__main__":
    print(score(local=True))
