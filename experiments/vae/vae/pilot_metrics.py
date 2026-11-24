"""Free reconstruction metrics; numeric agreement is literal, not algebraic equivalence."""
import re


def normalized(text):
    return re.sub(r"\s+", " ", text).strip()


def numeric_strings(text):
    """Ordered literal numeric strings; no algebraic equivalence is implied."""
    return re.findall(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)", text)


def reconstruction_values(records):
    total = len(records)
    exact = sum(bool(row["exact_match"]) for row in records)
    numeric = [row for row in records if numeric_strings(row["target"])]
    number_correct = sum(numeric_strings(row["target"]) == numeric_strings(row["text"]) for row in numeric)
    return {"exact_count": exact, "total": total, "exact_match_rate": exact / max(total, 1),
            "numeric_count": number_correct, "numeric_total": len(numeric),
            "numeric_match_rate": number_correct / len(numeric) if numeric else None,
            "eos_count": sum(row["eos_reached"] for row in records),
            "token_limit_count": sum(row["token_limit_reached"] for row in records)}
