"""Explicit token-weighted teacher-forcing and free reconstruction metrics."""
from dataclasses import dataclass
import re

from .reconstruction import normalized


@dataclass
class TokenMetrics:
    ce_sum: float = 0.0
    kl_sum: float = 0.0
    tokens: int = 0
    correct: int = 0
    examples: int = 0

    def add(self, output, batch_size):
        count = int(output["n_tokens"])
        self.ce_sum += float(output["ce_loss"].detach()) * count
        self.kl_sum += float(output["kl_loss"].detach()) * batch_size
        self.tokens += count
        self.correct += int(output["n_correct"])
        self.examples += batch_size

    def values(self, kl_weight):
        ce = self.ce_sum / max(self.tokens, 1)
        kl = self.kl_sum / max(self.examples, 1)
        return {"ce_loss": ce, "kl_loss": kl, "loss": ce + kl_weight * kl,
                "token_accuracy": self.correct / max(self.tokens, 1),
                "tokens": self.tokens, "examples": self.examples}


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
