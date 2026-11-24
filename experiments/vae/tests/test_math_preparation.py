import json
from pathlib import Path
import tempfile
import unittest
import contextlib
import io

import pyarrow as pa
import pyarrow.parquet as pq

from vae.prepare_sentences import (
    assign_split, balanced_boxes, canonical, extract_answer, problem_id,
    protected_ranges, quality_issue, sentence_spans, prepare,
)


class PreparationTests(unittest.TestCase):
    def test_balanced_nested_fraction_box(self):
        text = r"We calculate 1/2. The answer is: $\boxed{\frac{1}{2}}$."
        parsed, error = extract_answer(text, "dart-math-hard")
        self.assertIsNone(error)
        self.assertEqual(parsed["reasoning"], "We calculate 1/2.")
        self.assertEqual(parsed["final_answer"], r"\frac{1}{2}")

    def test_double_escaped_box(self):
        parsed, error = extract_answer(r"We calculate zero. The answer is: $\\boxed{0}$", "dart-math-hard")
        self.assertIsNone(error)
        self.assertEqual(parsed["final_answer"], "0")

    def test_gsm_delimiter(self):
        parsed, error = extract_answer("A total of 2 + 3 = 5.\n#### 5", "gsm8k")
        self.assertIsNone(error)
        self.assertEqual(parsed["reasoning"], "A total of 2 + 3 = 5.")
        self.assertEqual(parsed["final_answer"], "5")

    def test_ambiguous_two_answers_rejected(self):
        for tail in ["$2$ $-2$", r"$\boxed{2}$ or $\boxed{3}$", "$2"]:
            parsed, error = extract_answer("Compute the value. The answer is: " + tail, "dart-math-hard")
            self.assertIsNone(parsed)
            self.assertIsNotNone(error)

    def test_unbalanced_box_and_nonterminal_box(self):
        self.assertIsNotNone(balanced_boxes(r"\boxed{\frac{1}{2}")[1])
        self.assertEqual(extract_answer(r"We find \boxed{2}, but continue solving.", "dart-math-hard")[1], "nonterminal_answer_box")

    def test_terminal_box_extraction(self):
        parsed, error = extract_answer(r"We calculate 1+1=2. Therefore $\boxed{2}$.", "dart-math-hard")
        self.assertIsNone(error)
        self.assertEqual(parsed["final_answer"], "2")

    def test_displayed_equation_with_linebreaks(self):
        text = "Compute the value.\n" + r"\[x=1.5 + 2." + "\n" + r"5 =4.0\]" + "\nHence x=4."
        spans = sentence_spans(text)
        pieces = [text[a:b] for a, b in spans]
        self.assertEqual(len(pieces), 3)
        self.assertIn("\n", pieces[1])
        self.assertEqual("".join("".join(p.split()) for p in pieces), "".join(text.split()))

    def test_decimal_abbreviation_currency_and_short_equation(self):
        text = "Use e.g. 1.25 as a value.\nShe earns $12 an hour.\nx=3\nThen stop."
        self.assertEqual([text[a:b] for a, b in sentence_spans(text)], ["Use e.g. 1.25 as a value.", "She earns $12 an hour.", "x=3", "Then stop."])

    def test_no_split_in_inline_math_or_calculator_annotation(self):
        for text in ["We use $x=1.5. x=2$ here. Next step.", "Compute <<1.5+2.5=4>> now. End.", r"Use \(x=1.5\) here. End.", "Compute $1.5 + 2.\n5 = 4$. End."]:
            ranges, _ = protected_ranges(text)
            spans = sentence_spans(text)
            for a, b in spans:
                for left, right in ranges:
                    self.assertFalse(left < a < right)
                    self.assertFalse(left < b < right)

    def test_gsm_currency_does_not_merge_steps(self):
        text = "Weng earns $<<12/60=0.2>>0.2 per minute.\nWorking 50 minutes, she earned $<<0.2*50=10>>10."
        self.assertEqual(len(sentence_spans(text)), 2)
        text = "She works for $18/hour so she earns $144.00.\nHer base pay is $720 + $270 = $990."
        self.assertEqual(len(sentence_spans(text)), 2)
        self.assertEqual(len(sentence_spans("Compute $3 + 4 = 7$.\nThen finish.")), 2)

    def test_sentence_after_display_without_space(self):
        for text in [r"First calculate \[x=2.\]Then finish.", r"First calculate $$x=2.$$Then finish.", r"First calculate $x=2.$ Then finish."]:
            pieces = [text[a:b] for a, b in sentence_spans(text)]
            self.assertEqual(len(pieces), 2)
            self.assertEqual(pieces[-1], "Then finish.")

    def test_question_split_coordinated_and_test_reserved(self):
        pid = problem_id("What is  2 + 3?")
        self.assertEqual(pid, problem_id("What is 2 + 3?"))
        self.assertEqual(assign_split(pid, set()), assign_split(pid, set()))
        self.assertEqual(assign_split(pid, {pid}), "test")
        self.assertNotEqual(canonical("x"), canonical("X"))
        self.assertNotEqual(canonical("x²"), canonical("x2"))

    def test_damage_and_equation_preservation(self):
        self.assertEqual(quality_issue("bad\ufffdtext"), "replacement_or_control_characters")


    def test_end_to_end_shared_question_and_sentence_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ["data/raw/gsm8k/main", "data/raw/dart-math-hard/data", "manifests"]:
                (root / name).mkdir(parents=True, exist_ok=True)
            training_question = next(f"Question {i}?" for i in range(100) if assign_split(problem_id(f"Question {i}?"), set()) == "train")
            test_question = "Official test question?"
            train = [{"question": training_question, "answer": "x=3.\nTraining equation is y=4.\n#### 4"}]
            test = [{"question": test_question, "answer": "x=3.\nTesting equation is z=5.\n#### 5"}]
            pq.write_table(pa.Table.from_pylist(train), root / "data/raw/gsm8k/main/train-00000-of-00001.parquet")
            pq.write_table(pa.Table.from_pylist(test), root / "data/raw/gsm8k/main/test-00000-of-00001.parquet")
            dart = [{"query": test_question, "response": "x=3.\nA second test equation is z=6. The answer is: 6"}]
            pq.write_table(pa.Table.from_pylist(dart), root / "data/raw/dart-math-hard/data/train-00000-of-00001.parquet")
            for name in ["gsm8k", "dart-math-hard"]:
                (root / "manifests" / f"{name}.json").write_text(json.dumps({"revision": "fixture", "files": []}))
            output = root / "prepared"
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(root, output)
            load = lambda path: [json.loads(line) for line in path.read_text().splitlines()]
            train_rows = load(output / "sentences/gsm8k.train.jsonl")
            test_rows = load(output / "sentences/gsm8k.test.jsonl")
            dart_test = load(output / "parents/dart-math-hard.test.jsonl")
            self.assertEqual(dart_test[0]["problem_id"], problem_id(test_question))
            self.assertNotIn("x=3.", [r["reasoning_block"] for r in train_rows])
            self.assertIn("x=3.", [r["reasoning_block"] for r in test_rows])
            self.assertIn("Training equation is y=4.", [r["reasoning_block"] for r in train_rows])


if __name__ == "__main__":
    unittest.main()
