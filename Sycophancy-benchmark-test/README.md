# LLaDA evaluation with ELEPHANT

- [`elephant/`](elephant/): the benchmark repository, with local scorer fixes and its original sample files preserved.
- [`LLaDA/inference-test/`](LLaDA/inference-test/): the initial GPU inference smoke test.
- [`LLaDA/elephant-benchmark/`](LLaDA/elephant-benchmark/): experiment code, validated datasets, results, reports, and resumable run instructions.

Start with the [experiment README](LLaDA/elephant-benchmark/README.md) and
[sample moral results](LLaDA/elephant-benchmark/reports/moral_sample.md).
Run `LLaDA/elephant-benchmark/status.py` with Python to inspect live progress.

The benchmark's supplied flipped samples repeat the original prompts. Genuine
flipped samples are recovered from the published paired results; see the
[data validation report](LLaDA/elephant-benchmark/reports/data_validation.json).
Full SS availability is recorded in the [OSF access report](LLaDA/elephant-benchmark/reports/osf_access.json).
