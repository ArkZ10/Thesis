"""
Recurrent LLaDA evaluation entrypoint for lm-eval-harness.
"""

from lm_eval.__main__ import cli_evaluate

from dllm.pipelines.deterministic_fewshot import install_deterministic_fewshot_patch
from dllm.pipelines.recurrent_eval import RecurrentLLaDAHarness

install_deterministic_fewshot_patch()


if __name__ == "__main__":
    cli_evaluate()
