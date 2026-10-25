from types import SimpleNamespace
from dataclasses import dataclass

import accelerate
import torch
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

from dllm.core.samplers.recurrent_mdlm import (
    RecurrentMDLMSampler,
    RecurrentMDLMSamplerConfig,
)
from dllm.utils.utils import get_default_logger

logger = get_default_logger(__name__)

_RECURRENT_LIKELIHOOD_UNSUPPORTED = (
    "recurrent evaluation only supports generate_until in lm-eval. "
    "True recurrent loglikelihood scoring is not implemented yet."
)


@dataclass
class RecurrentLLaDAEvalConfig(RecurrentMDLMSamplerConfig):
    max_new_tokens: int = 1024
    max_length: int = 4096
    steps: int = 1024
    block_size: int = 1024
    pretrained: str = ""
    dtype: str | torch.dtype = "auto"
    batch_size: int = 32
    mc_num: int = 128
    is_check_greedy: bool = False
    device: str = "cuda"


class RuntimeModelProxy:
    def __init__(self, model, device) -> None:
        self._model = model
        self.device = torch.device(device)

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._model, name)


def _resolve_rank_world_size(accelerator_obj):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    if accelerator_obj is not None:
        return accelerator_obj.process_index, accelerator_obj.num_processes
    return 0, 1


def _load_runtime_model(model_args, **kwargs):
    from dllm.utils.models import get_model

    return get_model(model_args, **kwargs)


def _resolve_dtype(dtype):
    from lm_eval.models.utils import get_dtype

    return get_dtype(dtype)


def _prepare_runtime(
    *,
    runtime_model,
    runtime_tokenizer,
    runtime_accelerator,
    model_loader,
    pretrained: str,
    dtype,
    device,
):
    accelerator_obj = runtime_accelerator or accelerate.Accelerator()
    rank, world_size = _resolve_rank_world_size(accelerator_obj)
    runtime_device = getattr(accelerator_obj, "device", device)

    if runtime_model is None:
        from dllm.utils.models import get_tokenizer

        model = model_loader(pretrained, dtype)
        model.eval()
        if accelerator_obj.num_processes > 1:
            model = accelerator_obj.prepare(model)
        else:
            model = model.to(device)
        tokenizer = get_tokenizer(
            SimpleNamespace(model_name_or_path=pretrained, model=model)
        )
    else:
        model = runtime_model
        tokenizer = runtime_tokenizer

    wrapped_model = RuntimeModelProxy(model, runtime_device)
    wrapped_model.eval()
    return (
        wrapped_model,
        tokenizer,
        accelerator_obj,
        torch.device(runtime_device),
        rank,
        world_size,
    )


@register_model("recurrent_llada")
class RecurrentLLaDAHarness(LM):
    @staticmethod
    def _parse_token_list(value):
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                value = value[1:-1]
            if not value:
                return []
            return [int(x.strip()) for x in value.split(";") if x.strip()]
        if isinstance(value, list):
            return value
        if value is None:
            return []
        return []

    def __init__(
        self,
        config: RecurrentLLaDAEvalConfig | None = None,
        *,
        runtime_model=None,
        runtime_tokenizer=None,
        runtime_accelerator=None,
        **kwargs,
    ):
        LM.__init__(self)
        if config is None:
            config = RecurrentLLaDAEvalConfig()

        pretrained = kwargs.get("pretrained", config.pretrained)
        dtype = kwargs.get("dtype", config.dtype)
        batch_size = kwargs.get("batch_size", config.batch_size)
        mc_num = kwargs.get("mc_num", config.mc_num)
        is_check_greedy = kwargs.get("is_check_greedy", config.is_check_greedy)
        device = kwargs.get("device", config.device)
        cfg = kwargs.get("cfg", config.cfg_scale)
        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        block_size = kwargs.get("block_size", config.block_size)
        max_length = kwargs.get("max_length", config.max_length)
        remasking = kwargs.get("remasking", config.remasking)
        suppress_tokens = self._parse_token_list(
            kwargs.get("suppress_tokens", config.suppress_tokens)
        )
        begin_suppress_tokens = self._parse_token_list(
            kwargs.get("begin_suppress_tokens", config.begin_suppress_tokens)
        )
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        debug_stats = kwargs.get("debug_stats", False)
        if isinstance(debug_stats, str):
            debug_stats = debug_stats.lower() == "true"
        use_cache = kwargs.get("use_cache", None)
        if use_cache == "none":
            use_cache = None
        threshold = kwargs.get("threshold", None)
        if isinstance(threshold, str):
            threshold = float(threshold) if threshold.lower() != "none" else None
        factor = kwargs.get("factor", None)
        if isinstance(factor, str):
            factor = float(factor) if factor.lower() != "none" else None

        (
            self.model,
            self.tokenizer,
            self.accelerator,
            self.device,
            self._rank,
            self._world_size,
        ) = _prepare_runtime(
            runtime_model=runtime_model,
            runtime_tokenizer=runtime_tokenizer,
            runtime_accelerator=runtime_accelerator,
            model_loader=lambda model_name, model_dtype: _load_runtime_model(
                SimpleNamespace(
                    model_name_or_path=model_name, dtype=_resolve_dtype(model_dtype)
                ),
                is_recurrent=True,
            ),
            pretrained=pretrained,
            dtype=dtype,
            device=device,
        )

        self.mask_id = self.tokenizer.mask_token_id
        self.batch_size = int(batch_size)
        self.max_length = max_length
        self.max_new_tokens = int(max_new_tokens)
        self.block_size = int(block_size)
        self.steps = int(steps)
        self.cfg = float(cfg)
        self.remasking = remasking
        self.is_check_greedy = is_check_greedy
        self.suppress_tokens = suppress_tokens
        self.begin_suppress_tokens = begin_suppress_tokens
        self.right_shift_logits = right_shift_logits
        self.mc_num = int(mc_num)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.0
        self.debug_stats = debug_stats
        self.use_cache = use_cache
        self.threshold = threshold
        self.factor = factor

        warmup = kwargs.get("warmup", False)
        if isinstance(warmup, str):
            warmup = warmup.lower() == "true"
        self.warmup = warmup

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_decode(
        self, tokens: torch.Tensor | list[int], skip_special_tokens: bool = True
    ) -> str:
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def tok_encode(self, text: str, add_special_tokens: bool = True) -> torch.Tensor:
        return self.tokenizer(
            text, return_tensors="pt", add_special_tokens=add_special_tokens
        ).input_ids

    def apply_chat_template(
        self, chat_history: list[dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def generate_until(self, requests):
        out = []
        sampler_config = RecurrentMDLMSamplerConfig(
            steps=self.steps,
            max_new_tokens=self.max_new_tokens,
            block_size=self.block_size,
            temperature=0.0,
            cfg_scale=self.cfg,
            remasking=self.remasking,
            suppress_tokens=self.suppress_tokens,
            begin_suppress_tokens=self.begin_suppress_tokens,
            right_shift_logits=self.right_shift_logits,
            debug_stats=self.debug_stats,
            use_cache=self.use_cache,
            threshold=self.threshold,
            factor=self.factor,
            warmup=self.warmup,
        )
        sampler = RecurrentMDLMSampler(model=self.model, tokenizer=self.tokenizer)

        for instance in tqdm(requests, desc="Generating..."):
            context, gen_kwargs = instance.args
            prompt_ids = self.tokenizer(context)["input_ids"]
            if len(prompt_ids) > self.max_length - self.max_new_tokens:
                cutoff_len = self.max_length - self.max_new_tokens
                logger.warning(
                    f"Prompt length {len(prompt_ids)} exceeds {cutoff_len}, cutoff on the left side"
                )
                prompt_ids = prompt_ids[-cutoff_len:]
            prompt = [torch.tensor(prompt_ids, device=self.device, dtype=torch.long)]
            stop_tokens = gen_kwargs["until"]

            output = sampler.sample(
                inputs=prompt,
                config=sampler_config,
                return_dict=True,
            )

            generated_answer = self.tokenizer.decode(
                output.sequences[0][prompt[0].shape[0] :],
                skip_special_tokens=False,
            )
            for stop_seq in stop_tokens:
                if stop_seq in generated_answer:
                    generated_answer = generated_answer.split(stop_seq)[0]

            generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
            generated_answer = self.tokenizer.decode(
                generated_answer_ids, skip_special_tokens=True
            )
            out.append(generated_answer)

            if self.accelerator is not None and self._world_size > 1:
                self.accelerator.wait_for_everyone()

        return out

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError(_RECURRENT_LIKELIHOOD_UNSUPPORTED)

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        raise NotImplementedError(_RECURRENT_LIKELIHOOD_UNSUPPORTED)
