from .configuration_llada import LLaDAConfig
from .modeling_llada import LLaDAModelLM

try:
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

    AutoConfig.register("llada", LLaDAConfig)
    AutoModel.register(LLaDAConfig, LLaDAModelLM)
    AutoModelForMaskedLM.register(LLaDAConfig, LLaDAModelLM)
except ImportError:
    pass
