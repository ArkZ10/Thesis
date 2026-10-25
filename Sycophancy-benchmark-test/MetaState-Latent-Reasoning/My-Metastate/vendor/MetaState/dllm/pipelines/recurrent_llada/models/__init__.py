from .modeling_recurrent_llada import RecurrentLLaDAModel
from ..configs.recurrent_llada_config import RecurrentLLaDAConfig

# Register with HuggingFace Auto classes for local usage
try:
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

    AutoConfig.register("recurrent_llada", RecurrentLLaDAConfig)
    AutoModel.register(RecurrentLLaDAConfig, RecurrentLLaDAModel)
    AutoModelForMaskedLM.register(RecurrentLLaDAConfig, RecurrentLLaDAModel)

except ImportError:
    pass
