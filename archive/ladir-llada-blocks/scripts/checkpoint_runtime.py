"""Strict checkpoint restoration without pretrained downloads/reinitialization.

Supports the released CD4 tensor layout, one-block inspection only. This does not
implement paper-style prefix continuation, a denoising scheduler, or generation.
"""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
VENDOR=ROOT/'vendor/LaDiR'


def import_source(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def skeleton():
    from accelerate import init_empty_weights
    from peft import LoraConfig,get_peft_model
    from transformers import AutoTokenizer,LlamaConfig,LlamaForCausalLM
    # Explicitly load the recorded vendor files; don't import another model.py.
    # Keep vendor files untouched; supply the validated flow scheduler at import.
    import_source('fm_noise_scheduler',ROOT/'scripts/flow_scheduler.py')
    source=import_source('ladir_vendor_model',VENDOR/'model.py')
    vae_source=import_source('ladir_vendor_vae',VENDOR/'vae/model_vae.py')
    base=ROOT/'assets/llama-3.1-8b'
    tokenizer=AutoTokenizer.from_pretrained(str(base),local_files_only=True)
    tokenizer.pad_token_id=tokenizer.eos_token_id
    for spelling,attribute in [('<tht_s>','bot_token_id'),('<tht>','tht_token_id'),
                              ('</tht_s>','eot_token_id'),('<timestep>','time_token_id')]:
        tokenizer.add_special_tokens({'additional_special_tokens':[spelling]})
        setattr(tokenizer,attribute,tokenizer.convert_tokens_to_ids(spelling))
    config=LlamaConfig.from_pretrained(str(base),local_files_only=True)
    config._attn_implementation='eager'
    assert config.vocab_size==128256 and config.hidden_size==4096
    class RestoredVAE(vae_source.VAE):
        def __init__(self):
            # Skip vendor constructor downloads; retain the actual VAE methods.
            nn.Module.__init__(self)
            c=copy.deepcopy(config);c.vocab_size=128261
            self.icae=get_peft_model(LlamaForCausalLM(c),LoraConfig(r=512,lora_alpha=256,
                lora_dropout=0.05,bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','v_proj']))
            self.dim=128;self.mem_size=3
            self.mean=nn.Linear(4096,128);self.log_var=nn.Linear(4096,128)
            self.decompress_layer=nn.Linear(128,4096)
            self.memory_token_embed=nn.Embedding(4,128)
            self.vocab_size=128257;self.pad_token_id=128256
            self.vocab_size_with_mem=128260;self.ae_token_id=128260
            # Compression length is not recoverable from weights. Never silently
            # choose one for generation; this stage only tests explicit token spans.
            self.mean_compression_rate=None
            self.beta=0.;self.tokenizer=tokenizer
            self.bos_id=tokenizer.bos_token_id;self.eos_id=tokenizer.eos_token_id
            self.loss_fct=nn.CrossEntropyLoss(ignore_index=-100)
        def compute_num_segments(self,total_length):
            if self.mean_compression_rate is None:
                raise ValueError('Checkpoint compression length is unverified; specify a validated policy before compression')
            return super().compute_num_segments(total_length)
    class RestoredFusion(source.LMFusionModel):
        def initialize_weights(self):
            # Every persistent tensor must come from the checkpoint, checked below.
            pass
        def build_bidirectional_tht_mask(self,input_ids):
            allowed=super().build_bidirectional_tht_mask(input_ids).bool()
            # HF eager attention consumes additive 0/-inf masks, not float 0/1.
            return torch.zeros(allowed.shape,device=allowed.device,dtype=torch.bfloat16).masked_fill(
                ~allowed,torch.finfo(torch.bfloat16).min)
    with init_empty_weights():
        model=RestoredFusion(text_llama=LlamaForCausalLM(config),thought_llama=None,
            autoencoder=RestoredVAE(),tokenizer=tokenizer,model_config=SimpleNamespace(),
            hidden_dim=4096,freeze_text=True)
    model.eval()
    for p in model.parameters():p.requires_grad=False
    return model,tokenizer


def compare_layout(model,header):
    actual=model.state_dict()
    expected={k:v for k,v in header.items() if k!='__metadata__'}
    missing=sorted(set(actual)-set(expected));unexpected=sorted(set(expected)-set(actual))
    mismatches={k:{'model':list(actual[k].shape),'checkpoint':expected[k]['shape']}
        for k in set(actual)&set(expected) if list(actual[k].shape)!=expected[k]['shape']}
    if missing or unexpected or mismatches:
        raise ValueError(json.dumps({'missing':missing,'unexpected':unexpected,'shape_mismatches':mismatches}))
    return {'tensor_count':len(expected),'missing_keys':missing,'unexpected_keys':unexpected,
        'shape_mismatches':mismatches,'initial_parameters_all_meta':all(p.is_meta for p in model.parameters())}


def load_checkpoint(checkpoint,progress=None):
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    model,tokenizer=skeleton()
    header=json.loads((ROOT/'result/audit/checkpoint_header.json').read_text())
    layout=compare_layout(model,header)
    assert torch.cuda.device_count()>=2,'This loader explicitly places reasoner on GPU 0, VAE on GPU 1'
    expected=set(model.state_dict());loaded=[];conversion_max=0.
    with safe_open(str(checkpoint),framework='pt',device='cpu') as f:
        assert set(f.keys())==expected
        for index,key in enumerate(sorted(expected)):
            tensor=f.get_tensor(key)
            assert torch.isfinite(tensor).all(),f'Nonfinite checkpoint tensor: {key}'
            dtype=torch.bfloat16 if tensor.is_floating_point() else tensor.dtype
            device='cuda:1' if key.startswith('autoencoder.') else 'cuda:0'
            set_module_tensor_to_device(model,key,device,value=tensor,dtype=dtype)
            restored=model.get_parameter(key)
            # Compare every restored tensor against the source after the explicitly
            # requested BF16 conversion. No missing/random parameters are accepted.
            reference=tensor.to(dtype=dtype)
            assert torch.equal(restored.detach().cpu(),reference),key
            if tensor.dtype==torch.float32:
                # Only record small reductions, never retain another full copy.
                conversion_max=max(conversion_max,float((tensor-reference.float()).abs().max()))
            loaded.append(key)
            del tensor,reference
            if progress and (index%32==0 or index+1==len(expected)):progress(index+1,len(expected))
    assert set(loaded)==expected and not any(p.is_meta for p in model.parameters())
    # Nonpersistent rotary buffers are initialized from the byte-matched config.
    for name,module in model.named_modules():
        device='cuda:1' if name.startswith('autoencoder') else 'cuda:0'
        for key,buffer in list(module._buffers.items()):
            if buffer is not None:
                assert not buffer.is_meta
                module._buffers[key]=buffer.to(device)
        if hasattr(module,'original_inv_freq'):
            module.original_inv_freq=module.inv_freq
    model.autoencoder.device=torch.device('cuda:1')
    model.autoencoder.append_sequence=torch.arange(128257,128260,device='cuda:1').unsqueeze(0)
    model.eval()
    return model,tokenizer,{'layout':layout,'loaded_tensors':len(loaded),
        'all_loaded_tensors_exact_after_bf16_conversion':True,
        'max_fp32_to_bf16_absolute_roundoff':conversion_max,
        'device_placement':{'reasoner':'cuda:0','vae':'cuda:1'},
        'backbone_reinitialization_skipped':True,'base_weights_downloaded':False}
