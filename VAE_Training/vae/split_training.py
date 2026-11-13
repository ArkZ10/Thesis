"""Explicit two-device VAE and streamed FP32 CPU AdamW master parameters."""
import torch

from .model import ReasoningVAE


class SplitReasoningVAE(ReasoningVAE):
    def decoder_prefix(self,z):
        return super().decoder_prefix(z.to(self.latent_to_decoder.weight.device))

    def decoder_hidden(self,z,input_ids,attention_mask):
        device=self.latent_to_decoder.weight.device
        return super().decoder_hidden(z,input_ids.to(device),attention_mask.to(device))

    def forward(self,input_ids,attention_mask,labels=None,kl_weight=0.0):
        if labels is None:labels=input_ids.masked_fill(attention_mask==0,-100)
        mu,log_var=self.encode(input_ids,attention_mask)
        hidden=self.decoder_hidden(self.reparameterize(mu,log_var),input_ids,attention_mask)
        ce,n,correct=self.ce_loss(hidden,labels.to(hidden.device))
        kl=-0.5*torch.mean(1+log_var.float()-mu.float().pow(2)-log_var.float().exp())
        return dict(loss=ce+kl_weight*kl.to(ce.device),ce_loss=ce,kl_loss=kl,
                    n_tokens=n,n_correct=correct,mu=mu,log_var=log_var)


class StreamedCPUAdamW:
    """Full AdamW, without retaining a second complete CPU gradient copy.

    Live model parameters/gradients stay on their assigned GPUs. CPU master
    weights and moments remain FP32. Only one parameter's gradient and updated
    weights are transferred at a time; state_dict is ordinary optimizer state.
    """
    def __init__(self,named_parameters,lr,betas,weight_decay):
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        self.named=[(n,p) for n,p in named_parameters if p.requires_grad]
        self.masters=[torch.nn.Parameter(p.detach().to('cpu',dtype=torch.float32,copy=True)) for _,p in self.named]
        self.optimizer=DeepSpeedCPUAdam(self.masters,lr=lr,betas=betas,weight_decay=weight_decay,adamw_mode=True)
        self.steps=0

    @torch.no_grad()
    def step(self,max_grad_norm=1.0):
        norm_squared=0.0
        for _,p in self.named:
            if p.grad is None:continue
            if not torch.isfinite(p.grad).all():raise RuntimeError('Nonfinite model gradient')
            norm_squared+=float(p.grad.float().norm(2))**2
        norm=norm_squared**0.5
        scale=min(1.0,max_grad_norm/(norm+1e-6))
        group=self.optimizer.param_groups[0]
        original=group['params']
        try:
            for (_,p),master in zip(self.named,self.masters):
                if p.grad is None:continue
                master.grad=p.grad.detach().to('cpu',dtype=torch.float32)
                master.grad.mul_(scale)
                group['params']=[master]
                self.optimizer.step()
                p.copy_(master.to(p.device,dtype=p.dtype))
                master.grad=None
                p.grad=None
        finally:
            group['params']=original
        self.steps+=1
        return norm

    def state_dict(self):
        return dict(optimizer=self.optimizer.state_dict(),steps=self.steps,
                    parameter_names=[n for n,_ in self.named],
                    master_weights=[p.detach() for p in self.masters])

    @torch.no_grad()
    def load_state_dict(self,state):
        if state['parameter_names']!=[n for n,_ in self.named]:
            raise ValueError('Checkpoint trainable parameter names differ')
        if len(state['master_weights'])!=len(self.masters):
            raise ValueError('Checkpoint master parameter count differs')
        for master,saved,(_,live) in zip(self.masters,state['master_weights'],self.named):
            if master.shape!=saved.shape:raise ValueError('Checkpoint master shape differs')
            master.copy_(saved)
            live.copy_(master.to(live.device,dtype=live.dtype))
        self.optimizer.load_state_dict(state['optimizer'])
        self.steps=state['steps']
