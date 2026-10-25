"""Linear flow-matching Euler scheduler matching LaDiR paper Eq. (1).

z(t)=(1-t)*clean+t*noise; velocity=noise-clean, t decreases 1 -> 0.
Public model timesteps are scaled by num_train_timesteps (1000 by default).
FP32 integration avoids accumulating BF16 rounding at every Euler step.
"""
from types import SimpleNamespace
import torch


class FlowMatchEulerDiscreteScheduler:
    def __init__(self,num_train_timesteps=1000):
        if type(num_train_timesteps) is not int or num_train_timesteps<1:
            raise ValueError('num_train_timesteps must be a positive integer')
        self.num_train_timesteps=num_train_timesteps
        self.config=SimpleNamespace(prediction_type='flow',num_train_timesteps=num_train_timesteps)
        self.timesteps=None;self.sigmas=None;self._step_index=None

    def set_timesteps(self,num_inference_steps,device=None):
        if type(num_inference_steps) is not int or num_inference_steps<1:
            raise ValueError('num_inference_steps must be a positive integer')
        self.num_inference_steps=num_inference_steps
        self.sigmas=torch.linspace(1.,0.,num_inference_steps+1,device=device,dtype=torch.float32)
        self.timesteps=self.sigmas[:-1]*self.num_train_timesteps
        self._step_index=0

    def add_noise(self,original_samples,noise,timesteps):
        if original_samples.shape!=noise.shape or original_samples.device!=noise.device:
            raise ValueError('Clean and noise tensors must share shape and device')
        t=torch.as_tensor(timesteps,device=original_samples.device,dtype=torch.float32)/self.num_train_timesteps
        if t.ndim>1 or (t.ndim==1 and t.numel() not in (1,original_samples.shape[0])):
            raise ValueError('Timesteps must be scalar or one per batch element')
        if not torch.isfinite(t).all() or ((t<0)|(t>1)).any():
            raise ValueError('Timesteps must be finite and within [0,num_train_timesteps]')
        while t.ndim<original_samples.ndim:t=t.unsqueeze(-1)
        return (1-t)*original_samples.float()+t*noise.float()

    def get_velocity(self,original_samples,noise,timesteps=None):
        if original_samples.shape!=noise.shape:raise ValueError('Velocity shape mismatch')
        return noise.float()-original_samples.float()

    def step(self,model_output,timestep,sample,generator=None):
        # Deterministic ODE integration; generator is intentionally unused.
        if self.timesteps is None or self._step_index is None:
            raise ValueError('Call set_timesteps before step')
        i=self._step_index
        if i>=self.num_inference_steps:raise ValueError('All scheduled steps already consumed')
        requested=torch.as_tensor(timestep).detach().float().cpu()
        if requested.numel()!=1 or not torch.isclose(requested,self.timesteps[i].detach().cpu(),rtol=0,atol=1e-3).all():
            raise ValueError('Out-of-order or mismatched timestep')
        if sample.shape!=model_output.shape or sample.device!=model_output.device:
            raise ValueError('Sample and velocity must share shape and device')
        if not torch.isfinite(sample).all() or not torch.isfinite(model_output).all():
            raise ValueError('Nonfinite sample or predicted velocity')
        dt=(self.sigmas[i+1]-self.sigmas[i]).to(sample.device)
        previous=sample.float()+dt*model_output.float()
        if not torch.isfinite(previous).all():raise ValueError('Euler step produced nonfinite values')
        self._step_index+=1
        return SimpleNamespace(prev_sample=previous)


def check():
    gen=torch.Generator().manual_seed(42)
    clean=torch.randn(2,3,128,generator=gen);noise=torch.randn(2,3,128,generator=gen)
    scheduler=FlowMatchEulerDiscreteScheduler()
    assert torch.equal(scheduler.add_noise(clean,noise,0),clean)
    assert torch.equal(scheduler.add_noise(clean,noise,1000),noise)
    halfway=scheduler.add_noise(clean,noise,torch.tensor([500,500]))
    assert torch.equal(halfway,.5*clean+.5*noise)
    errors={}
    for steps in (1,2,10,50):
        scheduler.set_timesteps(steps)
        x=noise.clone();velocity=scheduler.get_velocity(clean,noise)
        for t in scheduler.timesteps:x=scheduler.step(velocity,t,x).prev_sample
        error=float((x-clean).abs().max());errors[str(steps)]=error
        assert error<1e-5,(steps,error)
    scheduler.set_timesteps(4)
    try:scheduler.step(noise,scheduler.timesteps[1],noise)
    except ValueError:pass
    else:raise AssertionError('Out-of-order step accepted')
    try:scheduler.add_noise(clean,noise,1001)
    except ValueError:pass
    else:raise AssertionError('Invalid noise timestep accepted')
    assert scheduler._step_index==0
    x=scheduler.step(torch.zeros_like(noise),scheduler.timesteps[0],noise.bfloat16()).prev_sample
    assert x.dtype==torch.float32
    return {'status':'passed','interpolation_endpoints_exact':True,'midpoint_exact':True,
        'oracle_velocity_max_errors':errors,'out_of_order_rejected':True,
        'invalid_timesteps_rejected':True,'integration_precision':'FP32',
        'equation_source':'https://arxiv.org/html/2510.04573v6#S2.SS2'}


if __name__=='__main__':
    import json
    print(json.dumps(check(),indent=2))
