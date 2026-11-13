import unittest
import io
import torch

from vae.split_training import StreamedCPUAdamW


class StreamedAdamTests(unittest.TestCase):
    def test_matches_reference_adamw_with_global_clipping(self):
        p=[torch.nn.Parameter(torch.tensor([0.3,-0.8])),torch.nn.Parameter(torch.tensor([1.2]))]
        ref=[torch.nn.Parameter(x.detach().clone()) for x in p]
        actual=StreamedCPUAdamW([(str(i),x) for i,x in enumerate(p)],lr=2e-5,betas=(0.9,0.95),weight_decay=0.01)
        reference=torch.optim.AdamW(ref,lr=2e-5,betas=(0.9,0.95),weight_decay=0.01)
        self.assertNotEqual(actual.masters[0].data_ptr(),p[0].data_ptr())
        for gradients in [[torch.tensor([0.7,-0.1]),torch.tensor([0.3])],
                          [torch.tensor([-0.2,0.6]),torch.tensor([-0.8])]]:
            for a,b,g in zip(p,ref,gradients):a.grad=g.clone();b.grad=g.clone()
            torch.nn.utils.clip_grad_norm_(ref,0.5)
            reference.step();reference.zero_grad(set_to_none=True)
            actual.step(max_grad_norm=0.5)
            for a,b in zip(p,ref):torch.testing.assert_close(a,b,rtol=1e-6,atol=1e-7)
        self.assertEqual(actual.steps,2)
        self.assertTrue(all(x.grad is None for x in p))
        self.assertEqual(actual.state_dict()['parameter_names'],['0','1'])

    def test_resume_preserves_unrounded_fp32_master_and_next_update(self):
        p=torch.nn.Parameter(torch.tensor([0.3,-0.8],dtype=torch.bfloat16))
        opt=StreamedCPUAdamW([('weight',p)],lr=2e-5,betas=(0.9,0.95),weight_decay=0.01)
        p.grad=torch.tensor([0.7,-0.1],dtype=p.dtype);opt.step()
        buffer=io.BytesIO();torch.save(opt.state_dict(),buffer);buffer.seek(0)
        checkpoint=torch.load(buffer,weights_only=True)
        q=torch.nn.Parameter(p.detach().clone())
        resumed=StreamedCPUAdamW([('weight',q)],lr=2e-5,betas=(0.9,0.95),weight_decay=0.01)
        self.assertFalse(torch.equal(resumed.masters[0],opt.masters[0]))
        resumed.load_state_dict(checkpoint)
        torch.testing.assert_close(resumed.masters[0],opt.masters[0],rtol=0,atol=0)
        for param,optimizer in [(p,opt),(q,resumed)]:
            param.grad=torch.tensor([-0.2,0.6],dtype=param.dtype);optimizer.step()
        torch.testing.assert_close(resumed.masters[0],opt.masters[0],rtol=0,atol=0)
        torch.testing.assert_close(q,p,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
