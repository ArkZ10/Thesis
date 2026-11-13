import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from vae.hybrid_checkpoint import save_hybrid_state,restore_hybrid_state


class HybridCheckpointTest(unittest.TestCase):
    def test_restored_gpu_branch_moments_preserve_next_adam_update(self):
        p=torch.nn.Parameter(torch.tensor([0.3,-0.8]))
        opt=torch.optim.AdamW([p],lr=2e-5)
        p.grad=torch.tensor([0.7,-0.1]);opt.step()
        model_at_checkpoint=p.detach().clone()
        fake=SimpleNamespace(backup_optimizer=opt,fp32_partitioned_groups_flat=[p])
        with tempfile.TemporaryDirectory() as temp:
            report=save_hybrid_state(fake,Path(temp)/'hybrid')
            self.assertEqual(len(report['subgroups']),1)
            p.grad=torch.tensor([-0.2,0.6]);opt.step()
            expected=p.detach().clone()
            with torch.no_grad():p.copy_(model_at_checkpoint)
            opt.state.clear()
            restore_hybrid_state(fake,Path(temp)/'hybrid')
            p.grad=torch.tensor([-0.2,0.6]);opt.step()
            torch.testing.assert_close(p,expected,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
