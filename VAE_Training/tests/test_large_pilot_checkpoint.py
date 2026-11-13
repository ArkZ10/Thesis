import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf
import torch
from scripts.run_large_pilot import LargePilot


class LargePilotCheckpointTests(unittest.TestCase):
    def test_epoch_exports_and_final_only_full_state(self):
        with tempfile.TemporaryDirectory() as temp:
            pilot=LargePilot.__new__(LargePilot);pilot.directory=Path(temp)
            pilot.cfg=OmegaConf.create({'train':{'epochs':3},'model':{'decoder_name':'pinned-source'}})
            model=torch.nn.Linear(2,1)
            optimizer=SimpleNamespace(masters=list(model.parameters()),state_dict=lambda:{'steps':15})
            with patch('torch.cuda.get_rng_state_all',return_value=[]):
                first=Path(pilot.save_epoch(model,optimizer,1,5))
                final=Path(pilot.save_epoch(model,optimizer,3,15))
            self.assertTrue((first/'trainable/index.json').exists())
            self.assertFalse((first/'optimizer.pt').exists())
            self.assertFalse(json.loads((first/'metadata.json').read_text())['exact_resume_available'])
            self.assertTrue((final/'optimizer.pt').exists())
            self.assertTrue(json.loads((final/'metadata.json').read_text())['exact_resume_available'])
            self.assertFalse(list((Path(temp)/'checkpoints').glob('*.incomplete')))


if __name__=='__main__':unittest.main()
