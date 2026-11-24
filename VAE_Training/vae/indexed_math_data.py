"""Read full-corpus token arrays without loading millions of Python rows into RAM."""
import json
import mmap
from pathlib import Path

import numpy as np


class IndexedMathData:
    def __init__(self, directory, source):
        directory = Path(directory)
        self.tokens = np.memmap(directory / 'train.tokens.u32', dtype='<u4', mode='r')
        self.offsets = np.memmap(directory / 'train.offsets.u64', dtype='<u8', mode='r')
        self.source_offsets = np.memmap(directory / 'train.source-offsets.u64', dtype='<u8', mode='r')
        self.stream = Path(source).open('rb')
        self.rows = mmap.mmap(self.stream.fileno(), 0, access=mmap.ACCESS_READ)
        assert len(self.offsets) == len(self.source_offsets) + 1
        assert self.offsets[0] == 0 and self.offsets[-1] == len(self.tokens)

    def __len__(self):
        return len(self.source_offsets)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        first = int(self.source_offsets[index])
        last = self.rows.find(b'\n', first)
        row = json.loads(self.rows[first:last if last >= 0 else len(self.rows)])
        row['input_ids'] = self.tokens[int(self.offsets[index]):int(self.offsets[index + 1])].tolist()
        return row

    def close(self):
        self.rows.close()
        self.stream.close()
