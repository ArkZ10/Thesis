"""Split each fixed 128-example update into length-aware physical batches."""
import torch


def physical_batches(rows, max_examples=32, max_tokens=4096):
    if max_examples < 1 or max_tokens < 1:
        raise ValueError('Positive physical-batch limits required')
    ordered = sorted(rows, key=lambda row: len(row['input_ids']), reverse=True)
    cursor = 0
    while cursor < len(ordered):
        length = len(ordered[cursor]['input_ids'])
        if length > max_tokens:
            raise ValueError('A source target exceeds the token budget; never truncate it')
        count = min(max_examples, max_tokens // length, len(ordered) - cursor)
        yield ordered[cursor:cursor + count]
        cursor += count


def padded_batch(rows, eos, device, pad_to=None):
    length = pad_to or max(len(row['input_ids']) for row in rows)
    if any(len(row['input_ids']) > length for row in rows):
        raise ValueError('Source target exceeds configured length')
    ids = torch.full((len(rows), length), eos, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for index, row in enumerate(rows):
        n = len(row['input_ids'])
        ids[index, :n] = torch.tensor(row['input_ids'], device=device)
        mask[index, :n] = 1
    return ids, mask
