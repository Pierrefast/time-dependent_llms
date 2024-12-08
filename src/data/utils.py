import numpy as np
from typing import Dict
import torch

from .shakespeare import get_shakespeare_data
from .wikitext import get_wikitext_data
from .arxiv import get_arxiv_2000, get_arxiv_full
from .openwebtext2 import get_openwebtext2_data
from .slimpajama import get_slimpajama_data
from .fineweb import get_fineweb_data

def get_dataset(args) -> Dict[str, np.ndarray]:
    """ Fetch the right dataset given by the args.dataset parameter. The logic for each dataset is
     contained in its own python file. The expected format at the moment is a dictionary of np.memmap
     containing two keys: 'train' and 'val', corresponding to the tokenized training and validation data. """
    if args.dataset == 'wikitext':
        return get_wikitext_data()
    if args.dataset == "shakespeare-char":
        return get_shakespeare_data()
    if args.dataset == "arxiv2000":
        return get_arxiv_2000()
    if args.dataset == "arxiv":
        return get_arxiv_full()
    if args.dataset == "arxiv+wiki":
        arxiv_data = get_arxiv_full()
        wiki_data = get_wikitext_data()
        train_data = np.concatenate((arxiv_data['train'], wiki_data['train']))
        val_data = np.concatenate((arxiv_data['val'], wiki_data['val']))
        return {'train': train_data, 'val': val_data}
    if args.dataset == 'openwebtext2':
        return get_openwebtext2_data()
    if args.dataset == "slimpajama":
        return get_slimpajama_data()
    if args.dataset == "fineweb":
        return get_fineweb_data()
    else:
        raise NotImplementedError(f"Unknow dataset key '{args.dataset}'")

class TimeDataset(torch.utils.data.Dataset):
    def __init__(self, data,  sequence_length):
        super().__init__()
        self.data = torch.Tensor(np.array(data["tokens"]))
        self.sequence_length = sequence_length
        self.dates = torch.Tensor(np.array(data["dates"]))
        self.data.to(device="cuda")
        self.dates.to(device="cuda")
        self.permutations = np.random.permutation(np.arange(len(self.data)))

    def __len__(self):
        total_length = len(self.data)
        # chunk the data into sequences of length `sequence_length`
        # NOTE: we discard the last remainding sequence if it's not of length `sequence_length`
        return (total_length - 1) // self.sequence_length

    def __getitem__(self, idx):
        seq_length = self.sequence_length
        idx = idx * seq_length
        idx = self.permutations[idx]
        x = torch.from_numpy((self.data[idx : idx + seq_length]).astype(np.int64))
        y = torch.from_numpy(
            (self.data[idx + 1 : idx + 1 + seq_length]).astype(np.int64)
        )
        dates = torch.from_numpy(
            (self.dates[idx: idx + seq_length]).astype(np.int64)
        )
        return (x, y, dates)


class DatesCollater:
    def __call__(self, batch):
        x = [t[0] for t in batch]
        x = torch.stack(x)
        dates = [t[2] for t in batch]
        dates = torch.stack(dates)
        y = [t[1] for t in batch]
        y = torch.stack(y)
        return x,y,dates

    def collate(self, batch):
        return self(batch)


def get_dataloader(data, sequence_length, batch_size, seed=0, distributed_backend=None):
    """Create a DataLoader for the given data. If distributed_backend is provided and is truly
    distributed (world size > 1), the DataLoader will be created with a DistributedSampler that
    splits the data across the processes (in conjunction with DDP).
    Otherwise, use a RandomSampler with the specified seed.

    Returns both the dataloader and the sampler.
    """
    dataset = TimeDataset(data, sequence_length=sequence_length)
    if distributed_backend and distributed_backend.get_world_size() > 1:
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            shuffle=True,
            seed=seed,
        )
    else:
        g = torch.Generator()
        g.manual_seed(seed)
        sampler = torch.utils.data.RandomSampler(
            dataset, replacement=False, generator=g
        )

    loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        collate_fn = DatesCollater(),
        num_workers=4,
    )
    return loader, sampler
