import math
import random
from typing import Iterator, List, Optional, TypeVar

import torch
from datasets.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler

T_co = TypeVar("T_co", covariant=True)


class DistributedMixedBatchSampler(DistributedSampler):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True, seed: int = 0,
                 drop_last: bool = False, indices_gt: Optional[List[int]] = None,
                 batch_size: Optional[int] = None) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

        self.replace_idcs, self.replace_vals = [], []
        if indices_gt is not None:
            if batch_size is None:
                raise ValueError("If indices_gt is set, batch_size must also be set for the "
                                 "Sampler")

            # We adapt the total_size as the number of samples per epoch gets increased by one
            # sample per batch that is sampled
            num_samples_mix = math.ceil(
                (len(self.dataset) // batch_size) / self.num_replicas
            )

            num_gt_reps = num_samples_mix * self.num_replicas

            # Adapt the number of samples per GPU and the total_size in the base class
            self.num_samples += num_samples_mix
            self.total_size = self.num_samples * self.num_replicas

            print("num_samples_mix", num_samples_mix)
            for i in range(num_gt_reps):
                self.replace_idcs.append(
                    i * batch_size + random.sample(range(0, batch_size - 1), 1)[0])
                self.replace_vals.append(random.sample(indices_gt, 1)[0])

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if self.replace_idcs:
            for idx_gt, replace_idx_val in enumerate(zip(self.replace_idcs, self.replace_vals)):
                indices.insert(replace_idx_val[0], replace_idx_val[1])

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples
