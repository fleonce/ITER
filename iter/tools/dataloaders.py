import sys
from typing import Sequence, Protocol, TypeVar, Optional

from torch.utils.data import DataLoader, DistributedSampler

from iter.data.data_collator import Batch, DataCollatorForITER
from iter.datasets import ITERDataset
from iter.datasets.training import Hparams


class DataLoaderProtocol(Protocol):
    def __call__(
        self,
        datasets: Sequence[ITERDataset],
        split: str,
        train: bool,
        hparams: Hparams,
        world_size: int,
        rank: int,
        seed: int,
        use_fsdp: bool
    ) -> DataLoader: ...


def create_primary_dataloader(
    datasets: Sequence[ITERDataset],
    split: str,
    train: bool,
    hparams: Hparams,
    world_size: int,
    rank: int,
    seed: int,
    use_fsdp: bool,
):
    dataloader_kwargs = {
        "batch_size": hparams.get_batch_size(train),
        "collate_fn": DataCollatorForITER(datasets[0]),
        "dataset": datasets[0][split],
        "shuffle": train,
    }
    if use_fsdp:
        dataloader_kwargs.pop("shuffle")
        sampler: DistributedSampler
        sampler = DistributedSampler(datasets[0][split], world_size, rank, train, seed)
        dataloader_kwargs.update({"sampler": sampler})
    dataloader = DataLoader(**dataloader_kwargs)
    return dataloader


_T_co = TypeVar("_T_co", covariant=True)
class SupportsNext(Protocol[_T_co]):
    def __next__(self) -> _T_co:
        ...


class MergingDataloader:
    dataloaders: Sequence[DataLoader]
    iters: tuple[SupportsNext[Batch], ...]

    def __init__(self, dataloaders: Sequence[DataLoader]):
        self.dataloaders = dataloaders
        self.iters = ()

    def __iter__(self):
        self.iters = tuple(iter(loader) for loader in self.dataloaders)
        return self

    def __len__(self):
        def try_len(dataloader: DataLoader):
            try:
                return len(dataloader)
            except:
                return sys.maxsize
        return min(try_len(loader) for loader in self.dataloaders)

    def __next__(self) -> Batch:
        try:
            batches = [next(it) for it in self.iters]
        except StopIteration:
            raise StopIteration
        batch = Batch()
        for elem in batches:
            batch.update(elem)
        return batch


class InfiniteDataLoader:
    iter: Optional[SupportsNext[Batch]]

    def __init__(self, inner_dataloader: DataLoader):
        self.inner = inner_dataloader
        self.iter = None

    def __iter__(self):
        self.iter = iter(self.inner)
        return self

    def __next__(self):
        try:
            if self.iter is None:
                raise ValueError("iter is None")
            return next(self.iter)
        except StopIteration:
            self.iter = iter(self.inner)
            return next(self)

    def __len__(self):
        return sys.maxsize
