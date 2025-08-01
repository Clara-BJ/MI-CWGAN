from collections.abc import Mapping
from typing import Any, List, Optional, Sequence, Union

import torch.utils.data
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Batch, Dataset
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from torch_geometric.typing import TensorFrame, torch_frame
from torch.utils.data import Dataset as utilsdataset

class Collater:
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
    ):
        self.dataset = dataset
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch: List[Any]) -> Any:
        list_ = batch[0]
        result_list = []
        for i in range(len(list_)):
            elem_list = [batch[j][i][0] for j in range(len(batch))]
            out_list = self.call(elem_list)
            result_list.append(out_list)
        y_list = [batch[j][0][1] for j in range(len(batch))]
        y_list = self.call(y_list)
        result_list.append(y_list)
        return result_list

    def call(self, batch: List[Any]) -> Any:
        return default_collate(batch)

        raise TypeError(f"DataLoader found invalid type: '{type(elem)}'")

class MultiDataLoader(torch.utils.data.DataLoader):

    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter, utilsdataset],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop('collate_fn', None)

        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=Collater(dataset, follow_batch, exclude_keys),
            **kwargs,
        )

class MultiModalDataset(utilsdataset):
    def __init__(self, datasets):
        self.datasets = datasets
        assert all(len(dataset) == len(datasets[0]) for dataset in datasets), "Datasets must have the same length"

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        sample = [dataset[idx] for dataset in self.datasets]
        return sample
