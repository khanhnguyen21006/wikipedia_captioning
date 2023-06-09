from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from modules.data_pool import *

from functools import partial


class DataModule(LightningDataModule):
    def __init__(self, _config, dist=False):
        super(DataModule, self).__init__()
        self.data_folder = _config["data_folder"]
        self.dataset = _config["dataset"]

        self.num_workers = _config["num_workers"]
        self.batch_size = _config["per_gpu_batchsize"]

        self.transforms = get_transform(_config["transform"])
        self.collate_hparams = get_dataset_hparams(_config)

        self.collate_fn = None
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_sampler = None
        self.val_sampler = None
        self.test_sampler = None

        self.dist = dist
        self.setup_flag = False

    @property
    def dataset_cls(self):
        return get_dataset(self.dataset)

    def setup(self, stage):
        if not self.setup_flag:
            self.set_train_dataset()
            self.set_val_dataset()
            self.set_test_dataset()

            self.collate_fn = partial(self.train_dataset.collate)
            self.setup_flag = True

            if self.dist:
                self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
                self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False)
                self.test_sampler = DistributedSampler(self.test_dataset, shuffle=False)
            else:
                self.train_sampler = SequentialSampler(self.train_dataset)
                self.val_sampler = None
                self.test_sampler = None

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,  # consider switching this off as images are already processed and only need to call cuda()
            collate_fn=self.collate_fn,
        )
        return loader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=self.test_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )
        return loader

    def val_dataloader(self) -> EVAL_DATALOADERS:
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )
        return loader

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def get_ret_dset(self, split):
        return self.dataset_cls(
            split=split,
            d_folder=os.path.join(self.data_folder, 'wiki_retrieval'),
            d_name=self.dataset,
            transform=self.transforms['val']
        )

    def set_train_dataset(self):
        self.train_dataset = self.dataset_cls(
            split="train",
            d_folder=self.data_folder,
            d_name=self.dataset,
            transform=self.transforms['train']
        )
        self.train_dataset.hparams = self.collate_hparams

    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(
            split="val",
            d_folder=self.data_folder,
            d_name=self.dataset,
            transform=self.transforms['val']
        )
        self.val_dataset.hparams = self.collate_hparams

    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(
            split="test",
            d_folder=self.data_folder,
            d_name=self.dataset,
            transform=self.transforms['val']
        )
        self.test_dataset.hparams = self.collate_hparams
