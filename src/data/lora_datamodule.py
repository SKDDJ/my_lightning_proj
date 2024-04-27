import os
from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from safetensors import safe_open
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

# from torchvision.datasets import MNIST
from torchvision.transforms import transforms


# here we set a LoRADataset class
class LoRADataset(Dataset):
    """A PyTorch dataset for the LoRA dataset. Get a Tensor List."""
    def __init__(self, tensors):
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        tensor = self.tensors[idx]
        return tensor, tensor


class LoRADataModule(LightningDataModule):
    """`LightningDataModule` for the MNIST dataset. A `LightningDataModule` implements 7 key
    methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        predict_dir: str = "predict/",
        train_factor=0.7,
        val_factor=0.2,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `LoRADataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.data_predict: Optional[Dataset] = None

        self.batch_size_per_device = batch_size
        self.train_factor = train_factor
        self.val_factor = val_factor

    def load_and_flatten(self, file_path):
        # 加载.pt文件 若加载的数据是单个张量  TODO add times 10
        loaded_data = torch.load(file_path)
        if torch.is_tensor(loaded_data):
            return loaded_data.flatten() * 10
        # 若加载的数据是包含多个张量的字典
        tensors = []
        for k, tensor in loaded_data.items():
            flattened_tensor = tensor.flatten()
            tensors.append(flattened_tensor)
        # 将处理后的多个张量串联为一个张量
        return torch.cat(tensors) * 10

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size
        if stage == "predict" and self.hparams.predict_dir:
            file_paths = []
            for root, dirs, files in os.walk(self.hparams.predict_dir):
                for file in files:
                    if file.endswith(".pt"):
                        file_path = os.path.join(root, file)
                        file_paths.append(file_path)

            self.flattened_tensors = [self.load_and_flatten(path) for path in file_paths]
            self.data_predict = LoRADataset(self.flattened_tensors)
        elif stage in ['fit', 'validate', 'test'] and self.hparams.data_dir:
            if not self.data_train and not self.data_val and not self.data_test:
                file_paths = []
                for root, dirs, files in os.walk(self.hparams.data_dir):
                    for file in files:
                        if file.endswith(".pt"):
                            file_path = os.path.join(root, file)
                            file_paths.append(file_path)

                self.flattened_tensors = [self.load_and_flatten(path) for path in file_paths]
                dataset = LoRADataset(self.flattened_tensors)

                total_len = len(dataset)
                train_len = int(total_len * self.hparams.train_factor)
                val_len = int(total_len * self.hparams.val_factor)
                test_len = total_len - train_len - val_len
                self.data_train, self.data_val, self.data_test = random_split(
                    dataset,
                    [train_len, val_len, test_len],
                    generator=torch.Generator().manual_seed(42),
                )
        # # load and split datasets only if not loaded already
        # if not self.data_train and not self.data_val and not self.data_test:
        #     # 初始化一个空列表来收集所有.pt文件的路径
        #     file_paths = []
        #     # 使用os.walk遍历data_dir下的所有子目录
        #     for root, dirs, files in os.walk(self.hparams.data_dir):
        #         for file in files:
        #             # 检查文件是否以.pt结尾
        #             if file.endswith(".pt"):
        #                 # 构造完整的文件路径并添加到列表中
        #                 file_path = os.path.join(root, file)
        #                 file_paths.append(file_path)

        #     # 现在file_paths列表包含了所有子文件夹下的.pt文件路径
        #     self.flattened_tensors = [self.load_and_flatten(path) for path in file_paths]
        #     dataset = LoRADataset(self.flattened_tensors)

        #     total_len = len(dataset)
        #     train_len = int(total_len * self.hparams.train_factor)
        #     val_len = int(total_len * self.hparams.val_factor)
        #     test_len = total_len - train_len - val_len
        #     self.data_train, self.data_val, self.data_test = random_split(
        #         dataset,
        #         [train_len, val_len, test_len],
        #         generator=torch.Generator().manual_seed(42),
        #     )
        
        # if len(self.hparams.predict_dir) != 0:
            
        #     # 初始化一个空列表来收集所有.pt文件的路径
        #     file_paths = []
        #     # 使用os.walk遍历data_dir下的所有子目录
        #     for root, dirs, files in os.walk(self.hparams.predict_dir):
        #         for file in files:
        #             # 检查文件是否以.pt结尾
        #             if file.endswith(".pt"):
        #                 # 构造完整的文件路径并添加到列表中
        #                 file_path = os.path.join(root, file)
        #                 file_paths.append(file_path)

        #     # 现在file_paths列表包含了所有子文件夹下的.pt文件路径
        #     self.flattened_tensors = [self.load_and_flatten(path) for path in file_paths]
        #     self.data_predict = LoRADataset(self.flattened_tensors)

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            dataset=self.data_predict,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
        
    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = LoRADataModule()
