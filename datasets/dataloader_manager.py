from torch.utils.data import DataLoader

from datasets.dataset_manager import DatasetManager


class DataloaderManager:
    def __init__(
        self,
        dataset_manager: DatasetManager,
        batch_size: int,
        workers: int,
        persistent_workers: bool,
    ):
        self.dataset_manager = dataset_manager
        self.batch_size = batch_size
        self.workers = workers
        self.persistent_workers = persistent_workers

    def create_dataloaders(
        self, shuffle_train: bool = True
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        train_dataset, val_dataset, test_dataset = self.dataset_manager.datasets

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle_train,
            num_workers=self.workers,
            persistent_workers=self.persistent_workers,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            persistent_workers=self.persistent_workers,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            persistent_workers=self.persistent_workers,
        )
        return test_dataloader, val_dataloader, train_dataloader
