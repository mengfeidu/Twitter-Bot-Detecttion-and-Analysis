from typing import Dict, List, Tuple

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, random_split

from util import DATA_DIR


class BotDataset(Dataset):
    def __init__(self, data: List[Tuple[torch.FloatTensor, torch.FloatTensor, int]],
                 device: torch.device) -> None:
        self.data: List[Tuple[torch.FloatTensor, torch.FloatTensor]] = data
        self.device = device

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.FloatTensor, torch.FloatTensor, int]:
        result: Tuple[torch.FloatTensor, torch.FloatTensor, int] = self.data[index]
        return result[0].to(self.device), result[1].to(self.device), result[2]


def collate(batch: List[Tuple[torch.FloatTensor, torch.FloatTensor,
                              int]]) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                              torch.LongTensor]:
    user_batch: Tuple[torch.FloatTensor]
    tweet_batch: Tuple[torch.FloatTensor]
    type_batch: Tuple[int]
    user_batch, tweet_batch, type_batch = zip(*batch)
    max_len: int = max(x.size(0) for x in tweet_batch)

    return (torch.stack(user_batch), torch.stack(
        [torch.constant_pad_nd(x, (0, 0, 0, max_len - x.size(0))) if x.size(0) != 0
         else torch.zeros((max_len, 774)).to(user_batch[0]) for x in tweet_batch]
    ), torch.LongTensor(type_batch).to(user_batch[0].device))


def load_data(batch_size: int, device: torch.device) -> Tuple[DataLoader, DataLoader, DataLoader]:
    raw: Dict[int, Tuple[torch.FloatTensor, torch.FloatTensor,
                         int]] = torch.load(DATA_DIR.joinpath('train.pt'))

    dataset: BotDataset = BotDataset(list(raw.values()), device)
    train_size: int = round(0.8 * len(dataset))
    valid_size: int = round(0.1 * len(dataset))
    test_size: int = len(dataset) - train_size - valid_size
    datasets: List[BotDataset] = random_split(dataset, [train_size, valid_size, test_size])

    return tuple(DataLoader(x, batch_size=batch_size, shuffle=True, collate_fn=collate)
                 for x in datasets)
