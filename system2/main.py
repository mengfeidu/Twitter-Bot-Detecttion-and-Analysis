import torch
from torch import cuda

from loader import DataLoader, load_data
from model import Model
from task import analyze_task
from trainer import Trainer
from util import prepare

if __name__ == '__main__':
    prepare()
    device: torch.device = torch.device('cuda' if cuda.is_available() else 'cpu')

    train: DataLoader
    valid: DataLoader
    test: DataLoader
    train, valid, test = load_data(16, device)

    model: Model = Model().to(device)
    trainer: Trainer = Trainer('model', model, {
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'step_size': 10,
        'lr_decay': 1,
        'gradient_clip': 1,
    })

    trainer.train(train, valid, test, 10, -1)
    analyze_task(model, device)
