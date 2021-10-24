from os import environ
from pathlib import Path
from subprocess import run

import torch

ROOT_DIR: Path = Path(__file__).parent.parent
DATA_DIR: Path = ROOT_DIR.joinpath('data')
MODEL_DIR: Path = ROOT_DIR.joinpath('model')
LOG_DIR: Path = ROOT_DIR.joinpath('log')
OUT_DIR: Path = ROOT_DIR.joinpath('out')


def sort_gpu() -> None:
    output: str = run('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free', capture_output=True,
                      shell=True, encoding='utf8').stdout
    memory: torch.LongTensor = torch.LongTensor([int(line.strip().split()[2])
                                                 for line in output.strip().split('\n')])

    gpu_order: str = ','.join(map(str, torch.argsort(memory, descending=True).tolist()))
    environ.setdefault('CUDA_VISIBLE_DEVICES', gpu_order)


def prepare() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sort_gpu()
