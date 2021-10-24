from json import dump
from typing import Dict, Tuple

import torch
from torch import cuda

from model import Model
from util import DATA_DIR, MODEL_DIR, OUT_DIR


def analyze_task(model: Model, device: torch.device) -> None:
    task: Dict[int, Tuple[torch.FloatTensor,
                          torch.FloatTensor]] = torch.load(DATA_DIR.joinpath('task.pt'))
    result: Dict[int, int] = {}
    model.eval()

    with torch.no_grad():
        for index, (user_id, (user_data, tweet_data)) in enumerate(task.items()):
            user_data = user_data.to(device).unsqueeze(0)
            tweet_data = tweet_data.to(device).unsqueeze(0)
            pred_type: torch.FloatTensor = model.forward(user_data, tweet_data)[0]
            result[user_id] = pred_type.argmax().item()

            if (index + 1) % 1000 == 0:
                print(f'Finished analyzing {index + 1} users ...')

    print("Finished analyzing all users.")
    with OUT_DIR.joinpath('result.json').open('w', encoding='utf8') as f:
        dump(result, f)


if __name__ == '__main__':
    device: torch.device = torch.device('cuda' if cuda.is_available() else 'cpu')
    model: Model = Model().to(device)
    model.load_state_dict(torch.load(MODEL_DIR.joinpath('model.pt')))
    analyze_task(model, device)
