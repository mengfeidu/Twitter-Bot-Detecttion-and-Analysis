from datetime import datetime
from json import load
from pathlib import Path
from random import sample, seed
from typing import Any, Dict, List, Set, Tuple

import torch
from bert_serving.client import BertClient


def parse_time(time_str: str) -> int:
    return int(datetime.strptime(time_str, '%a %b %d %H:%M:%S %z %Y').timestamp())


def num_digits(s: str) -> int:
    return sum(ord('0') <= ord(c) <= ord('9') for c in s)


def parse_user(user: Dict[str, Any]) -> Tuple[int, torch.FloatTensor]:
    user_name: str = user['name']
    # screen_name: str = user['screen_name']
    name_length: int = len(user_name)
    name_digits: int = num_digits(user_name)
    # screen_length: int = len(screen_name)
    # screen_digits: int = num_digits(screen_name)

    verified: int = int(user['verified'])
    default_profile: int = int(user.get('default_profile', 0))
    profile_use_bg: int = int(user.get('profile_use_background_image', 0))

    created_time: int = parse_time(user['created_at'])
    crawled_time: int = 1604678400
    user_age: int = (crawled_time - created_time) // 86400 + 1

    followers_count: int = user['followers_count']
    friends_count: int = user['friends_count']
    # listed_count: int = user['listed_count']
    favourites_count: int = user['favourites_count']
    statuses_count: int = user['statuses_count']

    followers_rate: float = followers_count / user_age
    friends_rate: float = friends_count / user_age
    # listed_rate: float = listed_count / user_age
    # favourites_rate: float = favourites_count / user_age
    statuses_rate: float = statuses_count / user_age

    # following_ratio: float = followers_count / friends_count

    return created_time, torch.tensor([
        statuses_rate,
        default_profile,
        name_length,
        favourites_count,
        friends_count,
        name_digits,
        verified,
        followers_count,
        profile_use_bg,
        friends_rate,
        followers_rate,
        # listed_count,
        # favourites_rate,
        # screen_digits,
        # statuses_count,
        # following_ratio,
        # screen_length,
        # listed_rate,
        # user_age
    ], dtype=torch.float)


def parse_timeline(timeline: List[Dict[str, Any]], create_time: int,
                   bert: BertClient) -> torch.FloatTensor:
    unique: Set[Tuple[int, int]] = set()
    tweets: List[Tuple[int, str, int, int, int, int, int]] = []

    for item in timeline:
        time: int = parse_time(item['time']) - create_time
        text: str = item['text']

        if (time, hash(text)) not in unique:
            mentions: int = text.count('@')
            hashtags: int = text.count('#')
            links: int = text.count('http://') + text.count('https://')
            retweet_count: int = item['retweet_count']
            favorite_count: int = item['favorite_count']

            unique.add((time, hash(text)))
            tweets.append((time, text, mentions, hashtags, links, retweet_count, favorite_count))

    if len(tweets) > 256:
        tweets = sample(tweets, 256)

    tweets = sorted(tweets, key=lambda x: x[0])
    text_vec: torch.FloatTensor = torch.tensor(bert.encode([x[1] for x in tweets]).copy())
    info_vec: torch.FloatTensor = torch.tensor([(x[0], ) + x[2:] for x in tweets],
                                               dtype=torch.float)
    return torch.cat([info_vec, text_vec], dim=1)


def parse_json(file: Path, bert: BertClient) -> Tuple[int, torch.FloatTensor, torch.FloatTensor]:
    with file.open('r', encoding='utf8') as f:
        data: Dict[str, Any] = load(f)

    create_time: int
    user_info: torch.FloatTensor
    create_time, user_info = parse_user(data['user'])
    return data['user']['id'], user_info, parse_timeline(data['timeline'], create_time, bert)


if __name__ == '__main__':
    seed(19260817)
    data_dir: Path = Path(__file__).parent.parent.joinpath('data')
    input_dir: Path = data_dir.joinpath('task')
    output_file: Path = data_dir.joinpath('task.pt')

    bert: BertClient = BertClient()
    all_data: Dict[int, Tuple[torch.FloatTensor, torch.FloatTensor]] = {}

    for index, file in enumerate(input_dir.iterdir()):
        user_id: int
        user_info: torch.FloatTensor
        timeline: torch.FloatTensor
        user_id, user_info, timeline = parse_json(file, bert)
        all_data[user_id] = user_info, timeline

        if (index + 1) % 100 == 0:
            print(f'Finished {index + 1} users ...')

    print(f'Total users: {len(all_data)}')
    print(f'Total tweets: {sum(x[1].size(0) for x in all_data.values())}')
    print('Finished processing users and tweets.')
    torch.save(all_data, output_file)
