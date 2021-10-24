from collections import defaultdict
from csv import DictReader
from datetime import datetime
from pathlib import Path
from random import sample, seed
from typing import DefaultDict, Dict, List, Set, Tuple

import torch
from bert_serving.client import BertClient

type_path: List[List[str]] = [
    ['genuine_accounts.csv'],
    ['fake_followers.csv'],
    [f'social_spambots_{i}.csv' for i in range(1, 4)],
    [f'traditional_spambots_{i}.csv' for i in range(1, 5)]
]


def to_int(text: str) -> int:
    try:
        return int(text)
    except Exception:
        return 0


def num_digits(s: str) -> int:
    return sum(ord('0') <= ord(c) <= ord('9') for c in s)


def parse_time(time_str: str) -> int:
    if time_str[-1] == 'L':
        return int(time_str[:-1]) // 1000
    return int(datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S').timestamp())


def parse_time2(time_str: str) -> int:
    if time_str[-1] == 'L':
        return int(time_str[:-1]) // 1000
    return int(datetime.strptime(time_str, '%a %b %d %H:%M:%S %z %Y').timestamp())


def parse_user(row: Dict[str, str]) -> Tuple[int, int, torch.FloatTensor]:
    user_id: int = to_int(row['id'])

    user_name: str = row['name']
    # screen_name: str = row['screen_name']
    name_length: int = len(user_name)
    name_digits: int = num_digits(user_name)
    # screen_length: int = len(screen_name)
    # screen_digits: int = num_digits(screen_name)

    verified: int = to_int(row['verified'])
    default_profile: int = to_int(row.get('default_profile', '0'))
    profile_use_bg: int = to_int(row.get('profile_use_background_image', '0'))

    created_time: int = parse_time2(row['created_at'])
    crawled_time: int = parse_time(row.get('crawled_at', row['updated']))
    user_age: int = (crawled_time - created_time) // 86400 + 1

    followers_count: int = to_int(row['followers_count'])
    friends_count: int = to_int(row['friends_count'])
    # listed_count: int = to_int(row['listed_count'])
    favourites_count: int = to_int(row['favourites_count'])
    statuses_count: int = to_int(row['statuses_count'])

    followers_rate: float = followers_count / user_age
    friends_rate: float = friends_count / user_age
    # listed_rate: float = listed_count / user_age
    # favourites_rate: float = favourites_count / user_age
    statuses_rate: float = statuses_count / user_age

    # following_ratio: float = (followers_count + 1) / (friends_count + 1)

    return user_id, created_time, torch.tensor([
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


def parse_tweet(row: Dict[str, str]) -> Tuple[int, int, Tuple[int, str, int, int, int, int, int]]:
    text: str = row['text'].replace('\xa0', '')
    mentions: int = text.count('@')
    hashtags: int = text.count('#')
    links: int = text.count('http://') + text.count('https://')

    created_time: int = parse_time2(row['created_at'])
    retweet_count: int = to_int(row['retweet_count'])
    favorite_count: int = to_int(row['favorite_count'])

    return to_int(row['id']), to_int(row['user_id']), (created_time, text, mentions, hashtags,
                                                       links, retweet_count, favorite_count)


def parse_paths(input_dir: Path, user_type: int,
                paths: List[str]) -> Tuple[Dict[int, Tuple[int, int, torch.FloatTensor]],
                                           DefaultDict[int, List[Tuple[int, str, int, int,
                                                                       int, int, int]]]]:
    all_users: Set[int] = set()
    user_data: Dict[int, Tuple[int, int, torch.FloatTensor]] = {}

    for path in paths:
        print(f'Processing users in {path} ...')
        users_file: Path = input_dir.joinpath(path).joinpath('users.csv')

        with users_file.open('r', encoding='utf8', errors='ignore') as f:
            reader: DictReader = DictReader(x.replace('\0', '') for x in f)

            for row in reader:
                user_id: int
                created_time: int
                user_info: torch.FloatTensor
                user_id, created_time, user_info = parse_user(row)
                user_data[user_id] = user_type, created_time, user_info
                all_users.add(user_id)

        print(f'Finished processing users in {path}.')

    print(f'Total users of type {user_type}: {len(all_users)}')
    all_tweets: DefaultDict[int, Set[int]] = defaultdict(set)
    tweet_data: DefaultDict[int, List[Tuple[int, str, int, int, int, int, int]]] = defaultdict(list)

    for path in paths:
        print(f'Processing tweets in {path} ...')
        tweets_file: Path = input_dir.joinpath(path).joinpath('tweets.csv')

        if tweets_file.exists():
            with tweets_file.open('r', encoding='utf8', errors='ignore') as f:
                reader: DictReader = DictReader(x.replace('\0', '') for x in f)

                for index, row in enumerate(reader):
                    if not row['id'].isnumeric():
                        continue

                    tweet_id: int
                    user_id: int
                    tweet_info: Tuple[int, str, int, int]
                    tweet_id, user_id, tweet_info = parse_tweet(row)

                    if user_id in all_users and tweet_id not in all_tweets[user_id] and \
                            tweet_info[1] != '' and not tweet_info[1].isspace():
                        tweet_data[user_id].append(tweet_info)
                        all_tweets[user_id].add(tweet_id)

                    if (index + 1) % 100000 == 0:
                        print(f'Finished {index + 1} tweets ...')

        print(f'Finished processing tweets in {path}.')

    print(f'Total tweets of type {user_type}: {sum(len(x) for x in all_tweets.values())}')
    print(f'Finished processing type {user_type}.')
    return user_data, tweet_data


def parse_all_tweets(tweets: List[Tuple[int, str, int, int, int, int, int]], created_time: int,
                     bert: BertClient) -> torch.FloatTensor:
    if len(tweets) == 0:
        return torch.tensor([])
    if len(tweets) > 256:
        tweets = sample(tweets, 256)

    tweets = sorted(tweets, key=lambda x: x[0])
    text_vec: torch.FloatTensor = torch.tensor(bert.encode([x[1] for x in tweets]).copy())
    info_vec: torch.FloatTensor = torch.tensor([(x[0] - created_time, ) + x[2:] for x in tweets],
                                               dtype=torch.float)

    return torch.cat([info_vec, text_vec], dim=1)


if __name__ == '__main__':
    seed(19260817)
    data_dir: Path = Path(__file__).parent.parent / 'data'
    input_dir: Path = data_dir / 'train'
    output_file: Path = data_dir / 'train.pt'

    all_user_data: Dict[int, Tuple[int, int, torch.FloatTensor]] = {}
    all_tweet_data: DefaultDict[int, List[Tuple[int, str, int, int,
                                                int, int, int]]] = defaultdict(list)

    for user_type, paths in enumerate(type_path):
        cur_user_data: Dict[int, Tuple[int, int, torch.FloatTensor]]
        cur_tweet_data: DefaultDict[int, List[Tuple[int, str, int, int, int, int, int]]]
        cur_user_data, cur_tweet_data = parse_paths(input_dir, user_type, paths)
        all_user_data.update(cur_user_data)
        all_tweet_data.update(cur_tweet_data)

    torch.save((all_user_data, all_tweet_data), data_dir / 'med.pt')
    # all_user_data, all_tweet_data = torch.load(data_dir / 'med.pt')

    bert: BertClient = BertClient()
    all_data: Dict[int, Tuple[torch.FloatTensor, torch.FloatTensor, int]] = {}

    for index, (user_id, (user_type, created_time, user_info)) in enumerate(all_user_data.items()):
        all_data[user_id] = user_info, parse_all_tweets(all_tweet_data[user_id], created_time,
                                                        bert), user_type
        if (index + 1) % 100 == 0:
            print(f'Combined {index + 1} users and their tweets ...')

    print(f'Total users: {len(all_data)}')
    print(f'Total tweets: {sum(x[1].size(0) for x in all_data.values())}')
    print('Finished processing users and tweets.')
    torch.save(all_data, output_file)
