# encoding: utf-8
from config import key_dict# twitter api account
import tweepy
import requests
import json


class Crawler:
    def __init__(self,token_id):
        self.token_id = token_id

    def set_api(self, token_id):
        consumer_key = key_dict[token_id]["consumer_key"]
        consumer_secret = key_dict[token_id]["consumer_secret"]
        access_key = key_dict[token_id]["access_key"]
        access_secret = key_dict[token_id]["access_secret"]
        auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        auth.set_access_token(access_key, access_secret)
        self.api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    def get_user(self, id):
        id = int(id)
        user = self.get_profile(id)
        if user['label'] == 'normal' and user['statuses_count'] > 0:
            user['tweets'] = self.get_tweets(id, user['statuses_count'])
            #network = self.get_social_network(screen_name)
            #user['friends'], user['followers'] = network['friends'], network['followers']
        return user

    def get_profile(self, id):
        try:
            profile_obj = self.api.get_user(id = id)
            profile = profile_obj._json
            clear_profile = {"id": profile['id'],
                             "created_at": profile['created_at'],
                             "screen_name": profile['screen_name'],
                             "name": profile['name'],
                             "lang": profile['lang'], # setting of language
                             "location": profile['location'],
                             "profile_location": self.get_location(profile['profile_location']), # detailed location
                             "description": profile['description'],
                             # "url": profile['url'],
                             # same content as url and description, with display_url (expanded_url)
                             #"entities_urls": [url['display_url'] for url in profile['entities']['url']['urls']],
                             #"entities_description_urls": [url['display_url'] for url in profile['entities']['description']['urls']],
                             "entities": profile['entities'],
                             "friends_count": profile['friends_count'], # following
                             "followers_count": profile['followers_count'],
                             "statuses_count": profile['statuses_count'], # tweets
                             "favourites_count": profile['favourites_count'],
                             "listed_count": profile['listed_count'],
                             "verified": profile['verified'], # an account of public interest is authentic
                             "utc_offset": profile['utc_offset'],
                             # "following": profile['following'], # if the token owner follows the user
                             # "follow_request_sent": # profile['follow_request_sent'], # if token owner has sent a follow request to the protected user
                             "notifications": profile['notifications'],
                             "geo_enabled": profile['geo_enabled'], # enable Tweet location
                             "contributors_enabled": profile['contributors_enabled'],
                             "is_translation_enabled": profile['is_translation_enabled'],
                             "translator_type": profile['translator_type'],
                             "is_translator": profile['is_translator'],
                             "time_zone": profile['time_zone'],
                             "default_profile": profile['default_profile'],
                             "has_extended_profile": profile['has_extended_profile'],
                             "profile_use_background_image": profile['profile_use_background_image'],
                             "profile_background_image_url": profile['profile_background_image_url_https'],
                             "default_profile_image": profile['default_profile_image'],
                             "profile_image_url": profile['profile_image_url_https'],
                             # "profile_background_tile": profile['profile_background_tile']  # background style: tile
                             # "profile_text_color": profile['profile_text_color'],
                             # "profile_sidebar_border_color": profile['profile_sidebar_border_color'],
                             # "profile_background_color": profile['profile_background_color'],
                             # "profile_link_color": profile['profile_link_color'],
                             # "profile_sidebar_fill_color": profile['profile_sidebar_fill_color']
                             }

            if profile['protected']:
                clear_profile['label'] = 'protected'
            else:
                clear_profile['label'] = 'normal'

            return clear_profile
        except:
            _ = 1
        # except tweepy.error.TweepError, ex:
        #     print ex.message
        #     # message: User not found.
        #     if ex.api_code == 50:
        #         return {"screen_name": screen_name, "label": 'deleted'}
        #     # message: User has been suspended.
        #     elif ex.api_code == 63:
        #         return {"screen_name": screen_name, "label": 'suspended'}
        #     # other: internal error: 16?

    def get_location(self, location):
        if not location:
            clear_location = None
        else:
            clear_location = {
                "full_name": location['full_name'],
                "country_code": location['country_code']
            }
        return clear_location

    def get_social_network(self, screen_name):
        # only for normal user
        friends = self.api.friends_ids(screen_name=screen_name)
        followers = self.api.followers_ids(screen_name=screen_name)
        return {'friends': friends, 'followers': followers}

    def get_tweets(self, screen_name, total_tweets):
        # 3200 tweets at most
        # initialize a list to hold all the tweepy Tweets
        alltweets = []

        # make initial request for most recent tweets (200 is the maximum allowed count)
        new_tweets = self.api.user_timeline(screen_name=screen_name, count=200)

        # save most recent tweets
        alltweets.extend(new_tweets)

        # keep grabbing tweets until there are no tweets left to grab
        while len(alltweets) < total_tweets and len(new_tweets) > 0:
            # update the id of the oldest tweet less one
            latest = alltweets[-1].id - 1
            #print "getting tweets before %s" % (latest)

            # all subsiquent requests use the max_id param to prevent duplicates
            new_tweets = self.api.user_timeline(screen_name=screen_name, count=200, max_id=latest)

            # save most recent tweets
            alltweets.extend(new_tweets)

            #print "...%s tweets downloaded so far" % (len(alltweets))
        clear_tweets = []
        for tweet in alltweets:
            clear_tweets.append({
                'id': tweet.id,
                'created_at': str(tweet.created_at),
                'retweet_count': tweet.retweet_count,
                'favorite_count': tweet.favorite_count,
                'is_quote_status': tweet.is_quote_status, # retweet and make comment
                'retweeted': tweet.retweeted,
                'source': tweet.source,
                "place": self.get_location(tweet._json['place']),
                'text': tweet.text.encode("utf-8"),
                'lang': tweet.lang, # detected language
                'entities': tweet.entities
                #'entity_urls': [url['display_url'] for url in tweet.entities['urls']],
                # 'entity_mentions': [mention['screen_name'] for mention in tweet.entities['user_mentions']]
            })

        return clear_tweets

# public: yeung2000
# suspended: Twicsy
# deleted: andyrwong
if __name__ == '__main__':

    crawler = Crawler(token_id=0)
    user = crawler.get_user('363426861')
    with open('test_crawler.json', 'w') as wf:
        wf.write(json.dumps(user, indent=4))