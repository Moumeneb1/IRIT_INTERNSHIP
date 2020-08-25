import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pandas_profiling import ProfileReport


class Scrapper:

    def __init__(self):
        pass

    def __tweet_line(self, tt):
        """
        Convert tweeterscrapper object to dict object 
        :param tt: tweet line as twitterscrapper object   
        :returns: tweet as dict object 
        """

        tweet_line = {}

        # user name & id
        tweet_line['user'] = tt.user_id
        tweet_line['fullname'] = tt.username

        # tweet basic data
        tweet_line['id'] = tt.tweet_id
        tweet_line['url'] = tt.tweet_url
        tweet_line['timestamp_GMT0'] = tt.timestamp

        # tweet text
        tweet_line['text'] = tt.text
        tweet_line['hashtags'] = tt.hashtags
        tweet_line['links '] = tt.links

        # tweet media

        tweet_line['has_media'] = tt.has_media
        tweet_line['img_urls '] = tt.img_urls
        tweet_line['video_url '] = tt.video_url

        # tweet actions numbers

        tweet_line['likes'] = tt.likes
        tweet_line['replies'] = tt.replies
        tweet_line['retweets'] = tt.retweets

        return tweet_line

    def get_tweets_df(self, keywords, lang, begindate, enddate, limit):
        """
        scrap tweets with keywords, languages , begin date and endate and within a limit  
        :param keywords: a list of string keywords
        :param lang: a list of string keywords    
        :param begindate: datetime object of begin date     
        :param enddate: datetime object of end date
        :param limit: int limit of scrapping data      
        :returns: df of scrapped tweets with all features  
        """
        tweets = query_tweets(
            query=keywords, begindate=begindate, enddate=enddate, lang='fr', limit=limit)
        df = pd.DataFrame(list(map(lambda x: self.__tweet_line(x), tweets)))
        return df

    def __from_soup(self, tweet_div):
        """
        return tweet data 
        :param tweet_div: tweet div html object that we use to extract tweet data from  
        :returns: tweet data 
        """
        # user name & id
        screen_name = tweet_div["data-screen-name"].strip('@')
        username = tweet_div["data-name"]
        user_id = tweet_div["data-user-id"]

        # tweet basic data
        tweet_id = tweet_div["data-tweet-id"]  # equal to 'data-item-id'
        tweet_url = tweet_div["data-permalink-path"]
        timestamp_epochs = int(tweet_div.find(
            'span', '_timestamp')['data-time'])
        timestamp = datetime.datetime.utcfromtimestamp(timestamp_epochs)

        # tweet text
        soup_html = tweet_div \
            .find('div', 'js-tweet-text-container') \
            .find('p', 'tweet-text')
        text_html = str(soup_html) or ""
        text = soup_html.text or ""
        links = [
            atag.get('data-expanded-url', atag['href'])
            for atag in soup_html.find_all('a', class_='twitter-timeline-link')
            if 'pic.twitter' not in atag.text  # eliminate picture
        ]
        hashtags = [tag.strip('#')for tag in re.findall(r'#\w+', text)]

        # tweet media
        # --- imgs
        soup_imgs = tweet_div.find_all('div', 'AdaptiveMedia-photoContainer')
        img_urls = [
            img['data-image-url'] for img in soup_imgs
        ] if soup_imgs else []

        # --- videos
        video_div = tweet_div.find('div', 'PlayableMedia-container')
        video_url = video_div.find(
            'div')['data-playable-media-url'] if video_div else ''
        has_media = True if img_urls or video_url else False

        # update 'links': eliminate 'video_url' from 'links' for duplicate
        links = list(filter(lambda x: x != video_url, links))

        # tweet actions numbers
        action_div = tweet_div.find('div', 'ProfileTweet-actionCountList')

        # --- likes
        likes = int(action_div.find(
            'span', 'ProfileTweet-action--favorite').find(
            'span', 'ProfileTweet-actionCount')['data-tweet-stat-count'] or '0')
        # --- RT
        retweets = int(action_div.find(
            'span', 'ProfileTweet-action--retweet').find(
            'span', 'ProfileTweet-actionCount')['data-tweet-stat-count'] or '0')
        # --- replies
        replies = int(action_div.find(
            'span', 'ProfileTweet-action--reply u-hiddenVisually').find(
            'span', 'ProfileTweet-actionCount')['data-tweet-stat-count'] or '0')
        is_replied = False if replies == 0 else True

        # detail of reply to others
        # - reply to others
        parent_tweet_id = tweet_div['data-conversation-id']  # parent tweet

        if tweet_id == parent_tweet_id:
            is_reply_to = False
            parent_tweet_id = ''
            reply_to_users = []
        else:
            is_reply_to = True
            soup_reply_to_users = \
                tweet_div.find('div', 'ReplyingToContextBelowAuthor') \
                .find_all('a')
            reply_to_users = [{
                'screen_name': user.text.strip('@'),
                'user_id': user['data-user-id']
            } for user in soup_reply_to_users]

        return self.__tweet_line(Tweet(
            screen_name, username, user_id, tweet_id, tweet_url, timestamp,
            timestamp_epochs, text, text_html, links, hashtags, has_media,
            img_urls, video_url, likes, retweets, replies, is_replied,
            is_reply_to, parent_tweet_id, reply_to_users
        ))

    def get_tweet_byID(self, ID):
        """
        returns tweet data only using it's ID only 
        :param ID: tweet ID       
        :returns: tweet dict object   
        """
        query = "https://twitter.com/anyuser/status/"+ID
        response = requests.get(query)
        soup = BeautifulSoup(response.text, 'lxml')
        # We only retreieve the first tweet because the other are retweets
        tweet_div_list = soup.find_all('div', {"class": "tweet"})

        tweet_div = next(x for x in tweet_div_list if x["data-tweet-id"] == ID)
        tweet = self.__from_soup(tweet_div)
        return tweet