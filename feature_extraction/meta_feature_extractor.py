import re
import treetaggerwrapper


class FeaturesExtraction:

    def __init__(self, df, text_column):
        self.df = df
        self.text_column = text_column

    def has_url(text):
        regex_url = r'(http[s]?://|www)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        bool_result = bool(re.search(regex_url, text))
        return bool_result

    def has_img(text):
        regex_IMG = r'pic.twitter.com/[a-zA-Z0-9]+'
        bool_result = bool(re.search(regex_IMG, text))
        return bool_result

    def has_user_mention(text):
        bool_result = bool(re.search(r'@[\w_]+', text))
        return bool_result

    def has_hashtag(text):
        bool_result = bool(re.search(r'#[\w_]+', text))
        return bool_result

    def has_number(text):
        bool_result = bool(re.search(r'\d', text))
        return bool_result

    def has_exclamation_marks(text):
        bool_result = bool(re.search(r'!', text))
        return bool_result

    def fit_transform(self, features=[
        has_url,
        has_img,
        has_user_mention,
        has_number,
        has_exclamation_marks,
        has_exclamation_marks,
    ]):

        for func in features:
            self.df[func.__name__] = self.df[self.text_column].apply(
                lambda x: func(x))