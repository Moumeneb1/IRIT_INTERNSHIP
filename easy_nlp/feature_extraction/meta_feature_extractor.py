import re
import treetaggerwrapper


class MetaFeaturesExtraction:

    def __init__(self, df, text_column):

        self.df = df
        self.text_column = text_column

    def has_url(text):
        """
        asserts if text has_url       
        :param text: text to look for urls in       
        :returns: true if text has url and false if it's not the case 
        """
        regex_url = r'(http[s]?://|www)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        bool_result = bool(re.search(regex_url, text))
        return bool_result

    def has_img(text):
        """
        asserts if tweet has img       
        :param text: text      
        :returns: true if tweet has image mention False if it's not 
        """
        regex_IMG = r'pic.twitter.com/[a-zA-Z0-9]+'
        bool_result = bool(re.search(regex_IMG, text))
        return bool_result

    def has_user_mention(text):
        """
        asserts if tweet has user_mention       
        :param text: text      
        :returns: true if tweet has user_mention False if it's not 
        """
        bool_result = bool(re.search(r'@[\w_]+', text))
        return bool_result

    def has_hashtag(text):
        """
        asserts if tweet has hashtags        
        :param text: text      
        :returns: true if tweet has_hastaghs mention False if it's not 
        """
        bool_result = bool(re.search(r'#[\w_]+', text))
        return bool_result

    def has_number(text):
        """
        asserts if tweet has number       
        :param text: text      
        :returns: true if tweet has number False if it's not 
        """
        bool_result = bool(re.search(r'\d', text))
        return bool_result

    def has_exclamation_marks(text):
        """
        asserts if tweet has exlamation marks        
        :param text: text      
        :returns: true if tweet has exclamation mark False if it's not 
        """
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
        """
        asserts a list of criteras       
        :param features : list of critera functions      
        :returns: return df with feature columns  
        """
        for func in features:
            self.df[func.__name__] = self.df[self.text_column].apply(
                lambda x: func(x))
