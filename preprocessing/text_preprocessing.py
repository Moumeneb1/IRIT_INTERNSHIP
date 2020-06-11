# PREPROCESS
import re


class TextPreprocessing:

    def __init__(self, df, text_column):
        self.df = df
        self.text_column = text_column

    def remove_url(text):
        text = re.sub(r'http(\S)+', '', text)
        text = re.sub(r'http ...', '', text)
        return text

    def remove_rt(text):
        return re.sub(r'(RT|rt)[ ]*@[ ]*[\S]+', '', text)

    def remove_at(text):
        return re.sub(r'@\S+', '', text)

    def remove_extraspace(text):
        return re.sub(r' +', ' ', text)

    def replace_and(text):
        return re.sub(r'&amp;?', 'and', text)

    def replace_lt_lg(text):
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        return text

    def lower(text):
        return text.lower()

    def lower_then_k(text, k=3):
        return len(text) > k

    def remove_numero(text):
        return re.sub(r'\d+', 'numero', text)

    def remove_punctuations(text):
        return re.sub('["$#%()*+,-@./:;?![\]^_`{|}~\n\t’\']', ' ', text)

    def fit_transform(self, inplace=False, pretraitement=[
            remove_url,
            remove_rt,
            remove_at,
            replace_and,
            replace_lt_lg,
            remove_numero,
            remove_punctuations,
            remove_extraspace]):

        processed_column = self.text_column if inplace else 'processed_text'
        self.df[processed_column] = self.df[self.text_column]

        for func in pretraitement:
            self.df[processed_column] = self.df[processed_column].apply(
                lambda x: func(x))