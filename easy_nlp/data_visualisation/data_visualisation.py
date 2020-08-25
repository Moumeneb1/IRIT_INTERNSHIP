import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pandas_profiling import ProfileReport


class WordCloudMaker:

    def __init__(self):
        super().__init__()

    def fit(sentences):
        """
        display wordcloud of a sentence 
        :param sentence : list of sentences   
        :param        
        :returns: true if tweet has image mention False if it's not 
        """
        sentences = [word_tokenize(sentence,language='french') for sentence in sentences]
        stp_words = stopwords.words('french')

        sentences_cleaned = [
            word for sentence in sentences for word in sentence if word not in stp_words]
        sentences_cleaned = ' '.join(sentences_cleaned)

        wordcloud = WordCloud(max_font_size=50, max_words=100,
                              background_color="white").generate(sentences_cleaned)

        # Display the generated image:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
 
    def fit_export(sentences):
        """
        display wordcloud of a sentence 
        :param sentence : list of sentences   
        :param        
        :returns: true if tweet has image mention False if it's not 
        """
        sentences = [word_tokenize(sentence,language='french') for sentence in sentences]
        stp_words = stopwords.words('french')

        sentences_cleaned = [
            word for sentence in sentences for word in sentence if word not in stp_words]
        sentences_cleaned = ' '.join(sentences_cleaned)

        wordcloud = WordCloud(max_font_size=50, max_words=100,
                              background_color="white").generate(sentences_cleaned)

        # Display the generated image:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.savefig('tmp.png')

class ReportGenerator:
    def __init__(self):
        super().__init__()

    def fit(df_path, report_path):
        """
        generate report from df       
        :param df_path: dataframe path      
        :returns: null  
        """
        df = pd.read_csv(df_path)
        profile = ProfileReport(
            df, title="Pandas Profiling Report Before prepocessing")
        print("report generated on" + report_path+".html")
        profile.to_file(report_path+".html")
