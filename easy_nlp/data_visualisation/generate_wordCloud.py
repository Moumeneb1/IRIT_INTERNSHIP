import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def make_wordcloud(sentences):
    """
    display wordcloud of a sentence 
    :param sentence : list of sentences   
    :param        
    :returns: true if tweet has image mention False if it's not 
    """
    sentences = [word_tokenize(sentence) for sentence in sentences]
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
