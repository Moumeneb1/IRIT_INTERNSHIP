import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt


def make_wordcloud(sentence):
    """
    display wordcloud of a sentence 
    :param sentence : text   
    :param        
    :returns: true if tweet has image mention False if it's not 
    """
    wordcloud = WordCloud(max_font_size=50, max_words=100,
                          background_color="white").generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
