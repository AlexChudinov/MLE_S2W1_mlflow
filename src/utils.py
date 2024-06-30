import os
from dotenv import load_dotenv
from typing import Tuple
import pandas as pd
import re
from string import punctuation

from wordcloud import WordCloud
from bs4 import BeautifulSoup

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from textblob import Word
nltk.download('wordnet')

stop = set(stopwords.words('english'))
stop.update(punctuation)

from warnings import filterwarnings
filterwarnings('ignore')

def preprocess(df: pd.DataFrame, lemmatize: bool = False):
    def strip_html(text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()
    # Removing the square brackets
    def remove_between_square_brackets(text):
        return re.sub('\[[^]]*\]', '', text)
    # Removing URL's
    def remove_between_square_brackets(text):
        return re.sub(r'http\S+', '', text)
    #Removing the stopwords from text
    def remove_stopwords(text):
        final_text = []
        for i in text.split():
            if i.strip().lower() not in stop:
                final_text.append(i.strip())
        return " ".join(final_text)
    # Removing the noisy text
    def denoise_text(text):
        text = strip_html(text)
        text = remove_between_square_brackets(text)
        text = remove_stopwords(text)
        return text
    
    if df is None:
        raise('DataFrame is None, but we need it!')
    # convert uppercase letters to lowercase letters
    df['text'] = df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    # delete punctuation marks
    df["text"] = df["text"].str.replace('[^\w\s]', '')
    # delete numbers
    df["text"] = df["text"].str.replace('\d','')
    # delete stopwords
    sw = stopwords.words("english")
    df['headline'] = df['headline'].apply(denoise_text)

    if lemmatize:
        df["headline"] = df["headline"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    return df
    
    
def load_data(path: str):
    if path is None:
        raise('Path to data is emty, change parameters and try again!')
    df = pd.read_json('data/Sarcasm_Headlines_Dataset_v2.json', lines=True)
    df.drop_duplicates(inplace=True)
    df.rename(columns={'headline': 'text'}, inplace=True)
    return df.drop(columns=['article_link'], inplace=True)


def create_cloud(df: pd.DataFrame):
    wc1 = WordCloud(
        width = 1600,
        height = 800,
        max_words = 2000
    ).generate(" ".join(df[df.is_sarcastic == 0].text))
    wc2 = WordCloud(
        width = 1600,
        height = 800,
        max_words = 2000
    ).generate(" ".join(df[df.is_sarcastic == 1].text))
    return wc1, wc2


def setup_env():
    load_dotenv()
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
    os.environ["AWS_BUCKET_NAME"] = str(os.environ.get('S3_BUCKET_NAME'))
    os.environ["AWS_ACCESS_KEY_ID"] = str(os.environ.get('AWS_ACCESS_KEY_ID'))
    os.environ["AWS_SECRET_ACCESS_KEY"] = str(os.environ.get('AWS_SECRET_ACCESS_KEY'))