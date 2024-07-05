import os

import click

from dotenv import load_dotenv
import pandas as pd
import re, string
from string import punctuation

from bs4 import BeautifulSoup
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from textblob import Word
nltk.download('wordnet')

from utils import _create_engine, setup_env

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
    
    stop = set(stopwords.words('english'))
    punctuation = list(string.punctuation)
    stop.update(punctuation)

    # convert uppercase letters to lowercase letters
    df['text'] = df['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    # delete punctuation marks
    df["text"] = df["text"].str.replace('[^\w\s]', '')
    # delete numbers
    df["text"] = df["text"].str.replace('\d','')
    # delete stopwords
    sw = stopwords.words("english")
    df['text'] = df['text'].apply(denoise_text)

    if lemmatize:
        df["text"] = df["text"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    return df

@click.command(
    help="Download the Sarcasm data, prepare and load to Databse"
)
@click.option('--path', default='./data/Sarcasm_Headlines_Dataset_v2.json', type=str)
@click.option('--table_name', default='sarcasm_via', type=str)
@click.option('--lemmatize', default=False, type=bool)
def prepare_data(path: str, lemmatize: bool, table_name: str = 'sarcasm_data'):
    setup_env()
    df = pd.read_json(path, lines=True)
    df.drop(columns='article_link', inplace=True)
    df.rename(columns={'headline': 'text'}, inplace=True)

    df = preprocess(df, lemmatize=bool(lemmatize))

    engine = _create_engine('DESTINATION')
    df.to_sql(
        table_name,
        con=engine,
        index=True,
        if_exists='replace',
    )

if __name__ == '__main__':
    prepare_data()
