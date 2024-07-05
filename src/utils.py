import os
from dotenv import load_dotenv
import mlflow
import pandas as pd
from string import punctuation

from wordcloud import WordCloud
from sqlalchemy import create_engine
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')

stop = set(stopwords.words('english'))
stop.update(punctuation)

from warnings import filterwarnings
filterwarnings('ignore')
  
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

def _create_engine(source: str):
    """Create engine for `source` database.
    
    For example:
    If `source = "DESTINATION"`, then load envirment from `.env` file with keys:
        - DB_DESTINATION_HOST
        - DB_DESTINATION_PORT
        - DB_DESTINATION_USER
        - DB_DESTINATION_PASSWORD
        - DB_DESTINATION_NAME
    After that create engine with `sqlalchemy.create_engine()`
    """
    load_dotenv()
    host = os.environ.get(f'DB_{source}_HOST')
    port = os.environ.get(f'DB_{source}_PORT')
    username = os.environ.get(f'DB_{source}_USER')
    password = str(os.environ.get(f'DB_{source}_PASSWORD'))
    db = os.environ.get(f'DB_{source}_NAME')
    
    return create_engine(
        f'postgresql://{username}:{password}@{host}:{port}/{db}')

def setup_env():
    load_dotenv()
    #REGION: Это не обязательная часть, которую при вызове `load_dotenv()` можно не делать
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
    os.environ["AWS_BUCKET_NAME"] = str(os.environ.get('S3_BUCKET_NAME'))
    os.environ["AWS_ACCESS_KEY_ID"] = str(os.environ.get('AWS_ACCESS_KEY_ID'))
    os.environ["AWS_SECRET_ACCESS_KEY"] = str(os.environ.get('AWS_SECRET_ACCESS_KEY'))

    os.environ["DB_DESTINATION_HOST"] = str(os.environ.get('DB_DESTINATION_HOST'))
    os.environ["DB_DESTINATION_PORT"] = os.environ.get('DB_DESTINATION_PORT')
    os.environ["DB_DESTINATION_NAME"] = str(os.environ.get('DB_DESTINATION_NAME'))
    os.environ["DB_DESTINATION_USER"] = str(os.environ.get('DB_DESTINATION_USER'))
    os.environ["DB_DESTINATION_PASSWORD"] = str(os.environ.get('DB_DESTINATION_PASSWORD'))

    os.environ["EXPERIMANT_NAME"] = str(os.environ.get('EXPERIMANT_NAME'))
    os.environ["REGISTRY_MODEL_NAME"]  = str(os.environ.get('REGISTRY_MODEL_NAME'))
    os.environ["RUN_NAME"]  = str(os.environ.get('RUN_NAME'))
    os.environ['SOURCE_TABLE_NAME'] = str(os.environ.get('SOURCE_TABLE_NAME'))
    #ENDREGION

    mlflow.set_tracking_uri(f"http://{os.environ.get('MLFLOW_SERVER_HOST')}:{os.environ.get('MLFLOW_SERVER_PORT')}")
    mlflow.set_registry_uri(f"http://{os.environ.get('MLFLOW_SERVER_HOST')}:{os.environ.get('MLFLOW_SERVER_PORT')}")

if __name__ == '__main__':
    setup_env()