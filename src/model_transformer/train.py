import os
import warnings
import argparse
from typing import Dict
from catboost import CatBoostClassifier
import pandas as pd
import numpy as np
import json

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    log_loss,
)

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.model_selection import train_test_split

import mlflow

from warnings import filterwarnings
filterwarnings('ignore')

import src.utils as p
from dotenv import load_dotenv

class ClassicModel(mlflow.pyfunc.PythonModel):
    def __init__(self, vector_params: Dict, model_params: Dict) -> None:
        super().__init__()
        self.vect_params = vector_params
        self.model_type = model_params['model_type']
        self.model_params = model_params.pop('model_type')
        ct = ColumnTransformer((['tf-idf', TfidfVectorizer(**self.vect_params), 'text']))
        self.pipeline = Pipeline(
            [
                ('ct', ct),
                (
                    'model', 
                    (
                        LGBMClassifier(**self.model_params) if 
                        self.model_params['model_type'] == 'lgbm' else 
                        (
                            XGBClassifier(**self.model_params) if 
                            self.model_params['model_type'] == 'xgb' else 
                            CatBoostClassifier(**self.model_params)))
                )
            ]

        )
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.pipeline.fit(X, y)

    def predict(self, X: pd.DataFrame):
        return self.pipeline.predict(X)
    
    def predict_proba(self, X: pd.DataFrame):
        return self.pipeline.predict_proba(X)




def main():
    EXPERIMENT_NAME = "churn_ivanvasilev"
    RUN_NAME = "model_sarcasm_classic"
    REGISTRY_MODEL_NAME = "churn_model_sarcasm"

    warnings.filterwarnings("ignore")
    np.random.seed(42)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('params_path', type=str,
                        help='Name of mlflow artifact path location to drop model.')

    args = parser.parse_args()

    
    params_path = str(args.params_path)
    if not (os.path.exists('../' + params_path) and params_path.endswith('.json')):
        raise('We need some parameters in format json!')

    with open(f'../{params_path}') as f:
        params = json.load(f)
        params = params['classic']
    
    load_dotenv()
    mlflow.set_tracking_uri(f'http://{os.environ.get('MLFLOW_SERVER_HOST')}:{os.environ.get('MLFLOW_SERVER_PORT')}')
    mlflow.set_registry_uri(f'http://{os.environ.get('MLFLOW_SERVER_HOST')}:{os.environ.get('MLFLOW_SERVER_PORT')}')

    if mlflow.get_experiment_by_name(name=EXPERIMENT_NAME):
        experiment_id = dict(mlflow.get_experiment_by_name(name=EXPERIMENT_NAME))['experiment_id']
    else:
        mlflow.set_experiment(EXPERIMENT_NAME)
        experiment_id = dict(mlflow.get_experiment_by_name(name=EXPERIMENT_NAME))['experiment_id']

    # Preprocess data
    df = p.load_data(params.get('data_path', '../data/Sarcasm_Headlines_Dataset_v2.json'))
    df = p.preprocess(df, bool(params.get('lemmatize', False)))
    wc1, wc2 = p.create_cloud(df)
    mlflow.log_image(wc1.to_image(), "non_sarcastic_cloud.png")
    mlflow.log_image(wc2.to_image(), "sarcastic_cloud.png")
    # Split the data into training and test sets. (0.75, 0.25) split.
    X_train, X_test,  y_train, y_test = train_test_split(df.drop(['is_sarcastic'], axis=1), df['is_sarcastic'], test_size=params.get("test_size", 0.2))
    
    pip_requirements="../requirements.txt" 
    signature = mlflow.models.infer_signature(X_test, y_test)
    input_example = X_test[:10]
    metadata = {"model_type": "daily"}

    with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id): # ['experiment_id']
        model = ClassicModel(params.get('vect_params', {}), params.get('model_params', {}))
        model.fit(X_train, y_train)

        y_pred = model.predict(y_test)

        precision, recall, f_score, _ = precision_recall_fscore_support(y_test, y_pred)
        logloss = log_loss(y_test, y_pred)
        _, err1, err2, _ = confusion_matrix(y_test, y_pred, normalize='all').ravel()
        auc = roc_auc_score(y_test, model.predict_proba(y_test))

        model_info = mlflow.pyfunc.log_model(
            python_model=model,
            artifact_path="via_models",
            signature=signature,
            input_example=input_example,
            registered_model_name=REGISTRY_MODEL_NAME,
            await_registration_for=60,
            pip_requirements=pip_requirements,
            metadata=metadata,
        )

        mlflow.log_param(params)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f_score", f_score)
        mlflow.log_metric("logloss", logloss)
        mlflow.log_metric("err1", err1)
        mlflow.log_metric("err2", err2)
        mlflow.log_metric("auc", auc)


if __name__ == '__main__':
    main()
