from typing import Dict
import logging

import click
import numpy as np
import pandas as pd
import os
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from catboost import CatBoostClassifier
from sklearn.metrics import (
    f1_score,
    recall_score,
    accuracy_score,
    precision_score,
    roc_auc_score,
    log_loss,
)
import optuna
from optuna.trial import Trial
from sklearn.linear_model import LogisticRegression
import mlflow

from utils import setup_env, _create_engine

warnings.filterwarnings("ignore")
np.random.seed(42)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def get_vect_trial(trial: Trial, vect_type: str) -> Dict:
    # logging.info('Get vectorizetion parameters - Start')
    if vect_type == 'count_vect':
        vect_params = {
            'type': "count_vect",
            "ngram_range": trial.suggest_categorical('count_vect.ngram_range', 
                                                     [(1, 1), (1, 2)]),
            "min_df": trial.suggest_int('count_vect.min_df', 1, 5),
            "max_df": trial.suggest_float('count_vect.max_df', 0.7, 0.95)                                      
        }
    elif vect_type == 'tfidf':
        vect_params = {
            'type': "tfidf",
            "ngram_range": trial.suggest_categorical('tfidf_vect.ngram_range', 
                                                     [(1, 1), (1, 2)]),
            "min_df": trial.suggest_int('tfidf_vect.min_df', 1, 5),
            "max_df": trial.suggest_float('tfidf_vect.max_df', 0.7, 0.95),
            "sublinear_tf": trial.suggest_categorical('tfidf_vect.sublinear_tf',
                                                      [True, False])                                    
        }
    # logging.info('Get vectorizetion parameters - End')
    return vect_params

def get_trial_params(trial: Trial) -> Dict:
    # logging.info('Get full trial parameters - Start')
    vect_space = trial.suggest_categorical('vectorizer_type',
                                           ['count_vect', 'tfidf'])
    vect_params = get_vect_trial(trial=trial, vect_type=vect_space)

    model_space = trial.suggest_categorical('classifier_type',
                                            ['catboost', 'logreg'])
    if model_space == 'catboost':
        model_params = {
            'type': 'catboost',
            'max_depth': trial.suggest_int('catboost.max_depth', 2, 5),
            'n_estimators': trial.suggest_int('catboost.n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('catboost.learning_rate', 0.001, 1),
            'reg_lambda': trial.suggest_float('catboost.learning_rate', 0.001, 1),
            'auto_class_weights': trial.suggest_categorical('catboost.auto_class_weights',
                                                            ['Balanced', None]),
        }
    elif model_space == 'logreg':
        model_params = {
            'type': 'logreg',
            'class_weight': trial.suggest_categorical('logreg.class_weight',
                                                      ['balanced', None]),
            'penalty': trial.suggest_categorical('logreg.penalty',
                                                 ['l1', 'l2', None]),
            'C': trial.suggest_float('logreg.C', 0.1, 100)
        }
    
    full_space = {
        'vect': vect_params,
        'model': model_params
    }
    # logging.info('Get full trial parameters - End')
    return full_space

def get_optuna_pipeline(space: Dict):
    # logging.info('Get optuna pipeline - Start')
    vect_type = space['vect']['type']
    model_type = space['model']['type']
    del space['model']['type'], space['vect']['type']

    if vect_type == 'count_vect':
        vect = CountVectorizer(**space['vect'])
    elif vect_type == 'tfidf':
        vect = TfidfVectorizer(**space['vect'])
    else:
        logging.info(f"Don't match vect_type == {vect_type}")

    if model_type == 'catboost':
        model = CatBoostClassifier(**space['model'], verbose=False)
    elif model_type == 'logreg':
        solver = 'lbfgs'
        if space['model']['penalty'] == 'l1':
            solver = 'saga'
        model = LogisticRegression(**space['model'], solver=solver, random_state=42)
    else:
        logging.info(f"Don't match model_type == {model_type}")

    # logging.info('Get optuna pipeline - End')
    return build_pipeline(vect, model)

def build_pipeline(vect, model):
    # logging.info('Build Pipeline - Start')
    pipe = Pipeline(
        [
            ('vect', vect),
            ('model', model)
        ]
    )
    # logging.info('Build Pipeline - End')
    return pipe

def collect_optuna_metrics(trial: Trial):
    engine = _create_engine('DESTINATION')
    df = pd.read_sql(f"select * from {os.environ.get('SOURCE_TABLE_NAME')}", engine)

    experiment_name = os.environ.get('EXPERIMENT_NAME')
    if mlflow.get_experiment_by_name(name=os.environ.get('EXPERIMENT_NAME')):
        experiment_id = dict(mlflow.get_experiment_by_name(name=os.environ.get('EXPERIMENT_NAME')))['experiment_id']
    else:
        mlflow.set_experiment(os.environ.get('EXPERIMENT_NAME'))
        experiment_id = dict(mlflow.get_experiment_by_name(name=os.environ.get('EXPERIMENT_NAME')))['experiment_id']
    
    mlflow_client = mlflow.tracking.MlflowClient()
    run = mlflow_client.create_run(experiment_id=experiment_id, run_name=os.environ.get('RUN_NAME')+"_optuna_"+str(trial.number))

    with mlflow.start_run(experiment_id=experiment_id, run_id=run.info.run_id, run_name=os.environ.get('RUN_NAME')+"_optuna_"+str(trial.number), nested=True):
        X = df['text']
        y = df['is_sarcastic']
        full_space = get_trial_params(trial=trial)
        pipeline = get_optuna_pipeline(full_space.copy())
        cv = StratifiedKFold(n_splits=5)
        cv_f1_macro = []
        cv_f1_micro = []
        cv_accuracy = []
        cv_logloss = []
        cv_f1_1_score = []
        cv_precision_1_score = []
        cv_recall_1_score = []
        cv_roc_auc_score = []
        overfit_penalty = []
        for i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train = X[train_idx]
            y_train = y[train_idx]
            X_test = X[test_idx]
            y_test = y[test_idx]
            pipeline.fit(X_train, y_train)
            predict = pipeline.predict(X_test)
            overfit_penalty.append(
                f1_score(y_true=y_train, y_pred=pipeline.predict(X_train), labels=[1]) -  f1_score(y_true=y_test, y_pred=predict, labels=[1])
            )
            cv_roc_auc_score.append(roc_auc_score(y_true=y_test, y_score=pipeline.predict_proba(X_test)[:, 1]))
            cv_f1_macro.append(f1_score(y_true=y_test, y_pred=predict, average='macro'))
            cv_f1_micro.append(f1_score(y_true=y_test, y_pred=predict, average='micro'))
            cv_accuracy.append(accuracy_score(y_true=y_test, y_pred=predict))
            cv_logloss.append(log_loss(y_true=y_test, y_pred=predict))
            cv_f1_1_score.append(f1_score(y_true=y_test, y_pred=predict, labels=[1]))
            cv_precision_1_score.append(precision_score(y_true=y_test, y_pred=predict, labels=[1]))
            cv_recall_1_score.append(recall_score(y_true=y_test, y_pred=predict, labels=[1]))
        
        answer_info = {
            'overfit_penalty': np.mean(overfit_penalty),
            'cv_f1_macro': np.mean(cv_f1_macro),
            'cv_f1_micro': np.mean(cv_f1_micro),
            'cv_accuracy': np.mean(cv_accuracy),
            'logloss': np.mean(cv_logloss),
            'f_score': np.mean(cv_f1_1_score),
            'precision': np.mean(cv_precision_1_score),
            'recall': np.mean(cv_recall_1_score),
            'auc': np.mean(cv_roc_auc_score),
            'status': optuna.trial.TrialState.COMPLETE
        }
        if np.mean(cv_accuracy) or np.mean(cv_accuracy) == 1.0 or np.mean(overfit_penalty) > 0.3:
            answer_info.update({'status': optuna.trial.TrialState.FAIL})
        mlflow.log_params(full_space)
        mlflow.log_metrics(answer_info)
        # trial.set_user_attr('report', answer_info)
        # logging.info(f'End collect metrics trial {trial.number}')
    return np.mean(cv_f1_1_score)

def champion_callback(study, frozen_trial):
    """
    Logging callback that will report when a new trial iteration improves upon existing
    best trial values.

    Note: This callback is not intended for use in distributed computing systems such as Spark
    or Ray due to the micro-batch iterative implementation for distributing trials to a cluster's
    workers or agents.
    The race conditions with file system state management for distributed trials will render
    inconsistent values with this callback.
    """

    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (abs(winner - study.best_value) / study.best_value) * 100
            print(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent: .4f}% improvement"
            )
        else:
            print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")

@click.command(
    help="Download the Sarcasm data, and optimize hyperparameters"
)
@click.option('--n_trials', default=5, type=int)
@click.option('--version', default=1, type=int)
def run_optimization(n_trials: int, version: int = 1):
    setup_env()
    engine = _create_engine('DESTINATION')
    df = pd.read_sql(f"select * from {os.environ.get('SOURCE_TABLE_NAME')}", engine)
    experiment_name = os.environ.get('EXPERIMENT_NAME')
    if mlflow.get_experiment_by_name(name=experiment_name):
        experiment_id = dict(mlflow.get_experiment_by_name(name=experiment_name))['experiment_id']
        mlflow.set_experiment(experiment_id=experiment_id)
    else:
        mlflow.set_experiment(experiment_name=experiment_name)
        experiment_id = dict(mlflow.get_experiment_by_name(name=experiment_name))['experiment_id']
        mlflow.set_experiment(experiment_id=experiment_id)
    
    mlflow_client = mlflow.tracking.MlflowClient()
    run = mlflow_client.create_run(experiment_id=experiment_id, run_name=os.environ.get('RUN_NAME')+"_optuna")

    with mlflow.start_run(description="Run optimize model", 
                          experiment_id=experiment_id, run_id=run.info.run_id, nested=True):
        pip_requirements="./requirements.txt" 
        signature = mlflow.models.infer_signature(df)
        input_example = df.text.values[:10]
        metadata = {"model_type": "daily"}
        study = optuna.create_study(direction='maximize')
        study.optimize(collect_optuna_metrics, n_trials=n_trials, callbacks=[champion_callback])

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_precision", study.best_value)

        mlflow.set_tags(
            tags={
                'project': os.environ.get('EXPERIMENT_NAME'),
                'engine': 'optuna',
                'version': version
            }
        )
        def get_params_after_learning(params, key: str) -> Dict:
            _params = {}
            for _key in study.best_params.keys():
                if _key.startswith(key):
                    _key = _key.replace(key, '')
                    _params[_key] = params[key+_key]
            return _params

        if study.best_params['vectorizer_type'] == 'tfidf':
            vect_params = get_params_after_learning(study.best_params, 'tfidf_vect.')
            vect = TfidfVectorizer(**vect_params)
        else:
            vect_params = get_params_after_learning(study.best_params, 'count_vect.')
            vect = CountVectorizer(**vect_params)

        if study.best_params['classifier_type'] == 'logreg':
            model_params = get_params_after_learning(study.best_params, 'logreg.')
            model_params['solver'] = 'lbfgs'
            if model_params['penalty'] == 'l1':
                model_params['solver'] = ('saga')
            model = LogisticRegression(**model_params, random_state=42)
        else:
            model_params = get_params_after_learning(study.best_params, 'catboost.')
            model = CatBoostClassifier(**model_params, verbose=False)

        pipe = Pipeline(
            [
                ('vect', vect), 
                ('model', model)
            ]
        )
        pipe.fit(X=df.text, y=df.is_sarcastic)

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="via_models",
            signature=signature,
            input_example=input_example,
            registered_model_name=os.environ.get('REGISTRY_MODEL_NAME')+'_tune',
            await_registration_for=60,
            pip_requirements=pip_requirements,
            metadata=metadata,
        )

        mlflow.log_param('vect_type', study.best_params['vectorizer_type'])
        mlflow.log_params(vect_params)
        mlflow.log_param('model_type', study.best_params['classifier_type'])
        mlflow.log_params(model_params)


if __name__ == '__main__':
    run_optimization()