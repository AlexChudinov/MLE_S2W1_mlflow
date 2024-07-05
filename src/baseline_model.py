import os
from catboost import CatBoostClassifier
import click
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, log_loss, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from utils import _create_engine, create_cloud, setup_env

@click.command(
    help="Download the Sarcasm data, and generate mlflow run with baseline model"
)
@click.option('--table_name', default='sarcasm_via', type=str)
def run_create_baseline(table_name: str):
    setup_env()
    experiment_name = os.environ.get('EXPERIMENT_NAME')
    print(experiment_name)
    if mlflow.get_experiment_by_name(name=os.environ.get('EXPERIMENT_NAME')):
        experiment_id = dict(mlflow.get_experiment_by_name(name=os.environ.get('EXPERIMENT_NAME')))['experiment_id']
    else:
        mlflow.set_experiment(os.environ.get('EXPERIMENT_NAME'))
        experiment_id = dict(mlflow.get_experiment_by_name(name=os.environ.get('EXPERIMENT_NAME')))['experiment_id']
    
    mlflow_client = mlflow.tracking.MlflowClient()
    run = mlflow_client.create_run(experiment_id=experiment_id, run_name=os.environ.get('RUN_NAME')+"_baseline")

    with mlflow.start_run(description="Run baseline model", 
                          experiment_id=experiment_id, run_id=run.info.run_id, nested=True):
        # Preprocess data
        engine = _create_engine('DESTINATION')
        df = pd.read_sql(f'select * from {table_name}', engine)
        wc1, wc2 = create_cloud(df)
        mlflow.log_image(wc1.to_image(), "non_sarcastic_cloud.png")
        mlflow.log_image(wc2.to_image(), "sarcastic_cloud.png")

        # Split the data into training and test sets. (0.8, 0.20) split.
        X_train, X_test,  y_train, y_test = train_test_split(df.drop(columns='is_sarcastic'), df['is_sarcastic'], test_size=0.2)
        
        pip_requirements="./requirements.txt" 
        signature = mlflow.models.infer_signature(X_test, y_test)
        input_example = X_test[:10]
        metadata = {"model_type": "daily"}
        params  = {
            'ngram_range': (1, 2), 
            'max_df': 0.95, 
            'min_df': 2
        }
        ct = ColumnTransformer([('vect', TfidfVectorizer(), 'text')])
        model = Pipeline([
                ('transformer', ct),
                ('model', CatBoostClassifier(verbose=False))
            ]
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        precision, recall, f_score, _ = precision_recall_fscore_support(y_test, y_pred)
        logloss = log_loss(y_test, y_pred)
        _, err1, err2, _ = confusion_matrix(y_test, y_pred, normalize='all').ravel()
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="via_models",
            signature=signature,
            input_example=input_example,
            registered_model_name=os.environ.get('REGISTRY_MODEL_NAME')+'_baseline',
            await_registration_for=60,
            pip_requirements=pip_requirements,
            metadata=metadata,
        )

        mlflow.log_params(params=params)
        mlflow.log_metric("precision", precision[1])
        mlflow.log_metric("recall", recall[1])
        mlflow.log_metric("f_score", f_score[1])
        mlflow.log_metric("logloss", logloss)
        mlflow.log_metric("err1", err1)
        mlflow.log_metric("err2", err2)
        mlflow.log_metric("auc", auc)


if __name__ == '__main__':
    run_create_baseline()