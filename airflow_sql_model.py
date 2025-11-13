Apache Airflow → SQL database → scikit-learn model

Connects to a SQL database (e.g., PostgreSQL).

Runs a query to fetch data.

Loads that data into a model 
Logs the model’s predictions.

Install dependencies
pip install apache-airflow[postgres] psycopg2-binary pandas scikit-learn


from airflow import DAG
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
from sklearn.linear_model import LinearRegression
import logging

# DAG definition
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 1, 1),
    'retries': 1
}

dag = DAG(
    'sql_to_model_dag',
    default_args=default_args,
    description='Fetch data from SQL and run ML model',
    schedule_interval=None,
    catchup=False
)

# Step 1: Query the SQL database
def fetch_data_from_sql():
    hook = PostgresHook(postgres_conn_id='postgres_default')
    sql = """
        SELECT feature1, feature2, target
        FROM training_data
        WHERE feature1 IS NOT NULL AND feature2 IS NOT NULL;
    """
    df = hook.get_pandas_df(sql)
    logging.info(f"Fetched {len(df)} records from database")
    df.to_csv('/tmp/sql_data.csv', index=False)
    return '/tmp/sql_data.csv'

# Step 2: Train or run the ML model
def run_model(**context):
    data_path = context['ti'].xcom_pull(task_ids='fetch_data')
    df = pd.read_csv(data_path)
    
    X = df[['feature1', 'feature2']]
    y = df['target']
    
    model = LinearRegression()
    model.fit(X, y)
    
    predictions = model.predict(X)
    df['predictions'] = predictions
    
    # Log or save predictions
    logging.info("Model training completed. Sample predictions:")
    logging.info(df.head())
    
    df.to_csv('/tmp/model_predictions.csv', index=False)

# Define tasks
fetch_data = PythonOperator(
    task_id='fetch_data',
    python_callable=fetch_data_from_sql,
    dag=dag
)

model_task = PythonOperator(
    task_id='run_model',
    python_callable=run_model,
    provide_context=True,
    dag=dag
)

# Task order
fetch_data >> model_task

What this DAG does

fetch_data task: Connects to PostgreSQL using Airflow’s PostgresHook, runs a SQL query, and saves data locally.

run_model task: Reads that CSV, runs a simple LinearRegression, and logs predictions.

