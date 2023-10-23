from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from airflow.operators.bash import BashOperator

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pipelines import *

with DAG(
        'train_hyperopt', # Name of the DAG / workflow
        default_args=default_args,
        catchup=False,
        schedule=None
) as dag:
    # This operator does nothing. 
    start_task = EmptyOperator(
        task_id='start_task', # The name of the sub-task in the workflow.
        dag=dag # When using the "with Dag(...)" syntax you could leave this out
    )
    
    training_individual_task = PythonOperator(
        task_id='training_individual_task',
        python_callable=training_individual,
        dag=dag
    )

    # Define the order in which the tasks are supposed to run
    # You can also define paralell tasks by using an array 
    # I.e. task1 >> [task2a, task2b] >> task3
    start_task >> training_individual_task