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

def convert_csv_to_parquet(df, output_file_path, drop_option):
    # Read CSV file into a Pandas DataFrame

    # Remove rows or columns with NaN fields based on the drop_option argument
    if drop_option == 'row':
        df = df.dropna()
    elif drop_option == 'column':
        df = df.dropna(axis=1)

    # Convert Pandas DataFrame to PyArrow Table
    table = pa.Table.from_pandas(df)

    # Write PyArrow Table to Parquet file
    pq.write_table(table, output_file_path)

    # Open the Parquet file
    table = pq.read_table(output_file_path)

    # Convert the table to a Pandas DataFrame
    df = table.to_pandas()

    # Print the DataFrame
    print(df.head(100))
    
def feature_engineer():
    input_file_path = '/home/tuananh/assignment/feature_repo/data/data_for_project.csv'
    output_file_path = '/home/tuananh/assignment/feature_repo/data/data_for_project.parquet'
    drop_option = 'column'  # options: 'row' or 'column'
    scaler = MinMaxScaler()
        
    df = pd.read_csv(input_file_path,  sep=';')
    df['HourUTC'] = df['HourUTC'].astype('datetime64[ns]')
    df['HourDK'] = df['HourDK'].astype('datetime64[ns]')

    df['HourUTC_year'] = df['HourUTC'].dt.year
    df['HourUTC_month'] = df['HourUTC'].dt.month
    df['HourUTC_day'] = df['HourUTC'].dt.day
    df['HourUTC_hour'] = df['HourUTC'].dt.hour
    df['HourUTC_minute'] = df['HourUTC'].dt.minute
    del df['HourUTC']

    df['HourDK_year'] = df['HourDK'].dt.year
    df['HourDK_month'] = df['HourDK'].dt.month
    df['HourDK_day'] = df['HourDK'].dt.day
    df['HourDK_hour'] = df['HourDK'].dt.hour
    df['HourDK_minute'] = df['HourDK'].dt.minute
    del df['HourDK']

    df.PriceArea.unique()
    df['PriceArea'].loc[df['PriceArea'] == 'DK1'] = 1
    df['PriceArea'].loc[df['PriceArea'] == 'DK2'] = 2

    df[['HourUTC_year', 'HourUTC_month', 
        'HourUTC_day', 'HourUTC_hour', 
        'HourUTC_minute', 'HourDK_year',
        'HourDK_month', 'HourDK_day',
        'HourDK_hour', 'HourDK_minute',
        'PriceArea', 'ConsumerType_DE35',
        'TotalCon']] = scaler.fit_transform(df[['HourUTC_year', 'HourUTC_month', 
                        'HourUTC_day', 'HourUTC_hour', 
                        'HourUTC_minute', 'HourDK_year',
                        'HourDK_month', 'HourDK_day',
                        'HourDK_hour', 'HourDK_minute',
                        'PriceArea', 'ConsumerType_DE35',
                        'TotalCon']])
    if (df['HourUTC_year'] == df['HourUTC_year'][0]).all():
        df = df.drop(['HourUTC_year'], axis = 1)
    if (df['HourUTC_month'] == df['HourUTC_month'][0]).all():
        df = df.drop(['HourUTC_month'], axis = 1)
    if (df['HourUTC_day'] == df['HourUTC_day'][0]).all():
        df = df.drop(['HourUTC_day'], axis = 1)
    if (df['HourUTC_hour'] == df['HourUTC_hour'][0]).all():
        df = df.drop(['HourUTC_hour'], axis = 1)
    if (df['HourUTC_minute'] == df['HourUTC_minute'][0]).all():
        df = df.drop(['HourUTC_minute'], axis = 1)
    if (df['HourDK_year'] == df['HourDK_year'][0]).all():
        df = df.drop(['HourDK_year'], axis = 1)
    if (df['HourDK_month'] == df['HourDK_month'][0]).all():
        df = df.drop(['HourDK_month'], axis = 1)
    if (df['HourDK_day'] == df['HourDK_day'][0]).all():
        df = df.drop(['HourDK_day'], axis = 1)
    if (df['HourDK_minute'] == df['HourDK_minute'][0]).all():
        df = df.drop(['HourDK_minute'], axis = 1)
    if (df['PriceArea'] == df['PriceArea'][0]).all():
        df = df.drop(['PriceArea'], axis = 1)
    if (df['ConsumerType_DE35'] == df['ConsumerType_DE35'][0]).all():
        df = df.drop(['ConsumerType_DE35'], axis = 1)
    if (df['TotalCon'] == df['TotalCon'][0]).all():
        df = df.drop(['TotalCon'], axis = 1)

    convert_csv_to_parquet(df, output_file_path, drop_option)

import sys
sys.path.insert(0, '/root/airflow/dags/hyperparam')
import train, search_hyperopt
import subprocess
import os

def training():

    os.chdir('/root/airflow/dags')
    subprocess.call(("mlflow", "run -e train --experiment-id 327287586705285493 --env-manager=local hyperparam"))

    return

# Default parameters for the workflow
default_args = {
    'depends_on_past': False,
    'owner': 'airflow',
    'start_date': datetime(2022, 11, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
        'assignment', # Name of the DAG / workflow
        default_args=default_args,
        catchup=False,
        schedule=None
) as dag:
    # This operator does nothing. 
    start_task = EmptyOperator(
        task_id='start_task', # The name of the sub-task in the workflow.
        dag=dag # When using the "with Dag(...)" syntax you could leave this out
    )
    # With the PythonOperator you can run a python function.
    feature_engineer_task = PythonOperator(
        task_id='feature_engineer',
        python_callable=feature_engineer,
        dag=dag
    )

    training_task = PythonOperator(
        task_id='training_task',
        python_callable=training,
        dag=dag
    )

    # Define the order in which the tasks are supposed to run
    # You can also define paralell tasks by using an array 
    # I.e. task1 >> [task2a, task2b] >> task3
    start_task >> feature_engineer_task >> training_task