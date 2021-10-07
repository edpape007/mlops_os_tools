from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from airflow.operators.bash import BashOperator
from airflow.operators.email import EmailOperator
# from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator

default_args = {
    "owner": "ml_admin",
    "email_on_failure": False,
    "email_on_retry": False,
    "email": "admin@localhost.com",
    "retries": 1,
    "retry_delay": timedelta(minutes=5)
}

def _choose_best_model(ti):
    print('choose best model')
    accuracies_nn = ti.xcom_pull(key='return_value', task_ids=['cancer_ml_NN'])
    accuracies_svm = ti.xcom_pull(key='return_value', task_ids=['cancer_ml_SVM'])

    accuracies = ti.xcom_pull(key='return_value', task_ids=['cancer_ml_NN','cancer_ml_SVM'])

    if accuracies_svm > accuracies_nn:
        print(f'The best model is svm with accuracy of: {accuracies_svm}')

    if accuracies_svm < accuracies_nn:
        print(f'The best model is nn with accuracy of: {accuracies_nn}')

    print(accuracies)

with DAG("cancer_breast_pipeline",
    start_date=datetime(2021, 6, 15),
    schedule_interval="@daily",
    default_args=default_args,catchup=False) as dag:

    cancer_ml_NN = BashOperator(
        task_id="cancer_ml_NN",
        bash_command="python /Users/edpape/miniforge3/envs/dags/files/CancerBreast_NN.py"
    )

    cancer_ml_SVM = BashOperator(
        task_id="cancer_ml_SVM",
        bash_command="python /Users/edpape/miniforge3/envs/dags/files/CancerBreast_SVM.py"
    )

    send_email_notification = EmailOperator(
        task_id = "send_email_notification",
        to="vivalaed@gmail.com",
        subject="MLops_Cancer_Breast",
        html_content="<h3>Your Dag run succesfully</h3>"
    )

    choose_model = PythonOperator(
        task_id='choose_model',
        python_callable=_choose_best_model
    )

    cancer_ml_NN >> choose_model
    cancer_ml_SVM >> choose_model
    choose_model >> send_email_notification

    # slack_notification = SlackWebhookOperator(
    #     task_id="slack_notification",
    #     http_conn_id='slack',
    #     message="Funciona!",
    #     channel="#monitoring"
    # )

    # task_http_sensor_check = HttpSensor(
    #     task_id='http_sensor_check',
    #     http_conn_id='http_default',
    #     endpoint='',
    #     request_params={},
    #     response_check=lambda response: "httpbin" in response.text,
    #     poke_interval=5,
    #     dag=dag,
    # )
