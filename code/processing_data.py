import boto3
import snowflake.connector
import pandas as pd                              
import joblib
import os
import json
import numpy as np
import pickle
import logging
import xgboost as xgb
import argparse
import sys
# sys.path.append("/opt/ml/model/code")


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

params=['/SNOWFLAKE/URL','/SNOWFLAKE/ACCOUNT_ID'
        ,'/SNOWFLAKE/USER_ID','/SNOWFLAKE/PASSWORD'
        ,'/SNOWFLAKE/DATABASE','/SNOWFLAKE/SCHEMA'
        ,'/SNOWFLAKE/WAREHOUSE','/SNOWFLAKE/BUCKET'
        ,'/SNOWFLAKE/PREFIX']
region='us-east-1'

def get_credentials(params):
    ssm = boto3.client('ssm',region)
    response = ssm.get_parameters(
       Names=params,
       WithDecryption=True
    )
    #Build dict of credentials
    param_values={k['Name']:k['Value'] for k in  response['Parameters']}
    return param_values

def snowflake_connection(param_values):
    
    # Connecting to Snowflake using the default authenticator
    ctx = snowflake.connector.connect(
      user=param_values['/SNOWFLAKE/USER_ID'],
      password=param_values['/SNOWFLAKE/PASSWORD'],
      account=param_values['/SNOWFLAKE/ACCOUNT_ID'],
      warehouse=param_values['/SNOWFLAKE/WAREHOUSE'],
      database=param_values['/SNOWFLAKE/DATABASE'],
      schema=param_values['/SNOWFLAKE/SCHEMA']
    )

    # Query Snowflake Data
    cs=ctx.cursor()
    return cs

def data_preparation(cs):
    allrows=cs.execute("""select Cust_ID,STATE,ACCOUNT_LENGTH,AREA_CODE,PHONE,INTL_PLAN,VMAIL_PLAN,VMAIL_MESSAGE,
                       DAY_MINS,DAY_CALLS,DAY_CHARGE,EVE_MINS,EVE_CALLS,EVE_CHARGE,NIGHT_MINS,NIGHT_CALLS,
                       NIGHT_CHARGE,INTL_MINS,INTL_CALLS,INTL_CHARGE,CUSTSERV_CALLS,
                       CHURN from CUSTOMER_CHURN """).fetchall()

    churn = pd.DataFrame(allrows)
    churn.columns=['Cust_id','State','Account Length','Area Code','Phone','Intl Plan', 'VMail Plan', 'VMail Message','Day Mins',
                'Day Calls', 'Day Charge', 'Eve Mins', 'Eve Calls', 'Eve Charge', 'Night Mins', 'Night Calls','Night Charge',
                'Intl Mins','Intl Calls','Intl Charge','CustServ Calls', 'Churn?']

    churn = churn.drop('Phone', axis=1)
    churn['Area Code'] = churn['Area Code'].astype(object)

    churn = churn.drop(['Day Charge', 'Eve Charge', 'Night Charge', 'Intl Charge'], axis=1)

    model_data = pd.get_dummies(churn)
    model_data = pd.concat([model_data['Churn?_True.'], model_data.drop(['Churn?_False.', 'Churn?_True.'], axis=1)], axis=1)
    to_split_data = model_data.drop(['Cust_id'], axis=1)

    return to_split_data
    


def parse_args():
    """
    Parse arguments passed from the SageMaker API
    to the container
    """

    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument("--num_round", type=int, default=6)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--eta", type=float, default=0.2)
    parser.add_argument("--objective", type=str, default="binary:logistic")
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--model_dir", type=str, default="/opt/ml/processing/output/model")
    return parser.parse_known_args()



def cross_validation(df, K, hyperparameters):
    """
    Perform cross validation on a dataset.
    :param df: pandas.DataFrame
    :param K: int
    :param hyperparameters: dict
    """
    train_indices = list(df.sample(frac=1).index)
    k_folds = np.array_split(train_indices, K)
    if K == 1:
        K = 2

    eval_list = []
    for i in range(len(k_folds)):
        training_folds = [fold for j, fold in enumerate(k_folds) if j != i]
        training_indices = np.concatenate(training_folds)
        x_train, y_train = df.iloc[training_indices, 1:], df.iloc[training_indices, :1]
        x_validation, y_validation = df.iloc[k_folds[i], 1:], df.iloc[k_folds[i], :1]
        dtrain = xgb.DMatrix(data=x_train, label=y_train)
        dvalidation = xgb.DMatrix(data=x_validation, label=y_validation)

        model = xgb.train(
            params=hyperparameters,
            dtrain=dtrain,
            evals=[(dtrain, "train"), (dvalidation, "validation")],
        )
        eval_results = model.eval(dvalidation)
        eval_list.append(float(eval_results.split("eval-error:")[1]))
    return eval_list, model


def train():
    """
    Train the PyTorch model
    """

    K = args.K

    hyperparameters = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "objective": args.objective,
        "num_round": args.num_round,
    }
    
    param_values=get_credentials(params)
    cs = snowflake_connection(param_values)
    train_df = data_preparation(cs)

    eval_list, model = cross_validation(train_df, K, hyperparameters)
    k_fold_avg = sum(eval_list) / len(eval_list)
    print(f"log loss across folds: {k_fold_avg}")

    model_location = args.model_dir + "/xgboost-model"
    pickle.dump(model, open(model_location, "wb"))
    logging.info("Stored trained model at {}".format(model_location))


if __name__ == "__main__":

    args, _ = parse_args()
    train()