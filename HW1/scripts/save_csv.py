# A code snippet to help you save your results into a kaggle accepted csv
import pandas as pd
import numpy as np

def results_to_csv(y_test, file_name):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1
    df.to_csv(file_name, index_label='Id')


if __name__=="__main__":

    y_test=pd.read_csv('./../mnist_predictions.txt',header=None)
    y_test.columns=['Category']
    print(y_test.head(10))
    y_test['Id'] = range(1, len(y_test) + 1)  # Id从1开始
    y_test = y_test[['Id', 'Category']]
    y_test=y_test.head(1000)
    y_test.to_csv('./../mnist_predictions.csv',index=False)
    # results_to_csv(y_test,'./../mnist_predictions.csv')

    y_test=pd.read_csv('./../spam_predictions.txt',header=None)
    y_test.columns=['Category']
    print(y_test.head(10))
    y_test['Id'] = range(1, len(y_test) + 1)  # Id从1开始
    y_test = y_test[['Id', 'Category']]
    y_test=y_test.head(1000)
    y_test.to_csv('./../spam_predictions.csv',index=False)
    # results_to_csv(y_test,'./../spam_predictions.csv')