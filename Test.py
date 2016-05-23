from sklearn import tree
import numpy as np
import pandas as pd

# Change url to local location of data
train_url = 'C:/Users/krazy_000/Downloads/Kaggle Digit Project/train.csv'
test_url = 'C:/Users/krazy_000/Downloads/Kaggle Digit Project/test.csv'

#Load train and test data into dataframes
train_df = pd.read_csv(train_url)
test_df = pd.read_csv(test_url)


