from sklearn import linear_model
import matplotlib.pyplot as pl
import numpy as np
import pandas as pd

# Change url to local location of data
train_url = 'C:/Users/krazy_000/Downloads/Kaggle Digit Project/train.csv'
test_url = 'C:/Users/krazy_000/Downloads/Kaggle Digit Project/test.csv'

# Load train and test data into dataframes
df = pd.read_csv(train_url)
# test_df = pd.read_csv(test_url)

# Convert pandas dataframe into a 2D numpy array
np_df = df.as_matrix()

# Delete all columns
labels = np.delete(np_df, np.s_[1::], 1)
features = np.delete(np_df, np.s_[0:1], 1)

# Count the number of colored pixels per image
color_count = labels.copy()
for x in range(0, len(features)):
    color_count[x] = (np.count_nonzero(features[x]))

data = np.concatenate((labels, color_count), axis=1)

idata = np.arange(10).reshape(10, 1)

column_names = ['Col1', 'Col2']
index = ['Row'+str(i) for i in range(1, len(data)+1)]

df = pd.DataFrame(data, index=index, columns=column_names)
for _, value in df.iterrows():
     if value['Col1'] = 1:
         

pl.figure(figsize=(7, 5))
pl.title('Number vs Color Count')
pl.scatter(labels, color_count)
pl.xlabel('Number')
pl.ylabel('Counts')
#pl.show()
