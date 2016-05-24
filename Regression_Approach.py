import pandas as pd
from sklearn import linear_model
from sklearn import cross_validation

# Change url to local location of data
train_url = 'C:/Users/krazy_000/Downloads/Kaggle Digit Project/train.csv'
test_url = 'C:/Users/krazy_000/Downloads/Kaggle Digit Project/test.csv'

# Load train and test data into dataframes
train = pd.read_csv(train_url)
test = pd.read_csv(test_url)

# Seperate train data into data and target
data = train.loc[:, 'pixel0':]
target = train.loc[:, 'label']

# Split training data using cross validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(data, target, test_size=0.4, random_state=1)

# Apply simple linear regression on training data
clf = linear_model.LinearRegression()
clf.fit(X_train, y_train)

# Make predictions on test data and put into a dataframe
df = pd.DataFrame({"ImageId": range(1, test.shape[0]+1), 'Label': clf.predict(test)})

df.to_csv("submission 1.csv", index=False)

