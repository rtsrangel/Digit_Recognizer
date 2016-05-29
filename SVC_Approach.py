import pandas as pd
from sklearn import cross_validation
from sklearn import svm
import time

# Initialize timer
start_time = time.time()

# Change url to local location of data
train_url = 'C:/Users/krazy_000/Downloads/Kaggle Digit Project/train.csv'
test_url = 'C:/Users/krazy_000/Downloads/Kaggle Digit Project/test.csv'

# Load train and test data into dataframes
train = pd.read_csv(train_url)
test = pd.read_csv(test_url)

# Separate train data into data and target
data = train.loc[:, 'pixel0':]
target = train.loc[:, 'label']

# Split training data using cross validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(data, target, test_size=0.4, random_state=1)

# Apply SVC on the training data
clf = svm.SVC(kernel='poly', C=.01)
clf.fit(X_train, y_train)
# Print accuracy of the testing data
print clf.score(X_test, y_test)

# Make predictions on competition data and put into a dataframe
df = pd.DataFrame({"ImageId": range(1, test.shape[0]+1), 'Label': clf.predict(test)})
df = df.round(0)
df.to_csv("submission 2.csv", float_format='%.f', index=False)

# Print run time
print("--- %s seconds ---" % (time.time() - start_time))
