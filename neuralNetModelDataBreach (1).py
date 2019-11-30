# coding: utf-8

# packages
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import confusion_matrix

# import data
data = pd.read_csv('kaggleSubsetData.csv')

print(data.head())

# create a list element for variable columns to be become dummy variable
dummy_cols = []

# creating data subset
sample_set = data[['Year', 'Records Lost', 'Sector', 'Method of Leak']].copy(deep=True)

# binned variable records lost into a binary of low = 0, high = 1 for risk as expressed by the amount of records lost
sample_set['catRisk'] = pd.cut(x=data['Records Lost'], bins=2, labels=['0', '1'])

# dropped records column
sample_set.drop('Records Lost', axis=1, inplace=True)

# iterate over variables and if not col year or category of risk then convert to dummy variables
for col in list(sample_set.columns):
    if col not in ['Year', 'catRisk']:
        dummy_vars = pd.get_dummies(sample_set[col])
        dummy_vars.columns = [col+str(x) for x in dummy_vars.columns]
        sample_set = pd.concat([sample_set, dummy_vars], axis=1)

# dropping no longer needed cols
sample_set.drop(['Sector', 'Method of Leak'], axis=1, inplace=True)

# assigned col header for target to variable target_var and then seperated it from the sub dataset to create a feature set
target_var = 'catRisk'
features = [x for x in list(sample_set.columns) if x != target_var]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(sample_set['catRisk'])
encoded_Y = encoder.transform(sample_set['catRisk'])

# baseline model


def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(60, input_dim=len(features), kernel_initializer='normal', activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# splitting data into train and test
X_train, X_test, y_train, y_test = train_test_split(sample_set[features], encoded_Y, test_size=0.1)

# evaluating model performance (accuracy) with  data set train and 3 k-folds
estimator = KerasClassifier(build_fn=create_baseline, epochs=100, batch_size=50, verbose=0)
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=21)
results = cross_val_score(estimator, X_train, y_train, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# evaluting model performance (accuracy) with test data and 5 k-folds
results = cross_val_score(estimator, X_test, y_test, cv=5)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

# RANDOM FOREST MODEL UPDATE

# Display the dimensions of your Training and Testing Data
print('X Train | X Test:')
print(X_train.shape, X_test.shape)
print('-'*50)
print('Y Train | Y Test:')
print(y_train.shape, y_test.shape)

# Create the Random Forest Classifier
classifier = RandomForestClassifier(random_state=5)

print(classifier)

# Compute k-fold cross validation on training data set and see mean accuracy score
scores = cross_val_score(classifier,X_train, y_train, cv=2, scoring='accuracy')
print(f'Training Data Accuracy Scores : {scores}')
print('-'*50)
print(f'Training Data Mean Accuracy Score : {scores.mean()}')

# Fit the model to training data
classifier.fit(X_train, y_train)
prediction = classifier.predict(X_test)

# Display scores
print(f'Accuracy Score : {accuracy_score(y_test, prediction)}')
print('-'*50)
print(f'Log Loss : {log_loss(y_test, prediction)}')

# describe the performance of a classifier using a confusion matrix table: 'TN' 'FN', 'FP' 'TP'
print(confusion_matrix(y_test, prediction))

