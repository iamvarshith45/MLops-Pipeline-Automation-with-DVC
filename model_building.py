import numpy as np 
import pandas as pd 
import pickle

from sklearn.ensemble import GradientBoostingClassifier


# fetch the data from data/pre_processed
train_data = pd.read_csv('/Users/varshithmohangadupu/Desktop/MLops_DVC/data/features/train_bow.csv')


X_train = train_data.iloc[:,0:-1].values
y_train = train_data.iloc[:,-1].values

# Define and train your choosen model

model = GradientBoostingClassifier(n_estimators=50)
model.fit(X_train, y_train)

# save the trained model
pickle.dump(model,open('model.pkl', 'wb'))

print('Model trained and saved successfully..')