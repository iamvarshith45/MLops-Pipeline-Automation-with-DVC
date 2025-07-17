import numpy as np 
import pandas as pd
import pickle 
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

load_model = pickle.load(open('model.pkl', 'rb'))
test_data = pd.read_csv('/Users/varshithmohangadupu/Desktop/MLops_DVC/data/features/test_bow.csv')


X_test = test_data.iloc[:,0:-1].values
y_test = test_data.iloc[:, -1].values

y_pred = load_model.predict(X_test)
y_pred_proba = load_model.predict_proba(X_test)[:, 1]


# Calculate evaluation metrics

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

metrics_dict= {
    'accuracy' : accuracy,
    'precision' : precision,
    'recall' :recall,
    'auc' : auc
}


with open('metrics.json', 'w') as file: 
    json.dump(metrics_dict, file, indent = 4)

print('Model evaluation completed successfully..')
