import logging
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

logging.info('Importing census data.')
df = pd.read_csv('data/census.csv')

logging.info('Split dataset in train and test data.')
train, test = train_test_split(df, test_size=0.20)

logging.info('Processing train data.')
cat_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
x_train, y_train, encoder, lb = process_data(train, cat_features, 'salary', True)

logging.info('Processing test data.')
x_test, y_test, encoder, lb = process_data(test, cat_features, 'salary', False, encoder, lb)

logging.info('Training model.')
model = train_model(x_train, y_train)

logging.info('Saving model artifacts.')
with open('model/model.pkl', 'wb') as model_file:
    artifacts = {'model': model, 'encoder': encoder, 'lb': lb, 'cat_features': cat_features}
    pickle.dump(artifacts, model_file, protocol=pickle.HIGHEST_PROTOCOL)

logging.info('Scoring model.')
preds = inference(model, x_test)
precision, recall, f_beta = compute_model_metrics(y_test, preds)
logging.info(f'Precision: {precision:.3f} | Recall: {recall:.3f} | F-Beta: {f_beta:.3f}')

logging.info('Saving model metrics.')
with open('model/metrics.pkl', 'wb') as metric_file:
    metrics = {'precision': precision, 'recall': recall, 'f_beta': f_beta}
    pickle.dump(metrics, metric_file, protocol=pickle.HIGHEST_PROTOCOL)
