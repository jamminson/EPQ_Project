import pandas as pd
from EDA_Intial_Preprocessing import preprocessing
from algorithms import evaluation
from algorithms import feed_forward

df = pd.read_csv('data.csv')
preprocessing_index = ['Standardised', 'Normalized', 'Unit Normalized']
algorithm_index = ['logistic_regression', 'svm', 'naive_bayes', 'feed_forward', 'K-Nearest Neighbours','decision_tree', 'random_forest']

data_preprocess = preprocessing.Preprocessing(df)
data = data_preprocess.preprocess()
standardised_data = data[0]
normalized_data = data[1]
unit_normalized_data = data[2]

ff_models = feed_forward.get_empty_models()
ff_scores = evaluation.evaluate_cv(ff_models, standardised_data, normalized_data, unit_normalized_data, 5, ff=True)
