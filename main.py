import preprocessing
import pandas as pd

df = pd.read_csv('data.csv')

# Returns lists of datasets that have been processed
# For each list, 0: x training data, 1: x testing data, 2: y training data and 3: y testing data
data_preprocess = preprocessing.Preprocessing(df)
standardised_data, normalized_data, unit_normalized_data = data_preprocess.preprocess()
