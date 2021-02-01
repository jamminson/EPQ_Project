import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer


# Standardisation
def standardise(data):
    sc_x = StandardScaler()
    sc_x = sc_x.fit_transform(data)
    # Convert to table format - StandardScaler
    sc_x = pd.DataFrame(data=sc_x, columns=data.columns)
    return sc_x


# Normalization
def normalize(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    scaled_features = scaler.transform(data)
    # Convert to table format - MinMaxScaler
    norm_x = pd.DataFrame(data=scaled_features, columns=data.columns)
    return norm_x


# Unit Vector Normalization
def unit_normalize(data):
    transformer = Normalizer()
    transformer.fit(data)
    transformed_features = transformer.transform(data)
    unit_norm_x = pd.DataFrame(data=transformed_features, columns=data.columns)
    return unit_norm_x


class Preprocessing:

    # Initialise Variables
    def __init__(self, data):
        self.x = data
        self.y = None

        self.raw_x_train = None
        self.raw_x_test = None
        self.raw_y_train = None
        self.raw_y_test = None

        self.standard_x_train = None
        self.standard_x_test = None
        self.standard_y_train = None
        self.standard_y_test = None

        self.norm_x_train = None
        self.norm_x_test = None
        self.norm_y_train = None
        self.norm_y_test = None

        self.unit_x_train = None
        self.unit_x_test = None
        self.unit_y_train = None
        self.unit_y_test = None

    # Delete unneeded columns
    def delete_columns(self):
        del self.x['Unnamed: 32']
        del self.x['id']

    # Encode diagnosis feature vector and create y vector
    def encode_features(self):
        le = LabelEncoder()
        le.fit(self.x['diagnosis'])
        self.y = le.transform(self.x['diagnosis'])
        del self.x['diagnosis']

    # Split dataset into training and test sets
    def split_data(self):
        self.raw_x_train, self.raw_x_test, self.raw_y_train, self.raw_y_test = train_test_split(self.x, self.y,
                                                                                                test_size=0.3,
                                                                                                random_state=4)

    # Feature Scaling
    def scale_features(self):
        raw_data = [self.raw_x_train, self.raw_x_test, self.raw_y_train, self.raw_y_test]
        standard_data = [self.standard_x_train, self.standard_x_test, self.standard_y_train, self.standard_y_test]
        norm_data = [self.norm_x_train, self.norm_x_test, self.norm_y_train, self.norm_y_test]
        unit_data = [self.unit_x_train, self.unit_x_test, self.unit_y_train, self.unit_y_test]

        for i in range(2):
            standard_data[i] = standardise(raw_data[i])
            norm_data[i] = normalize(raw_data[i])
            unit_data[i] = unit_normalize(raw_data[i])

        for i in range(2, 4):
            standard_data[i] = raw_data[i]
            norm_data[i] = raw_data[i]
            unit_data[i] = raw_data[i]

        return standard_data, norm_data, unit_data

    # For each list, 0: x training data, 1: x testing data, 2: y training data and 3: y testing data

    def preprocess(self):
        # Returns lists of datasets that have been processed
        # For each list, 0: x training data, 1: x testing data, 2: y training data and 3: y testing data
        self.delete_columns()
        self.encode_features()
        self.split_data()
        standard_data, norm_data, unit_data = self.scale_features()
        return [standard_data, norm_data, unit_data]
