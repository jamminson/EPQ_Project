from keras.models import Sequential
from keras.layers import Dense
from algorithms import evaluation
import keras.metrics


def define_empty_model():
    metrics = [keras.metrics.TruePositives(name='tp'), keras.metrics.FalsePositives(name='fp'),
               keras.metrics.TrueNegatives(name='tn'), keras.metrics.FalseNegatives(name='fn'),
               keras.metrics.AUC(name='AUC')]

    model = Sequential()
    model.add(Dense(12, input_dim=30, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)

    return model


def get_feed_forward(data):
    model = define_empty_model()
    model.fit(data[0], data[2], epochs=150, batch_size=10, verbose=0)

    return model


def get_empty_models():
    sd_model = define_empty_model()
    nd_model = define_empty_model()
    ut_model = define_empty_model()

    return [sd_model, nd_model, ut_model]


def get_trained_models(standardised, normalised, unit, for_roc=False):
    if for_roc:
        roc_standardised = evaluation.roc_split(standardised)
        roc_normalised = evaluation.roc_split(normalised)
        roc_unit = evaluation.roc_split(unit)

        sd_model = get_feed_forward(roc_standardised)
        nd_model = get_feed_forward(roc_normalised)
        ut_model = get_feed_forward(roc_unit)

        return [sd_model, nd_model, ut_model], [roc_standardised, roc_normalised, roc_unit]

    sd_model = get_feed_forward(standardised)
    nd_model = get_feed_forward(normalised)
    ut_model = get_feed_forward(unit)

    return [sd_model, nd_model, ut_model]
