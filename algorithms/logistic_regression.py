from sklearn.linear_model import LogisticRegression
from algorithms import evaluation

# all parameters not specified are set to their defaults


def get_logistic(data):
    logistic_model = LogisticRegression()

    logistic_model.fit(data[0], data[2])
    return logistic_model


def get_empty_models():
    sd_model = LogisticRegression()
    nd_model = LogisticRegression()
    ut_model = LogisticRegression()

    return [sd_model, nd_model, ut_model]


def get_trained_models(standardised, normalised, unit, for_roc=False):
    if for_roc:
        roc_standardised = evaluation.roc_split(standardised)
        roc_normalised = evaluation.roc_split(normalised)
        roc_unit = evaluation.roc_split(unit)

        sd_model = get_logistic(roc_standardised)
        nd_model = get_logistic(roc_normalised)
        ut_model = get_logistic(roc_unit)

        return [sd_model, nd_model, ut_model], [roc_standardised, roc_normalised, roc_unit]

    sd_model = get_logistic(standardised)
    nd_model = get_logistic(normalised)
    ut_model = get_logistic(unit)

    return [sd_model, nd_model, ut_model]


def implement_logistic_regression_test(standardised, normalized, unit):
    # Returns a list of the scores, 0: standardised, 1: normalized, 2: unit_normalized
    sd_logistic = get_logistic(standardised)
    sd_score = sd_logistic.score(standardised[1], standardised[3])

    nd_logistic = get_logistic(normalized)
    # normalized_logistic = LogisticRegression()
    # normalized_logistic.fit(normalized[0], normalized[2])
    nd_score = nd_logistic.score(normalized[1], normalized[3])

    ut_logistic = get_logistic(unit)
    # unit_logistic = LogisticRegression()
    # unit_logistic.fit(unit[0], unit[2])
    ut_score = ut_logistic.score(unit[1], unit[3])

    return [sd_score, nd_score, ut_score]
