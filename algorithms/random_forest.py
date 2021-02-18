from sklearn.ensemble import RandomForestClassifier
from algorithms import evaluation


def get_forest(data):
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(data[0], data[2])
    return clf


def get_empty_models():
    sd_model = RandomForestClassifier(max_depth=2, random_state=0)
    nd_model = RandomForestClassifier(max_depth=2, random_state=0)
    ut_model = RandomForestClassifier(max_depth=2, random_state=0)

    return [sd_model, nd_model, ut_model]


def get_trained_models(standardised, normalised, unit, for_roc=False):
    if for_roc:
        roc_standardised = evaluation.roc_split(standardised)
        roc_normalised = evaluation.roc_split(normalised)
        roc_unit = evaluation.roc_split(unit)

        sd_model = get_forest(roc_standardised)
        nd_model = get_forest(roc_normalised)
        ut_model = get_forest(roc_unit)

        return [sd_model, nd_model, ut_model], [roc_standardised, roc_normalised, roc_unit]

    sd_model = get_forest(standardised)
    nd_model = get_forest(normalised)
    ut_model = get_forest(unit)

    return [sd_model, nd_model, ut_model]
