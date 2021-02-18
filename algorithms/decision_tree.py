from sklearn import tree
from algorithms import evaluation


def get_tree(data):
    clf = tree.DecisionTreeClassifier()
    clf.fit(data[0], data[2])
    return clf


def get_empty_models():
    sd_model = tree.DecisionTreeClassifier()
    nd_model = tree.DecisionTreeClassifier()
    ut_model = tree.DecisionTreeClassifier()

    return [sd_model, nd_model, ut_model]


def get_trained_models(standardised, normalised, unit, for_roc=False):
    if for_roc:
        roc_standardised = evaluation.roc_split(standardised)
        roc_normalised = evaluation.roc_split(normalised)
        roc_unit = evaluation.roc_split(unit)

        sd_model = get_tree(roc_standardised)
        nd_model = get_tree(roc_normalised)
        ut_model = get_tree(roc_unit)

        return [sd_model, nd_model, ut_model], [roc_standardised, roc_normalised, roc_unit]

    sd_model = get_tree(standardised)
    nd_model = get_tree(normalised)
    ut_model = get_tree(unit)

    return [sd_model, nd_model, ut_model]
