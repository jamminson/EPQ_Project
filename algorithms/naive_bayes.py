from sklearn.naive_bayes import GaussianNB
from algorithms import evaluation


def get_naive_bayes(data):
    gnb = GaussianNB()
    gnb.fit(data[0], data[2])
    return gnb


def get_empty_models():
    sd_model = GaussianNB()
    nd_model = GaussianNB()
    ut_model = GaussianNB()

    return [sd_model, nd_model, ut_model]


def get_trained_models(standardised, normalised, unit, for_roc=False):
    if for_roc:
        roc_standardised = evaluation.roc_split(standardised)
        roc_normalised = evaluation.roc_split(normalised)
        roc_unit = evaluation.roc_split(unit)

        sd_model = get_naive_bayes(roc_standardised)
        nd_model = get_naive_bayes(roc_normalised)
        ut_model = get_naive_bayes(roc_unit)

        return [sd_model, nd_model, ut_model], [roc_standardised, roc_normalised, roc_unit]

    sd_model = get_naive_bayes(standardised)
    nd_model = get_naive_bayes(normalised)
    ut_model = get_naive_bayes(unit)

    return [sd_model, nd_model, ut_model]
