from sklearn import svm
from algorithms import evaluation


def get_svm(data):

    svm_model = svm.SVC(kernel='rbf', probability=True)
    svm_model.fit(data[0], data[2])
    return svm_model


def get_empty_models():
    sd_model = svm.SVC(kernel='rbf', probability=True)
    nd_model = svm.SVC(kernel='rbf', probability=True)
    ut_model = svm.SVC(kernel='rbf', probability=True)

    return [sd_model, nd_model, ut_model]


def get_trained_models(standardised, normalised, unit, for_roc=False):
    if for_roc:
        roc_standardised = evaluation.roc_split(standardised)
        roc_normalised = evaluation.roc_split(normalised)
        roc_unit = evaluation.roc_split(unit)

        sd_model = get_svm(roc_standardised)
        nd_model = get_svm(roc_normalised)
        ut_model = get_svm(roc_unit)

        return [sd_model, nd_model, ut_model], [roc_standardised, roc_normalised, roc_unit]

    sd_model = get_svm(standardised)
    nd_model = get_svm(normalised)
    ut_model = get_svm(unit)

    return [sd_model, nd_model, ut_model]
