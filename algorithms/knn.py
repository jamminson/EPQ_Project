from sklearn.neighbors import KNeighborsClassifier
from algorithms import evaluation


def get_knn(data, n_neighbors):
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(data[0], data[2])
    return knn_model


def get_empty_models(n_neighbors):
    sd_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    nd_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    ut_model = KNeighborsClassifier(n_neighbors=n_neighbors)

    return [sd_model, nd_model, ut_model]


def get_trained_models(standardised, normalised, unit, n_neighbors, for_roc=False):
    if for_roc:
        roc_standardised = evaluation.roc_split(standardised)
        roc_normalised = evaluation.roc_split(normalised)
        roc_unit = evaluation.roc_split(unit)

        sd_model = get_knn(roc_standardised, n_neighbors)
        nd_model = get_knn(roc_normalised, n_neighbors)
        ut_model = get_knn(roc_unit, n_neighbors)

        return [sd_model, nd_model, ut_model], [roc_standardised, roc_normalised, roc_unit]

    sd_model = get_knn(standardised, n_neighbors)
    nd_model = get_knn(normalised, n_neighbors)
    ut_model = get_knn(unit, n_neighbors)

    return [sd_model, nd_model, ut_model]
