from sklearn.model_selection import cross_validate
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import roc_curve
from matplotlib import pyplot
from sklearn.model_selection import train_test_split


def roc_split(data):
    roc_x_train, roc_x_val, roc_y_train, roc_y_val = train_test_split(data[0], data[2], test_size=0.25,
                                                                      random_state=1)
    return [roc_x_train, roc_x_val, roc_y_train, roc_y_val]


def implement_cv(empty_models, standardised, normalized, unit):
    # Returns a list of the scores, 0: standardised, 1: normalized, 2: unit_normalized

    standardised_scores = cross_validate(empty_models[0], standardised[0], standardised[2], cv=5,
                                         scoring={'f_beta': make_scorer(fbeta_score, beta=2),
                                                  'balanced': 'balanced_accuracy', 'AUC': 'roc_auc'})

    normalized_scores = cross_validate(empty_models[1], normalized[0], normalized[2], cv=5,
                                       scoring={'f_beta': make_scorer(fbeta_score, beta=2),
                                                'balanced': 'balanced_accuracy', 'AUC': 'roc_auc'})

    unit_scores = cross_validate(empty_models[2], unit[0], unit[2], cv=5,
                                 scoring={'f_beta': make_scorer(fbeta_score, beta=2), 'balanced': 'balanced_accuracy',
                                          'AUC': 'roc_auc'})

    return [standardised_scores, normalized_scores, unit_scores]


def graph_roc_curve(trained_models, data, scaling_method):
    ns_probabilities = [0 for _ in range(len(data[3]))]
    lr_probabilities = trained_models[0].predict_proba(data[1])
    lr_probabilities = lr_probabilities[:, 1]

    ns_fpr, ns_tpr, _ = roc_curve(data[3], ns_probabilities)
    lr_fpr, lr_tpr, _ = roc_curve(data[3], lr_probabilities)

    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    pyplot.title(scaling_method)
    # axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    # show the legend
    pyplot.legend()
    # show the plot
    pyplot.show()


def evaluate_cv(untrained_models, standardised, normalized, unit_normalized):

    sd_scores = list()
    nd_scores = list()
    ut_scores = list()

    scores = [sd_scores, nd_scores, ut_scores]
    model_scores = implement_cv(untrained_models, standardised, normalized, unit_normalized)
    for i in range(3):

        scores[i].append(model_scores[i]['test_f_beta'].mean())
        scores[i].append(model_scores[i]['test_balanced'].mean())
        scores[i].append(model_scores[i]['test_AUC'].mean())

    return scores
