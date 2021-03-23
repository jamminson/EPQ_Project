from sklearn.model_selection import cross_validate
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from statistics import mean


def roc_split(data):
    roc_x_train, roc_x_val, roc_y_train, roc_y_val = train_test_split(data[0], data[2], test_size=0.25,
                                                                      random_state=1)
    return [roc_x_train, roc_x_val, roc_y_train, roc_y_val]


def implement_cv(empty_models, standardised, normalized, unit, splits):
    # Returns a list of the scores, 0: standardised, 1: normalized, 2: unit_normalized

    standardised_scores = cross_validate(empty_models[0], standardised[0], standardised[2], cv=splits,
                                         scoring={'f_beta': make_scorer(fbeta_score, beta=2),
                                                  'balanced': 'balanced_accuracy', 'AUC': 'roc_auc'})

    normalized_scores = cross_validate(empty_models[1], normalized[0], normalized[2], cv=splits,
                                       scoring={'f_beta': make_scorer(fbeta_score, beta=2),
                                                'balanced': 'balanced_accuracy', 'AUC': 'roc_auc'})

    unit_scores = cross_validate(empty_models[2], unit[0], unit[2], cv=splits,
                                 scoring={'f_beta': make_scorer(fbeta_score, beta=2), 'balanced': 'balanced_accuracy',
                                          'AUC': 'roc_auc'})

    return [standardised_scores, normalized_scores, unit_scores]


def implement_cv_ff(models, whole_data, splits):
    ff_scores = list()

    for feature_scaling_method in range(3):
        ff_scores.append(list())

        for metric in range(3):
            ff_scores[feature_scaling_method].append(list())

    for model in range(len(models)):

        x = whole_data[model][0]
        y = whole_data[model][2]
        kf = KFold(n_splits=splits)

        for train_index, test_index in kf.split(x):
            train_index_first = train_index[0]
            train_index_last = train_index[-1]
            test_index_first = test_index[0]
            test_index_last = test_index[-1]

            x_train, x_test = x.loc[train_index_first:train_index_last], x.loc[test_index_first:test_index_last]
            y_train, y_test = y[train_index_first:train_index_last + 1], y[test_index_first:test_index_last + 1]

            models[model].fit(x_train, y_train, epochs=150, batch_size=10, verbose=0)

            f_beta, balanced_accuracy, auc = ff_setup_evaluation(models[model],
                                                                 [x_train, x_test, y_train, y_test])

            ff_scores[model][0].append(f_beta)
            ff_scores[model][1].append(balanced_accuracy)
            ff_scores[model][2].append(auc)

    return ff_scores


def graph_roc_curve(trained_models, data, scaling_method, algorithm, ff=False):
    ns_probabilities = [0 for _ in range(len(data[0][1]))]

    for i in range(3):
        if ff:
            lr_probabilities = trained_models[i].predict(data[i][1])

        else:
            lr_probabilities = trained_models[i].predict_proba(data[i][1])
            lr_probabilities = lr_probabilities[:, 1]

        ns_fpr, ns_tpr, _ = roc_curve(data[i][3], ns_probabilities)
        lr_fpr, lr_tpr, _ = roc_curve(data[i][3], lr_probabilities)

        plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
        plt.plot(lr_fpr, lr_tpr, marker='.', label=algorithm)
        plt.title(scaling_method[i])
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # show the legend
        plt.legend()
        # show the plot
        plt.show()


def evaluate_cv(untrained_models, standardised, normalized, unit_normalized, splits, ff=False):
    sd_scores = list()
    nd_scores = list()
    ut_scores = list()

    scores = [sd_scores, nd_scores, ut_scores]

    if ff:
        model_scores = implement_cv_ff(untrained_models, [standardised, normalized, unit_normalized], splits)

        for feature_scaling_method in range(3):
            for metric in range(3):
                scores[feature_scaling_method].append(mean(model_scores[feature_scaling_method][metric]))

        return scores

    else:
        model_scores = implement_cv(untrained_models, standardised, normalized, unit_normalized, splits)

        for i in range(3):
            scores[i].append(model_scores[i]['test_f_beta'].mean())
            scores[i].append(model_scores[i]['test_balanced'].mean())
            scores[i].append(model_scores[i]['test_AUC'].mean())

        return scores


def ff_setup_evaluation(model, data):
    _, tp, fp, tn, fn, auc = model.evaluate(data[1], data[3])
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    balanced_accuracy = (sensitivity + specificity) / 2

    y_pred = (model.predict(data[1]) > 0.5).astype("int32")
    f_beta = fbeta_score(data[3], y_pred, beta=0.5)

    return f_beta, balanced_accuracy, auc
