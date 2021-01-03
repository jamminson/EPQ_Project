from sklearn.linear_model import LogisticRegression
# all parameters not specified are set to their defaults


def implement_logistic_regression(standardised, normalized, unit):
    # Returns a list of the scores, 0: standardised, 1: normalized, 2: unit_normalized

    standardised_logistic = LogisticRegression()
    standardised_logistic.fit(standardised[0], standardised[2])
    standardised_score = standardised_logistic.score(standardised[1], standardised[3])

    normalized_logistic = LogisticRegression()
    normalized_logistic.fit(normalized[0], normalized[2])
    normalized_score = normalized_logistic.score(normalized[1], normalized[3])

    unit_logistic = LogisticRegression()
    unit_logistic.fit(unit[0], unit[2])
    unit_score = unit_logistic.score(unit[1], unit[3])

    return [standardised_score, normalized_score, unit_score]
