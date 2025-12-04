"""
Hanna Pan
CSE 163 AA
This program tests some functions from my EDA.
"""
from cse163_utils import assert_equals
from sklearn.model_selection import train_test_split
import pandas as pd
import scipy.stats as stats
from scipy.stats import spearmanr
import utils


def test_load_data(test_df: pd.DataFrame) -> None:
    """
    Tests the load data method.
    """
    assert_equals(12, len(test_df.columns))
    assert_equals(4, len(test_df))


def test_check_missing(test_df: pd.DataFrame) -> None:
    """
    Tests the check missing method.
    """
    missing = utils.check_missing(test_df)
    for col in test_df.columns:
        assert_equals(0, missing[col])


def test_data_size(test_df: pd.DataFrame) -> None:
    """
    Tests the data size method.
    """
    assert_equals(12, len(test_df.columns))
    assert_equals(4, len(test_df))


def test_create_diabetes_binary(test_df: pd.DataFrame) -> None:
    """
    Tests the create diabetes binary method.
    """
    test_df = utils.create_diabetes_binary(test_df)
    assert_equals(True, 'Diabetes_Binary' in test_df.columns)
    expected = [0, 1, 1, 0]
    actual = list(test_df['Diabetes_Binary'])
    assert_equals(expected, actual)


def test_modeling_rq1(test_df: pd.DataFrame) -> None:
    """
    Tests the modeling rq1 method.
    """
    test_df = utils.create_diabetes_binary(test_df)

    X = test_df[['Smoker', 'PhysActivity', 'Fruits', 'Veggies',
                 'HvyAlcoholConsump']]
    y = test_df['Diabetes_Binary']

    assert_equals((4, 5), X.shape)
    assert_equals(4, len(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y)

    assert_equals(4, len(X_train) + len(X_test))
    assert_equals(4, len(y_train) + len(y_test))


def test_modeling_rq2(test_df: pd.DataFrame) -> None:
    """
    Tests the modeling rq2 method.
    """
    test_df = utils.create_diabetes_binary(test_df)

    X = test_df[['Income', 'Education', 'AnyHealthcare']]
    y = test_df['Diabetes_Binary']

    assert_equals((4, 3), X.shape)
    assert_equals(4, len(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y)

    assert_equals(4, len(X_train) + len(X_test))
    assert_equals(4, len(y_train) + len(y_test))


def test_modeling_rq3(test_df: pd.DataFrame) -> None:
    """
    Tests the modeling rq3 method.
    """
    test_df = utils.create_diabetes_binary(test_df)

    X = test_df[['Age', 'Sex', 'BMI']]
    y = test_df['Diabetes_Binary']

    assert_equals((4, 3), X.shape)
    assert_equals(4, len(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y)

    assert_equals(4, len(X_train) + len(X_test))
    assert_equals(4, len(y_train) + len(y_test))


def test_validity_rq1(test_df: pd.DataFrame) -> None:
    """
    Tests the validity rq1 method.
    """
    test_df = utils.create_diabetes_binary(test_df)
    factors = ['Smoker', 'PhysActivity', 'Fruits', 'Veggies',
               'HvyAlcoholConsump']
    for factor in factors:
        contingency = pd.crosstab(test_df[factor], test_df['Diabetes_Binary'])
        chi2, p_val, _, _ = stats.chi2_contingency(contingency)
        assert_equals((2, 2), contingency.shape)


def test_validity_rq2(test_df: pd.DataFrame) -> None:
    """
    Tests the validity rq2 method.
    """
    test_df = utils.create_diabetes_binary(test_df)

    # chi2
    contingency = pd.crosstab(test_df['AnyHealthcare'],
                              test_df['Diabetes_Binary'])
    chi2, p_val, _, _ = stats.chi2_contingency(contingency)
    assert_equals((2, 2), contingency.shape)
    assert 0 <= p_val <= 1

    # correlation
    income_corr, _ = spearmanr(test_df['Income'],
                               test_df['Diabetes_Binary'])
    education_corr, _ = spearmanr(test_df['Education'],
                                  test_df['Diabetes_Binary'])
    assert -1 <= income_corr <= 1
    assert -1 <= education_corr <= 1


def test_validity_rq3(test_df: pd.DataFrame) -> None:
    """
    Tests the validity rq3 method.
    """
    # correlation
    age_corr, _ = spearmanr(test_df['Age'],
                            test_df['Diabetes_Binary'])
    assert -1 <= age_corr <= 1

    # t-test
    bmi_no_diabetes = test_df[test_df['Diabetes_Binary'] == 0]['BMI']
    bmi_diabetes = test_df[test_df['Diabetes_Binary'] == 1]['BMI']

    t_stat, p_val_bmi = stats.ttest_ind(bmi_no_diabetes,
                                        bmi_diabetes,
                                        equal_var=False)
    assert 0 <= p_val_bmi <= 1

    sex_no_diabetes = test_df[test_df['Diabetes_Binary'] == 0]['Sex']
    sex_diabetes = test_df[test_df['Diabetes_Binary'] == 1]['Sex']

    t_stat_sex, p_val_sex = stats.ttest_ind(sex_no_diabetes,
                                            sex_diabetes,
                                            equal_var=False)
    assert 0 <= p_val_sex <= 1


def main():
    test_df = utils.load_data('data/test_data.csv')

    test_load_data(test_df)
    test_check_missing(test_df)
    test_create_diabetes_binary(test_df)
    test_modeling_rq1(test_df)
    test_modeling_rq2(test_df)
    test_modeling_rq3(test_df)
    test_validity_rq1(test_df)
    test_validity_rq2(test_df)
    test_validity_rq3(test_df)
    print("All tests passed!")


if __name__ == '__main__':
    main()