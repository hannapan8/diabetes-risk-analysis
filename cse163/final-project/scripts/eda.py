"""
Hanna Pan
CSE 163 AA
This program runs exploratory data analysis,
modeling, and result validation for diabetes risk analysis.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import spearmanr
from utils import (load_data,
                   check_missing,
                   data_size,
                   data_summary,
                   create_diabetes_binary)


def plot_lifestyle_diabetes(df: pd.DataFrame) -> None:
    """
    Given a pandas dataframe, plot lifestyle factors compared to
    diabetes prevalence.
    """
    factors = ['Smoker', 'PhysActivity', 'Fruits', 'Veggies',
               'HvyAlcoholConsump']

    percent_no = []
    percent_yes = []

    for factor in factors:
        no_group = df[df[factor] == 0]
        yes_group = df[df[factor] == 1]

        percentage_no = (no_group['Diabetes_012'] == 2).mean() * 100
        percentage_yes = (yes_group['Diabetes_012'] == 2).mean() * 100

        percent_no.append(percentage_no)
        percent_yes.append(percentage_yes)

    len_of_factors = np.arange(len(factors))
    width = 0.3

    plt.figure(figsize=(10, 6))

    plt.bar(len_of_factors - width/2, percent_no, width, label='0 (No)')
    plt.bar(len_of_factors + width/2, percent_yes, width, label='1 (Yes)')

    plt.title('Diabetes Risk based on Lifestyle Factor')
    plt.xlabel('Lifestyle Factor')
    plt.ylabel('Percentage Diagnosed with Diabetes per Lifestyle Factor')
    plt.xticks(len_of_factors, factors)
    plt.legend(title='Response to Lifestyle Factor')
    plt.text(0.5,
             -0.15,
             "This barplot indicates the prevalence of diabetes "
             "for individuals who answered yes or no to each lifestyle "
             "factor.",
             ha='center',
             va='center',
             transform=plt.gca().transAxes)
    plt.show()


def modeling_rq1(df: pd.DataFrame) -> None:
    """
    Given a pandas dataframe, fit and train a Logistic
    Regression and Random Forest model on the data for lifestyle factors
    and diabetes.
    """
    # log reg
    features = df[['Smoker', 'PhysActivity', 'Fruits', 'Veggies',
                   'HvyAlcoholConsump']]
    label = df['Diabetes_Binary']

    features_train, features_test, label_train, label_test = train_test_split(
        features, label, test_size=0.3, stratify=label)

    logreg_model = LogisticRegression(class_weight='balanced')
    logreg_model.fit(features_train, label_train)

    coefficients = pd.DataFrame({
        'Factor': features.columns,
        'Coefficient': logreg_model.coef_[0],
        'Odds Ratio': np.exp(logreg_model.coef_[0])
    })

    print(coefficients.sort_values(by='Odds Ratio', ascending=False))
    print()
    # random forest
    rf_model = RandomForestClassifier(class_weight='balanced')
    rf_model.fit(features_train, label_train)

    factor_importance = rf_model.feature_importances_
    importances = pd.DataFrame({
        'Factor': features.columns,
        'Importance': factor_importance
    })

    print(importances.sort_values(by='Importance', ascending=False))


def validity_rq1(df: pd.DataFrame) -> None:
    """
    Given a pandas dataframe, perform chi square tests
    to determine statistical significance.
    """
    factors = ['Smoker', 'PhysActivity', 'Fruits', 'Veggies',
               'HvyAlcoholConsump']
    for factor in factors:
        contingency_table = pd.crosstab(df[factor], df['Diabetes_Binary'])
        chi_squared, p_val, _, expected = stats.chi2_contingency(
            contingency_table)
        print(factor)
        print('Chi-squared value: ' + str(chi_squared))
        print('p-value: ' + str(p_val))
        print('expected values: ' + str(expected))
        print()


def plot_income_diabetes(df: pd.DataFrame) -> None:
    """
    Given a pandas dataframe, plot income and diabetes to determine
    relationship.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Income', y='Diabetes_Binary', errorbar=None)
    plt.title('Diabetes Prevalence Based on Income Levels')
    plt.xlabel('Income Levels')
    plt.ylabel('Proportion of Diabetic Individuals')
    plt.xticks(ticks=range(8),
               labels=['<$10k', '$10–15k', '$15–20k', '$20–25k',
                       '$25–35k', '$35–50k', '$50–75k', '>$75k'],
               rotation=30)
    plt.text(0.5,
             -0.25,
             "This plot describes the distribution of people who "
             "are diabetic in each income group.",
             ha='center',
             va='center',
             transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.show()


def plot_education_diabetes(df: pd.DataFrame) -> None:
    """
    Given a pandas dataframe, plot education and diabetes to determine
    relationship.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Education', y='Diabetes_Binary', errorbar=None)
    plt.title('Diabetes Prevalence Based on Education Levels')
    plt.xlabel('Education Levels')
    plt.ylabel('Proportion of Diabetic Individuals')
    plt.xticks(ticks=range(6),
               labels=['Never attended', 'Elementary', 'Some highschool',
                       'Highschool graduate', 'Some college',
                       'College graduate'],
               rotation=30)
    plt.text(0.5,
             -0.35,
             "This plot describes the distribution of people "
             "who are diabetic in each education level group.",
             ha='center',
             va='center',
             transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.show()


def plot_healthcare_diabetes(df: pd.DataFrame) -> None:
    """
    Given a pandas dataframe, plot healthcare access and
    diabetes to determine any relationship.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='AnyHealthcare', y='Diabetes_Binary', errorbar=None)
    plt.title('Diabetes Prevalence Based on Having Healthcare or Not')
    plt.xlabel('Healthcare Status')
    plt.ylabel('Proportion of Diabetic Individuals')
    plt.xticks(ticks=range(2),
               labels=['No healthcare', 'Has healthcare'],
               rotation=30)
    plt.text(0.5,
             -0.35,
             "This plot describes the distribution of people who are diabetic "
             "depending on if they have healthcare or not.",
             ha='center',
             va='center',
             transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.show()


def modeling_rq2(df: pd.DataFrame) -> None:
    """
    Given a pandas dataframe, fit and train a Logistic Regression
    and Random Forest model on the data for socioeconomic factors
    and diabetes.
    """
    # log reg
    features_rq2 = df[['Income', 'Education', 'AnyHealthcare']]
    label_rq2 = df['Diabetes_Binary']

    features_rq2_train, features_rq2_test, label_rq2_train, label_rq2_test = \
        train_test_split(features_rq2,
                         label_rq2,
                         test_size=0.3,
                         stratify=label_rq2)

    logreg_model_rq2 = LogisticRegression(class_weight='balanced')
    logreg_model_rq2.fit(features_rq2_train, label_rq2_train)

    coefficients_rq2 = pd.DataFrame({
        'Factor': features_rq2.columns,
        'Coefficient': logreg_model_rq2.coef_[0],
        'Odds Ratio': np.exp(logreg_model_rq2.coef_[0])
    })

    print(coefficients_rq2.sort_values(by='Odds Ratio', ascending=False))
    print()
    # random forest
    rf_model_rq2 = RandomForestClassifier(class_weight='balanced')
    rf_model_rq2.fit(features_rq2_train, label_rq2_train)

    factor_importance_rq2 = rf_model_rq2.feature_importances_
    importances_rq2 = pd.DataFrame({
        'Factor': features_rq2.columns,
        'Importance': factor_importance_rq2
    })

    print(importances_rq2.sort_values(by='Importance', ascending=False))


def validity_rq2(df: pd.DataFrame) -> None:
    """
    Given a pandas dataframe, perform chi square tests
    and spearman correlation tests to determine statistical significance.
    """
    # chi square test
    contingency_table_rq2 = pd.crosstab(df['AnyHealthcare'],
                                        df['Diabetes_Binary'])
    chi_squared_rq2, p_val_rq2, _, expected = stats.chi2_contingency(
        contingency_table_rq2)
    print('Chi-squared value: ' + str(chi_squared_rq2))
    print('p-value: ' + str(p_val_rq2))
    print(expected)
    print()

    # correlation test
    socio_factors_ord = ['Income', 'Education']
    for factor in socio_factors_ord:
        correlation_rq2, _ = spearmanr(df[factor], df['Diabetes_Binary'])
        print(factor)
        print('Correlation: ' + str(correlation_rq2))
        print()


def plot_age_diabetes(df: pd.DataFrame) -> None:
    """
    Given a pandas dataframe, plot age and
    diabetes to determine any relationship.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Age', y='Diabetes_Binary', errorbar=None)

    plt.title('Diabetes Prevalence by Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Proportion with Diabetes')

    plt.xticks(ticks=range(13),
               labels=['18–24', '25–29', '30–34', '35–39', '40–44', '45–49',
                       '50–54', '55–59', '60–64', '65–69', '70–74', '75–79',
                       '80+'],
               rotation=30)
    plt.text(0.5,
             -0.25,
             "This plot shows the proportion of people who are diabetic "
             "in each age group.",
             ha='center',
             va='center',
             transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.show()


def plot_sex_diabetes(df: pd.DataFrame) -> None:
    """
    Given a pandas dataframe, plot sex and
    diabetes to determine any relationship.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Sex', y='Diabetes_Binary', errorbar=None)
    plt.title('Diabetes Prevalence by Sex')
    plt.xlabel('Sex')
    plt.ylabel('Proportion with Diabetes')
    plt.xticks(ticks=range(2),
               labels=['Female', 'Male'],
               rotation=30)
    plt.text(0.5,
             -0.25,
             "This plot shows the proportion of people who are "
             "diabetic in each sex group.",
             ha='center',
             va='center',
             transform=plt.gca().transAxes)
    plt.show()


def plot_bmi_diabetes(df: pd.DataFrame) -> None:
    """
    Given a pandas dataframe, plot BMI and
    diabetes to determine any relationship.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Diabetes_012', y='BMI')
    plt.title('BMI Distribution by Diabetes Diagnosis')
    plt.xlabel('Diabetes Diagnosis')
    plt.ylabel('Body Mass Index (BMI)')
    plt.xticks(ticks=range(3),
               labels=['Healthy', 'Pre-Diabetic', 'Diabetic'],
               rotation=30)
    plt.text(0.5,
             -0.25,
             "This plot shows the BMI of people in each diabetes "
             "diagnosis group.",
             ha='center',
             va='center',
             transform=plt.gca().transAxes)
    plt.show()


def modeling_rq3(df: pd.DataFrame) -> None:
    """
    Given a pandas dataframe, fit and train a Logistic
    Regression and Random Forest model on the data for demographic
    factors and diabetes.
    """
    # log reg
    features_rq3 = df[['Age', 'Sex', 'BMI']]
    label_rq3 = df['Diabetes_Binary']

    features_rq3_train, features_rq3_test, label_rq3_train, label_rq3_test = \
        train_test_split(features_rq3,
                         label_rq3,
                         test_size=0.3,
                         stratify=label_rq3)

    logreg_model_rq3 = LogisticRegression(class_weight='balanced')
    logreg_model_rq3.fit(features_rq3_train, label_rq3_train)

    coefficients_rq3 = pd.DataFrame({
        'Factor': features_rq3.columns,
        'Coefficient': logreg_model_rq3.coef_[0],
        'Odds Ratio': np.exp(logreg_model_rq3.coef_[0])
    })

    print(coefficients_rq3.sort_values(by='Odds Ratio', ascending=False))
    print()
    # random forest
    rf_model_rq3 = RandomForestClassifier(class_weight='balanced')
    rf_model_rq3.fit(features_rq3_train, label_rq3_train)

    factor_importance_rq3 = rf_model_rq3.feature_importances_
    importances_rq3 = pd.DataFrame({
        'Factor': features_rq3.columns,
        'Importance': factor_importance_rq3
    })

    print(importances_rq3.sort_values(by='Importance', ascending=False))


def validity_rq3(df: pd.DataFrame) -> None:
    """
    Given a pandas dataframe, perform t-tests
    and spearman correlation tests to determine statistical significance.
    """
    # correlation
    correlation_rq3, _ = spearmanr(df['Age'], df['Diabetes_Binary'])
    print('Age')
    print('Correlation: ' + str(correlation_rq3))
    print()

    # t tests
    bmi_no_diabetes = df[df['Diabetes_Binary'] == 0]['BMI']
    bmi_diabetes = df[df['Diabetes_Binary'] == 1]['BMI']

    t_stat, p_val_bmi = stats.ttest_ind(bmi_no_diabetes,
                                        bmi_diabetes,
                                        equal_var=False)
    print('T-test statistic: ' + str(t_stat))
    print('p-value: ' + str(p_val_bmi))
    print()

    sex_no_diabetes = df[df['Diabetes_Binary'] == 0]['Sex']
    sex_diabetes = df[df['Diabetes_Binary'] == 1]['Sex']

    t_stat_sex, p_val_sex = stats.ttest_ind(sex_no_diabetes,
                                            sex_diabetes,
                                            equal_var=False)
    print('T-test statistic: ' + str(t_stat_sex))
    print('p-value: ' + str(p_val_sex))


def main():
    df = load_data('data/diabetes.csv')
    print('Missing values in the dataset: ')
    check_missing(df)

    print('Dataset size: ')
    data_size(df)

    print('Relevant variable summaries: ')
    data_summary(df)

    create_diabetes_binary(df)

    # RQ1
    print('RQ1: What lifestyle factors (physical activity, smoking, drinking, '
          'etc.) have the biggest influence in developing diabetes?')
    plot_lifestyle_diabetes(df)
    print('RQ1 Challenge Tasks: ML')
    modeling_rq1(df)
    print()
    print('RQ1 Challenge Tasks: Result Validity')
    validity_rq1(df)

    # RQ2
    print('RQ2: Do socioeconomic factors correlate to an increased risk of '
          'developing diabetes?')
    plot_income_diabetes(df)
    print()
    plot_education_diabetes(df)
    print()
    plot_healthcare_diabetes(df)
    print('RQ2 Challenge Tasks: ML')
    modeling_rq2(df)
    print('RQ2 Challenge Tasks: Result Validity')
    validity_rq2(df)

    # RQ3
    print('RQ3: Are there any demographic factors that are associated with '
          'the likelihood of being diagnosed with diabetes?')
    plot_age_diabetes(df)
    print()
    plot_sex_diabetes(df)
    print()
    plot_bmi_diabetes(df)
    print('RQ3 Challenge Tasks: ML')
    modeling_rq3(df)
    print('RQ3 Challenge Tasks: Result Validity')
    validity_rq3(df)


if __name__ == '__main__':
    main()