"""
Hanna Pan
CSE 163 AA
This program contains helper functions for my EDA.
"""
import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Given a file path, return a pandas data frame of the file.
    """
    df = pd.read_csv(file_path)
    return df


def check_missing(df: pd.DataFrame) -> pd.Series:
    """
    Given a pandas dataframe, check for missing values.
    """
    return df.isnull().sum()


def data_size(df: pd.DataFrame) -> None:
    """
    Given a pandas dataframe, print the size of the data frame.
    """
    print('The data set has ' + str(len(df)) + ' rows')
    print('The data set has ' + str(len(df.columns)) + ' columns')


def data_summary(df: pd.DataFrame) -> None:
    """
    Given a pandas dataframe, print the summaries of relevant variables.
    If numerical, pring 7-number summary, if categorical, find unique values.
    """
    # Target variable: Diabetes_012
    print(df['Diabetes_012'].value_counts())
    print()
    # Smoker
    print(df['Smoker'].value_counts())
    print()
    # PhysActivity
    print(df['PhysActivity'].value_counts())
    print()
    # Fruits
    print(df['Fruits'].value_counts())
    print()
    # Veggies
    print(df['Veggies'].value_counts())
    print()
    # HvyAlcoholConsump
    print(df['HvyAlcoholConsump'].value_counts())
    print()
    # Education
    print(df['Education'].value_counts())
    print()
    # Income
    print(df['Income'].value_counts())
    print()
    # AnyHealthcare
    print(df['AnyHealthcare'].value_counts())
    print()
    # Sex
    print(df['Sex'].value_counts())
    print()
    # Age
    print(df['Age'].value_counts())
    print()
    # BMI
    print(df['BMI'].describe())
    print()


def create_diabetes_binary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a pandas dataframe, return a modified dataframe for modeling.
    """
    df['Diabetes_Binary'] = df['Diabetes_012'].apply(
        lambda x: 1 if x == 2 else 0
    )
    return df