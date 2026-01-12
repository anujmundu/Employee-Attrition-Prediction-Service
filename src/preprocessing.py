import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERICAL_COLS = [
    "Age",
    "DistanceFromHome",
    "Education",
    "EnvironmentSatisfaction",
    "JobSatisfaction",
    "MonthlyIncome",
    "NumCompaniesWorked",
    "WorkLifeBalance",
    "YearsAtCompany",
]

CATEGORICAL_COLS = [
    "Department",
    "EducationField",
    "MaritalStatus",
]

TARGET_COL = "Attrition"


def get_preprocessor():
    """
    Returns a ColumnTransformer that applies:
    - StandardScaler to numerical features
    - OneHotEncoder to categorical features
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_COLS),
            ("cat", categorical_transformer, CATEGORICAL_COLS),
        ]
    )

    return preprocessor
