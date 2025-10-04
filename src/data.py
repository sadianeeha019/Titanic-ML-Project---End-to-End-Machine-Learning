import pandas as pd

def load_data(path: str):
    return pd.read_csv(path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=['Cabin', 'Ticket', 'Name', 'PassengerId'], errors='ignore')
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    return df