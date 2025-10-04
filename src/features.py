import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # map sex
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    # one-hot encode Embarked
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    # keep relevant numeric columns
    cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare'] + [c for c in df.columns if c.startswith('Embarked_')]
    return df[cols + (['Survived'] if 'Survived' in df.columns else [])]