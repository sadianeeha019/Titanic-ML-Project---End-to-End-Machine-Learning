
import pandas as pd, numpy as np

def prepare_df(df):
    df = df.copy()
    # feature engineering
    df['Title'] = df['Name'].str.extract(',\s*([^\.]+)\.')
    df['Title'] = df['Title'].replace(['Mlle','Ms'],'Miss').replace(['Mme'],'Mrs')
    rare = df['Title'].value_counts()[df['Title'].value_counts() < 10].index
    df['Title'] = df['Title'].replace(rare, 'Other')
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Sex'] = df['Sex'].map({'male':0,'female':1})
    df = pd.get_dummies(df, columns=['Embarked','Title'], drop_first=True)
    # select features
    features = ['Pclass','Sex','Age','SibSp','Parch','Fare','FamilySize','IsAlone'] + [c for c in df.columns if c.startswith('Embarked_') or c.startswith('Title_')]
    return df[features + (['Survived'] if 'Survived' in df.columns else [])]
