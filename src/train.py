
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib, os
from src.helpers import prepare_df

def main():
    os.makedirs('models', exist_ok=True)
    df = pd.read_csv('data/raw/train.csv')
    df = prepare_df(df)
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'models/best_model.joblib')
    print('Saved models/best_model.joblib')

if __name__=='__main__':
    main()
