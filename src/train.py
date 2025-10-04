import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from src.data import load_data, clean_data
from src.features import engineer_features
from src.model import save_model
import os

def main():
    os.makedirs('models', exist_ok=True)
    df = load_data('data/raw/train.csv')
    df = clean_data(df)
    df = engineer_features(df)

    if 'Survived' not in df.columns:
        raise ValueError('train.csv must include Survived column')

    X = df.drop('Survived', axis=1)
    y = df['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Accuracy: {acc:.2%}")

    save_model(model, 'models/model.joblib')

if __name__ == '__main__':
    main()