from sklearn.metrics import classification_report
from src.model import load_model
from src.data import load_data, clean_data
from src.features import engineer_features

def main():
    model = load_model('models/model.joblib')
    df = load_data('data/raw/train.csv')
    df = clean_data(df)
    df = engineer_features(df)

    X = df.drop('Survived', axis=1)
    y = df['Survived']

    preds = model.predict(X)
    print(classification_report(y, preds))

if __name__ == '__main__':
    main()