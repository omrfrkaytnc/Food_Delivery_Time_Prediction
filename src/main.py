import joblib
import pandas as pd
from src.evaluation.evaluate_model import hyperparameter_optimization
from src.models.train_model_after_fe import evaluate_models,voting_regressor


def main():
    test = pd.read_csv('data/raw/test.csv')
    train = pd.read_csv('data/raw/train.csv')
    df = pd.concat([train, test])
    evaluate_models(X_train, X_test, y_train, y_test)
    best_models = hyperparameter_optimization(X, y)
    voting_reg = voting_regressor(best_models, X, y)
    joblib.dump(voting_reg, "voting_clf3.pkl")
    return voting_reg






if __name__ == "__main__":
    print("İşlem başladı")
    main()
