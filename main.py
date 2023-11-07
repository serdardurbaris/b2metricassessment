from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
import optuna
from lofo import LOFOImportance
import pandas as pd
import numpy as np


def generate_data_csv():
    np.random.seed(0)
    count = 1000

    age = np.random.randint(18, 65, count)
    income = np.random.randint(10000, 100000, count)
    gender = np.random.choice(['Erkek', 'Kadin'], count)
    purchase_amount = np.random.uniform(1000, 5000, count)

    data = pd.DataFrame({
        'Age': age,
        'Gender': gender,
        'Income': income,
        'Expense': purchase_amount
    })
    data['Target'] = 100 + 2 * data['Age'] + 3 * data['Income'] - 50 * (data['Gender'] == 'Erkek') + 5 * data['Expense']
    data.to_csv('data.csv', index=False)


def lofo_score():
    data = pd.read_csv('data.csv')
    data.dropna(inplace=True)

    data = pd.get_dummies(data, columns=['Gender'])

    X = data.drop('Income', axis=1)
    y = data['Income']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        lgbm = LGBMRegressor()
        lgbm.fit(X_train, y_train)
        best_iteration = lgbm.best_iteration_
        lgbm_pred = lgbm.predict(X_val, num_iteration=best_iteration)

        ridge = Ridge()
        ridge.fit(X_train, y_train)
        ridge_pred = ridge.predict(X_val)

        lgbm_mse = mean_squared_error(y_val, lgbm_pred)
        ridge_mse = mean_squared_error(y_val, ridge_pred)
        print("LightGBM MSE:", lgbm_mse)
        print("Ridge MSE:", ridge_mse)

    def lgbm_objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        }

        lgbm = LGBMRegressor(**params)
        lgbm.fit(X_train, y_train)
        best_iteration = lgbm.best_iteration_
        y_pred = lgbm.predict(X_test, num_iteration=best_iteration)
        mse = mean_squared_error(y_test, y_pred)
        return mse

    study = optuna.create_study(direction="minimize")
    study.optimize(lgbm_objective, n_trials=100)
    best_params = study.best_params

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    data = pd.concat([X, y], axis=1)

    lofo_imp = LOFOImportance(dataset=data, model=lgbm, cv=cv, scoring="neg_mean_squared_error")
    lofo_scores = lofo_imp.get_importance()

    print("LOFO Importance Scores:", lofo_scores)


if __name__ == '__main__':
    # generate_data_csv()
    lofo_score()
