import xgboost as xgb
import numpy as np
from data_preprocessing import read_data, read_data_event, make_event, fill_days
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


def read_data_split_to_train_test():
    data = read_data()
    new_data = fill_days(data)
    event_data = read_data_event()
    new_data = make_event(new_data, event_data)
    train = new_data.iloc[0:-104]
    test = new_data.iloc[-104:-1]

    return train, test


def train_model_count(train, test):
    X_train_count, y_train_count = train[
        ['year', 'month', 'day', 'id_prd_to_plc', 'season', 'series', 'event', 'event_percent']], train[['amount']]
    X_test_count, y_test_count = test[
        ['year', 'month', 'day', 'id_prd_to_plc', 'season', 'series', 'event', 'event_percent']], test[['amount']]

    model_count = xgb.XGBRegressor(
        objective='reg:squarederror',
        min_child_weight=20,
        subsample=0.5,
        gamma=15,
        alpha=15,
        max_depth=5,
        learning_rate=0.001,
        n_estimators=15000,
        reg_lambda=0.001,
    )

    model_count.fit(X_train_count, y_train_count)

    return model_count


def train_model_price(model_count, train, test):
    X_train, y_train = train[['year', 'month', 'day', 'id_prd_to_plc', 'season',
                              'series', 'event', 'event_percent', 'amount']], train[['total_price']]
    X_test, y_test = test[['year', 'month', 'day', 'id_prd_to_plc', 'season',
                           'series', 'event', 'event_percent']], test[['total_price']]
    X_test_count = test[
        ['year', 'month', 'day', 'id_prd_to_plc', 'season', 'series', 'event', 'event_percent']]

    preds_count = model_count.predict(X_test_count)
    preds_count = np.round(preds_count)
    X_test['amount'] = preds_count

    model_xgb = xgb.XGBRegressor(
        objective='reg:squarederror',
        max_depth=5,
        learning_rate=0.001,
        n_estimators=15000,
        min_child_weight=20,
        reg_lambda=0.001,
        gamma=15,
        alpha=15,
        subsample=0.5,
    )

    model_xgb.fit(X_train, y_train)
    return model_xgb


def save_model(model, path_model):
    import pickle

    pickle.dump(model, open(path_model, "wb"))


# train_data, test_data = read_data_split_to_train_test()
# model_amount = train_model_count(train_data, test_data)
# model_price = train_model_price(model_amount, train_data, test_data)
#
# path_amount_model = "./models/predict_amount_xgboost_bamland.pkl"
# path_price_model = "./models/predict_price_xgboost_bamland.pkl"
#
# save_model(model_amount, path_amount_model)
# save_model(model_price, path_price_model)
