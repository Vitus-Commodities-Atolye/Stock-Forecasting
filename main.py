from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor, VotingRegressor
from finta import TA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.tree import DecisionTreeRegressor


def set_data(ticker, end_date, period):

    """
    The code defines a function set_data that takes three arguments: ticker, end_date, and period.
    It downloads historical financial data using Yahoo Finance API for the specified ticker within the time range between end_date and period days before end_date.
    It then uses the finta library to calculate technical indicators and moving averages from the historical data.
    The function returns a cleaned and processed Pandas DataFrame containing the financial data and calculated technical indicators as features.
    :param ticker:
    :param end_date:
    :param period:
    :return:
    """
    start_date = end_date - timedelta(days=period)

    data = yf.download(ticker, start=start_date, end=end_date)

    # Save the data as a CSV file
    # data.to_csv(f"{ticker}_{period}.csv", index=True)

    # List of symbols for technical indicators
    INDICATORS = ['RSI', 'MACD', 'STOCH', 'ADL', 'ATR', 'MOM', 'MFI', 'ROC', 'OBV', 'CCI', 'EMV', 'VORTEX']

    """
    Next we pull the historical data using yfinance
    Rename the column names because finta uses the lowercase names
    """
    data.rename(columns={"Close": 'close', "High": 'high', "Low": 'low', 'Volume': 'volume', 'Open': 'open'},
                inplace=True)

    def _get_indicator_data(data):
        """
        Function that uses the finta API to calculate technical indicators used as the features
        :return:
        """

        for indicator in INDICATORS:
            ind_data = eval('TA.' + indicator + '(data)')
            if not isinstance(ind_data, pd.DataFrame):
                ind_data = ind_data.to_frame()
            data = data.merge(ind_data, left_index=True, right_index=True)
        data.rename(columns={"14 period EMV.": '14 period EMV'}, inplace=True)

        # Also calculate moving averages for features
        data['ema50'] = data['close'] / data['close'].ewm(50).mean()
        data['ema21'] = data['close'] / data['close'].ewm(21).mean()
        data['ema15'] = data['close'] / data['close'].ewm(14).mean()
        data['ema5'] = data['close'] / data['close'].ewm(5).mean()

        # Instead of using the actual volume value (which changes over time), we normalize it with a moving volume average
        data['normVol'] = data['volume'] / data['volume'].ewm(5).mean()

        # Remove columns that won't be used as features
        del (data['open'])
        del (data['high'])
        del (data['low'])
        del (data['volume'])
        del (data['Adj Close'])

        return data

    data = _get_indicator_data(data)

    df = pd.DataFrame(data)

    return df


def _train_gbr(X_train, y_train, X_test, y_test):
    """
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:
    This is a function named _train_gbr that trains a Gradient Boosting Regressor model using GridSearchCV to find the best hyperparameters.
    """
    # create the model
    gb_model = GradientBoostingRegressor()

    # define the hyperparameters to search over
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.1, 0.01],
        'max_depth': [3, 5, 7]
    }

    # perform grid search to find the best hyperparameters
    gb_grid = GridSearchCV(gb_model, param_grid=param_grid, cv=5)
    gb_grid.fit(X_train, y_train)

    # get the best model
    gbr_best = gb_grid.best_estimator_
    return gbr_best


def _train_xgboost(X_train, y_train, X_test, y_test):
    """
    n XGBoost regressor model using grid search to find the best hyperparameters.
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:
    """
    # Define model
    xgb_model = xgb.XGBRegressor()

    # Define hyperparameters to tune
    param_grid = {
        'n_estimators': [100, 500, 1000],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.5],
    }

    # Perform grid search
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    # Save best model
    xgb_best = grid_search.best_estimator_
    # Make predictions on the testing data
    y_pred = xgb_best.predict(X_test)
    return xgb_best


def train_decisiontree_regression(X_train, y_train, X_test, y_test):
    """
   This function trains a decision tree regressor model using the provided training data (X_train and y_train).
   It then performs a grid search to find the best hyperparameters for the model (max_depth, min_samples_split, and min_samples_leaf).
   The best model is saved, and predictions are made on the testing data (X_tes
    :return:
    """

    # Define parameter grid for GridSearchCV
    param_grid = {
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Create Decision Tree Regressor
    dt = DecisionTreeRegressor()

    # Use GridSearchCV to find best hyperparameters
    grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Save best model
    dt_best = grid_search.best_estimator_

    # Make predictions on the testing data
    y_pred = dt_best.predict(X_test)

    return dt_best


def _train_random_forest(X_train, y_train, X_test, y_test):
    """
    This function trains a random forest regression model using the RandomForestRegressor class from scikit-learn library.
    It also performs hyperparameter tuning using GridSearchCV to find the best set of hyperparameters among n_estimators, max_depth, min_samples_split, and min_samples_leaf
    :return:
    """

    # Define parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Create Random Forest Regressor
    rf = RandomForestRegressor()

    # Use GridSearchCV to find best hyperparameters
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Save best model
    rf_best = grid_search.best_estimator_
    # Make predictions on the testing data
    y_pred = rf_best.predict(X_test)

    return rf_best


def _ensemble_model(rf_model, xgb_model, gbr_model, dt_model, X_train, y_train, X_test, y_test):
    """
    This function creates an ensemble model by combining four different regression models:
    random forest (rf_model), XGBoost (xgb_model), Gradient Boosting Regression (gbr_model), and Decision Tree Regression (dt_model).
    It uses the VotingRegressor class from the scikit-learn library to combine the models.
    :param rf_model:
    :param xgb_model:
    :param gbr_model:
    :param dt_model:
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:
    """
    # Create a dictionary of our models
    estimators = [('rf', rf_model), ('xgb', xgb_model), ('gbr', gbr_model), ('dt', dt_model)]

    # Create our voting classifier, inputting our models
    ensemble = VotingRegressor(estimators)

    # fit model to training data
    ensemble.fit(X_train, y_train)

    # test our model on the test data
    print(ensemble.score(X_test, y_test))

    prediction = ensemble.predict(X_test)

    return ensemble


def calculate_min_error_model(X_train, y_train, X_test, y_test):
    """
    This function takes in training and testing data and trains several regression models,
    including Random Forest, XGBoost, Gradient Boosting, Decision Tree, and an ensemble of all models.
    It then makes predictions on the testing data using each model and returns a dictionary of the predictions for each model.
    The goal is to determine the model with the minimum error, although the function does not actually calculate or return the minimum error itself.
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:
    """

    rf_model = _train_random_forest(X_train, y_train, X_test, y_test)
    xgb_model = _train_xgboost(X_train, y_train, X_test, y_test)
    gbr_model = _train_gbr(X_train, y_train, X_test, y_test)
    dt_model = train_decisiontree_regression(X_train, y_train, X_test, y_test)
    ensemble_model = _ensemble_model(rf_model, xgb_model, gbr_model, dt_model, X_train, y_train, X_test, y_test)

    rf_prediction = rf_model.predict(X_test)
    xgb_prediction = xgb_model.predict(X_test)
    gbr_prediction = gbr_model.predict(X_test)
    dt_prediction = dt_model.predict(X_test)
    ensemble_prediction = ensemble_model.predict(X_test)

    models = dict(random_forest=rf_prediction, xgb=xgb_prediction, gbr=gbr_prediction, dt=dt_prediction,
                  ensemble=ensemble_prediction)
    return models


def determine_best_model(predict_array, y_test, days, period):
    """
    The predict_array parameter is a dictionary containing predicted values for different models,
    where the keys represent the model names and the values represent the predicted values.
    The y_test parameter is a numpy array of actual values.
    The days parameter is an integer indicating the number of days for which to calculate the root mean squared error (RMSE) for each model.
    Finally, the period parameter is an integer indicating the period (in days) for which the models were trained.
    :param predict_array:
    :param y_test:
    :param days:
    :param period:
    :return:
    """
    best_model_info_dict = dict(pred_model=None, model_error=None, model_name=None, period=period)

    # Calculate the RMSE
    rf_rmse = mean_squared_error(y_test[:days], predict_array['random_forest'][:days], squared=False)
    xgb_rmse = mean_squared_error(y_test[:days], predict_array['xgb'][:days], squared=False)
    gbr_rmse = mean_squared_error(y_test[:days], predict_array['gbr'][:days], squared=False)
    dt_rmse = mean_squared_error(y_test[:days], predict_array['dt'][:days], squared=False)
    ensemb_rmse = mean_squared_error(y_test[:days], predict_array['ensemble'][:days], squared=False)

    models = ['Random Forest', 'XGBoost', 'GBR', 'Decision Tree', 'Ensemble']
    rmse_values = [rf_rmse, xgb_rmse, gbr_rmse, dt_rmse, ensemb_rmse]

    plt.plot(models, rmse_values, marker='o')
    plt.xlabel('Models')
    plt.ylabel('RMSE')
    plt.title(f"RMSE values for different models for {days} days within period of the {period}")
    plt.show()

    print(f"period = {period}, days = {days}")
    print("Root Mean Squared Error (rf_rmse): {:.2f}".format(rf_rmse))
    print("Root Mean Squared Error (xgb_rmse): {:.2f}".format(xgb_rmse))
    print("Root Mean Squared Error (gbr_rmse): {:.2f}".format(gbr_rmse))
    print("Root Mean Squared Error (dt_rmse): {:.2f}".format(dt_rmse))
    print("Root Mean Squared Error (ensemb_rmse): {:.2f}".format(ensemb_rmse))

    # Determine which model has the smallest RMSE
    if rf_rmse == min(rf_rmse, xgb_rmse, gbr_rmse, dt_rmse, ensemb_rmse):
        best_model_info_dict['pred_model'] = predict_array['random_forest'][:days]
        best_model_info_dict['model_error'] = rf_rmse
        best_model_info_dict['model_name'] = "Random Forest"
    elif xgb_rmse == min(rf_rmse, xgb_rmse, gbr_rmse, dt_rmse, ensemb_rmse):
        best_model_info_dict['pred_model'] = predict_array['xgb'][:days]
        best_model_info_dict['model_error'] = xgb_rmse
        best_model_info_dict['model_name'] = "XGB"
    elif gbr_rmse == min(rf_rmse, xgb_rmse, gbr_rmse, dt_rmse, ensemb_rmse):
        best_model_info_dict['pred_model'] = predict_array['gbr'][:days]
        best_model_info_dict['model_error'] = gbr_rmse
        best_model_info_dict['model_name'] = "gbr"
    elif dt_rmse == min(rf_rmse, xgb_rmse, gbr_rmse, dt_rmse, ensemb_rmse):
        best_model_info_dict['pred_model'] = predict_array['dt'][:days]
        best_model_info_dict['model_error'] = dt_rmse
        best_model_info_dict['model_name'] = "dt"
    else:
        best_model_info_dict['pred_model'] = predict_array['ensemble'][:days]
        best_model_info_dict['model_error'] = ensemb_rmse
        best_model_info_dict['model_name'] = "Ensemble"


    return best_model_info_dict


def main():
    """
     predict stock prices for a set of tickers (AAPL, MSFT, and AMZN) over a range of time periods (120, 180, 270, 365, and 720 days).
    :return:
    """
    df_all_predicts = pd.DataFrame()
    df_all_companies = pd.DataFrame(columns=['Stock'])

    # Define the ticker symbols and period of interest
    tickers = ["AAPL", "MSFT", "AMZN"]
    periods = [120, 180, 270, 365, 720]

    # take input from user as end date
    end_date = input("Enter end date (YYYY-MM-DD): ")
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    # Set actual_value_dates
    end_date = end_date + timedelta(days=1)
    actual_end_date = end_date + timedelta(days=30)

    for ticker in tickers:
        actual_one_month_values = yf.download(ticker, start=end_date, end=actual_end_date, interval='1d')
        min_7day_error_val = 100
        min_14day_error_val = 100
        min_one_month_error_val = 100

        for period in periods:
            dataframe = set_data(ticker, end_date, period)  # calculate start date as 1 year before end date

            train_data, test_data = train_test_split(dataframe, test_size=0.3, shuffle=False)

            train_data = train_data.fillna(train_data.mean())
            test_data = test_data.fillna(test_data.mean())

            # Prepare the training and test data
            X_train = train_data.drop(['close'], axis=1)
            y_train = train_data['close']

            y_test = test_data['close']
            X_test = test_data.drop(['close'], axis=1)
            predict_models = calculate_min_error_model(X_train, y_train, X_test, y_test)

            for_7_days = determine_best_model(predict_models, y_test, 7, period)
            if for_7_days['model_error'] < min_7day_error_val:
                best_for_7_days = for_7_days
                min_7day_error_val = for_7_days['model_error']

            for_14_days = determine_best_model(predict_models, y_test, 14, period)
            if for_14_days['model_error'] < min_14day_error_val:
                best_for_14_days = for_14_days
                min_14day_error_val = for_14_days['model_error']

            for_one_month_days = determine_best_model(predict_models, y_test, len(actual_one_month_values.index), period)
            if for_one_month_days['model_error'] < min_one_month_error_val:
                best_one_month_days = for_one_month_days
                min_one_month_error_val = for_one_month_days['model_error']

        print("Ticker name: ", ticker, "\n",
              "Best Model for 7 days: ", best_for_7_days['model_name'], "\n",
              "Period: ", best_for_7_days['period'], "\n",
              "Best Model for 14 days: ", best_for_14_days['model_name'], "\n",
              "Period: ", best_for_14_days['period'], "\n",
              "Best Model for one month days: ", best_one_month_days['model_name'], "\n",
              "Period: ", best_one_month_days['period'], "\n", )

        best_forecasts_for_month = list(best_for_7_days['pred_model']) + list(
            best_for_14_days['pred_model'][len(best_for_7_days['pred_model']):]) + list(
            best_one_month_days['pred_model'][len(best_for_14_days['pred_model']):])

        actual_pred_df = pd.DataFrame(
            {'Date': actual_one_month_values.index, 'Stock Name': ticker, 'Prediction': best_forecasts_for_month,
             'Actual': actual_one_month_values['Close'].values})

        print(actual_pred_df)
        df_all_predicts = df_all_predicts._append(actual_pred_df)
        dataframe.insert(0, 'Stock', ticker)
        print(dataframe)
        df_all_companies = df_all_companies._append(dataframe)

    # save the DataFrame as a CSV file
    df_all_predicts.to_csv("prediction_results.csv", index=True)

    # Rename the index column to "Date"
    # df_all_companies = df_all_companies.rename_axis("Date")
    # save the DataFrame as a CSV file
    df_all_companies.to_csv("historical_results.csv", index=True)


main()
