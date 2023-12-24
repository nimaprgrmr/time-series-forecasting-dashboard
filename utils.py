import pandas as pd
from datetime import date, timedelta
from data_preprocessing import read_data
from data_preprocessing import read_data, read_data_event, make_event, fill_days
import pickle
import numpy as np
from datetime import date, timedelta


data = read_data()
new_data = fill_days(data)
event_data = read_data_event()
new_data = make_event(new_data, event_data)

model_count = pickle.load(open("models/predict_amount_xgboost_bamland.pkl", 'rb'))
model_xgb = pickle.load(open("models/predict_price_xgboost_bamland.pkl", 'rb'))
print("models uploaded")


def jalali_to_gregorian(jy, jm, jd):
    jy += 1595
    days = -355668 + (365 * jy) + ((jy // 33) * 8) + (((jy % 33) + 3) // 4) + jd
    if (jm < 7):
        days += (jm - 1) * 31
    else:
        days += ((jm - 7) * 30) + 186
    gy = 400 * (days // 146097)
    days %= 146097
    if (days > 36524):
        days -= 1
        gy += 100 * (days // 36524)
        days %= 36524
        if (days >= 365):
            days += 1
    gy += 4 * (days // 1461)
    days %= 1461
    if (days > 365):
        gy += ((days - 1) // 365)
        days = (days - 1) % 365
    gd = days + 1
    if ((gy % 4 == 0 and gy % 100 != 0) or (gy % 400 == 0)):
        kab = 29
    else:
        kab = 28
    sal_a = [0, 31, kab, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    gm = 0
    while (gm < 13 and gd > sal_a[gm]):
        gd -= sal_a[gm]
        gm += 1
    return [gy, gm, gd]


def make_season(month):
    if month in (1, 2, 12):
        return 4
    elif month in (3, 4, 5):
        return 1
    elif month in (6, 7, 8):
        return 2
    else:
        return 3


def return_result(start_date, end_date, data, event=0, percent=0):
    date1_grg = jalali_to_gregorian(start_date[0], start_date[1], start_date[2])
    date2_grg = jalali_to_gregorian(end_date[0], end_date[1], end_date[2])

    start_date = pd.to_datetime("-".join([str(date1_grg[0]), str(date1_grg[1]), str(date1_grg[2])]))
    end_date = pd.to_datetime("-".join([str(date2_grg[0]), str(date2_grg[1]), str(date2_grg[2])]))

    if (start_date in (data['date'].values)) and (end_date in (data['date'].values)):

        print("dates in dataframe")

        result = data[(data['date'] >= start_date) & (data['date'] <= end_date)].drop('total_price', axis=1)

        amounts = np.round(model_count.predict(result.drop(['date', 'amount'], axis=1)))
        result['amount'] = amounts

        total_price = model_xgb.predict(result.drop('date', axis=1).values)

        result['total_price'] = total_price

        return result

    elif (start_date in (data['date'].values)) and (end_date not in (data['date'].values)):
        print("start date in dataframe but end date not in dataframe")

        # event = input("do you want this period of time be in event or not? (Y/n): ").lower()
        # event = 1 if event == 'y' else 0
        event = event
        # percent = 0
        if event == 1:
            # percent = int(input("what is the mean of your event percente? enter the number of percent: (20, 30, ...)"))
            percent = percent / 100
            percent = percent
        elif event ==0:
            percent = 0

        first_result = data[(data['date'] >= start_date)]
        predict_first_amount = model_count.predict(first_result.drop(['date', 'amount', 'total_price'], axis=1))
        first_result['amount'] = predict_first_amount
        predict_first_prices = model_xgb.predict(first_result.drop(['date', 'total_price'], axis=1))
        first_result['total_price'] = predict_first_prices

        # create data for last day to end day
        last_row_data = first_result.iloc[-1]

        last_day = last_row_data['date'] + timedelta(days=1)

        series = last_row_data['series']

        date_list = []

        while last_day <= end_date:
            series = series + 1

            id_prd = {'2018': 1397, '2019': 1398, '2020': 1399, '2021': 1400, '2022': 1401, '2023': 1402, '2024': 1403,
                      '2025': 1404, '2026': 1405}

            date = '-'.join([str(last_day.year), str(last_day.month), str(last_day.day)])

            season = make_season(last_day.month)

            id_prd_to_plc = id_prd[str(last_day.year)]

            date_list.append(
                (date, last_day.year, last_day.month, last_day.day, id_prd_to_plc, season, series, event, percent))
            last_day += timedelta(days=1)

        result_list = [t[1:] for t in date_list]
        amounts = np.round(model_count.predict(result_list))

        df = pd.DataFrame(date_list)
        cols = ['date', 'year', 'month', 'day', 'id_prd_to_plc', 'season', 'series', 'event', 'event_percent']
        df.columns = cols
        df['amount'] = amounts
        df['date'] = pd.to_datetime(df['date'])
        prices = model_xgb.predict(df.drop('date', axis=1).values)

        df['total_price'] = prices

        final_df = pd.concat([first_result, df])

        return final_df


    elif (start_date not in (data['date'].values)) and (end_date not in (data['date'].values)):
        print('both dayes not in dataset')

        # event = input("do you want this period of time be in event or not? (Y/n): ").lower()
        # event = 1 if event == 'y' else 0
        event = event
        # percent = 0
        if event == 1:
            # percent = int(input("what is the mean of your event percente? enter the number of percent: (20, 30, ...)"))
            percent = percent / 100
            percent = percent
        elif event == 0:
            percent = 0

        first_result = data

        # create data for last day to end day
        last_row_data = first_result.iloc[-1]

        last_day = last_row_data['date'] + timedelta(days=1)

        series = last_row_data['series']

        date_list = []

        while last_day <= end_date:
            series = series + 1

            id_prd = {'2018': 1397, '2019': 1398, '2020': 1399, '2021': 1400, '2022': 1401, '2023': 1402, '2024': 1403,
                      '2025': 1404, '2026': 1405}

            date = '-'.join([str(last_day.year), str(last_day.month), str(last_day.day)])

            season = make_season(last_day.month)

            id_prd_to_plc = id_prd[str(last_day.year)]

            date_list.append(
                (date, last_day.year, last_day.month, last_day.day, id_prd_to_plc, season, series, event, percent))

            last_day += timedelta(days=1)

        result_list = [t[1:] for t in date_list]
        amounts = np.round(model_count.predict(result_list))

        df = pd.DataFrame(date_list)
        cols = ['date', 'year', 'month', 'day', 'id_prd_to_plc', 'season', 'series', 'event', 'event_percent']
        df.columns = cols
        df['amount'] = amounts
        df['date'] = pd.to_datetime(df['date'])
        prices = model_xgb.predict(df.drop('date', axis=1).values)

        df['total_price'] = prices

        final_df = pd.concat([first_result, df])

        final_df = final_df[(final_df['date'] >= start_date) & (final_df['date'] <= end_date)]

        return final_df


# s = [1402, 11, 1]
# e = [1402, 11, 31]
#
# df = return_result(s, e, new_data, event=0, percent=0)
# print(df.head())
# pred_prices = df['total_price']
# print(pred_prices.sum())
