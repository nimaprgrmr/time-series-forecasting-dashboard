import numpy as np
import pandas as pd


def make_year(column):
    year = column.split('-')[0]
    return int(year)


def make_month(column):
    month = column.split('-')[1]
    return int(month)


def make_day(column):
    day = column.split('-')[2][0:2]
    return int(day)


def make_date(column):
    date = column.split(" ")[0]
    return date


def convert_prd(column):
    if column == 401:
        return 1397
    if column == 901:
        return 1398
    if column == 1001:
        return 1399
    if column == 1101:
        return 1400
    if column == 1201:
        return 1401
    elif column == 1301:
        return 1402


def make_prd(column):
    id_prd = column.split('-')[3]
    return int(id_prd)


def make_new_date(column):
    year, month, day = column.split('-')[0:3]
    date = '-'.join([year, month, day])
    return date


def read_data(path='./datasets/new_price.csv'):
    df = pd.read_csv(path, header=None)
    columns = ['date', 'id_prd_to_plc', 'id_br', 'amount', 'price']
    df.columns = columns

    df['total_price'] = df['price'] * df['amount']
    df['year'] = df['date'].apply(make_year)
    df['month'] = df['date'].apply(make_month)
    df['day'] = df['date'].apply(make_day)
    df['date'] = df['date'].apply(make_date)
    df['id_prd_to_plc'] = df['id_prd_to_plc'].apply(convert_prd)

    # Create date column for splitting
    df['date'] = df['date'] + '-' + df['id_prd_to_plc'].astype(str)

    # Group by day
    new_data = df.groupby(df['date'], as_index=False).sum(numeric_only=True)
    new_data['id_br'] = 51338  # id bamland branch

    # Create again columns after make new df for each day sales
    new_data['id_br'] = new_data['id_br'].astype(int)
    new_data = new_data.drop(['year', 'month', 'day'], axis=1)
    new_data['year'] = new_data['date'].apply(make_year)
    new_data['month'] = new_data['date'].apply(make_month)
    new_data['day'] = new_data['date'].apply(make_day)
    new_data['id_prd_to_plc'] = new_data['date'].apply(make_prd)
    new_data['date'] = new_data['date'].apply(make_new_date)
    new_data['series'] = np.arange(1, len(new_data)+1)

    def make_date_v2(column):
        year, month, day = column.split('-')[0:3]
        return '-'.join([year, month, day])

    new_data['date'] = new_data['date'].apply(make_date_v2)
    return new_data


def fill_days(new_data):
    new_data['date'] = pd.to_datetime(new_data['date'])
    # Initialize an empty DataFrame to store the appended rows
    appended_rows = pd.DataFrame(columns=new_data.columns)

    # Iterate through the DataFrame
    for i in range(len(new_data) - 1):
        current_date = new_data['date'].iloc[i]
        next_date = new_data['date'].iloc[i + 1]

        # Check if there is a gap between current_date and next_date
        if (next_date - current_date).days > 1 or (next_date - current_date).days < 29:
            # Append missing dates with the next day
            missing_dates = pd.date_range(current_date + pd.DateOffset(1), next_date - pd.DateOffset(1), freq='D')
            appended_rows = pd.concat([appended_rows, pd.DataFrame({'date': missing_dates})])

    # Append the missing rows to the original DataFrame
    new_data = pd.concat([new_data, appended_rows])

    # Sort the DataFrame by date again
    new_data = new_data.sort_values('date').reset_index(drop=True)

    new_data['total_price'] = new_data['total_price'].fillna(0)
    new_data['price'] = new_data['price'].fillna(0)
    new_data['amount'] = new_data['amount'].fillna(0)
    new_data['id_br'] = new_data['id_br'].fillna(51338)
    new_data['id_prd_to_plc'] = new_data['id_prd_to_plc'].fillna(0)
    new_data['date'] = new_data['date'].astype(str)
    new_data['year'] = new_data['date'].apply(make_year)
    new_data['month'] = new_data['date'].apply(make_month)
    new_data['day'] = new_data['date'].apply(make_day)
    new_data['series'] = np.arange(1, len(new_data) + 1)
    new_data = new_data.drop(['price', 'id_br'], axis=1)
    new_data['date'] = pd.to_datetime(new_data['date'])

    def make_season(month):
        if month in (1, 2, 12):
            return 4
        elif month in (3, 4, 5):
            return 1
        elif month in (6, 7, 8):
            return 2
        else:
            return 3

    new_data['season'] = new_data['month'].apply(make_season)

    def make_prd_to_plc(year):
        id_prd = {'2018': 1397, '2019': 1398, '2020': 1399, '2021': 1400, '2022': 1401, '2023': 1402, '2024': 1403,
                  '2025': 1404, '2026': 1405}
        id_prd_to_plc = id_prd[str(year)]
        return id_prd_to_plc

    for i, (y, p) in enumerate(new_data[['year', 'id_prd_to_plc']].values):
        if p == 0:
            prd = make_prd_to_plc(y)
            new_data.at[i, 'id_prd_to_plc'] = prd

    return new_data


def read_data_event(path="Datasets/events_date_bamland.csv"):
    df_event = pd.read_csv(path, header=None)
    columns = ['id_hdr', 'start_date', 'end_date', 'type', 'percent', 'id_br']
    df_event.columns = columns
    df_event.drop(['id_hdr'], axis=1)
    df_event['start_date'] = pd.to_datetime(df_event['start_date'])
    df_event['end_date'] = pd.to_datetime(df_event['end_date'])
    return df_event


def make_event(new_data, data_event):
    new_data['event'] = 0
    new_data['event_percent'] = 0

    for s, e, p in data_event[['start_date', 'end_date', 'percent']].values:
        for i, date in enumerate(new_data['date'].values):
            if (date >= s) and (date <= e):
                new_data.at[i, 'event'] = 1
                new_data.at[i, 'event_percent'] = p

    cols = ['date', 'year', 'month', 'day', 'id_prd_to_plc', 'season', 'series', 'event', 'event_percent',
            'amount', 'total_price']

    new_data = new_data[cols]
    return new_data


# data = read_data()
# new_data = fill_days(data)
# event_data = read_data_event()
# new_data = make_event(new_data, event_data)
# print(new_data.info())
