import marimo

__generated_with = "0.10.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import datetime as dt
    import warnings
    warnings.filterwarnings('ignore')
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import silhouette_score

    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as ex
    import plotly.graph_objects as go
    return (
        DBSCAN,
        KMeans,
        MinMaxScaler,
        StandardScaler,
        dt,
        ex,
        go,
        mo,
        np,
        pd,
        plt,
        silhouette_score,
        sns,
        warnings,
    )


@app.cell
def _():
    filepath = ('Online_sports_DIB.csv')
    return (filepath,)


@app.cell
def _(filepath, pd):
    df = pd.read_csv(filepath)
    return (df,)


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df):
    df.rename(columns= {'ReqTimeUTC': 'date', 'TransactionType': 'type', 'TransactionAmount': 'amount', 'Status': 'status', 'AccountIdentifier': 'customer'}, inplace=True)
    return


@app.cell
def _(df):
    df.head(2)
    return


@app.cell
def _(df):
    df['type'].unique()
    return


@app.cell
def _(df):
    df['type'] = df['type'].map({'LOYALTYCARDDEBIT': 'L2D','LOYALTYCARDCREDITCL': 'L1D', 'LOYALTYCARDCREDIT': 'L2W','LOYALTYCARDCREDITACH': 'L1W'})
    return


@app.cell
def _(df):
    customer_type1_ = df.groupby(['customer', 'type']).count().reset_index()
    return (customer_type1_,)


@app.cell
def _(customer_type1_, sns):
    # Histplot of data grouped by customer and date with new column titles
    # Since L1D and L2D track together and L2D has more data we'll use it going forward

    sns.histplot(data=customer_type1_, x='date', hue='type', cumulative=True, stat='density', element='step', fill=False)
    return


@app.cell
def _(df):
    df['customer'] = df['customer'].str.replace(r'^customer', '', regex=True)
    return


@app.cell
def _(df, pd):
    df['date'] = pd.to_datetime(df['date'])
    return


@app.cell
def _(df):
    df.dtypes
    return


@app.cell
def _(df):
    df['customer'].isna()
    return


@app.cell
def _(df):
    df['type'].unique()
    return


@app.cell
def _(df):
    df_l2d = df[(df['type'] == 'L2D')].reset_index(drop=True)
    return (df_l2d,)


@app.cell
def _(df_l2d):
    df_l2d['type'].unique()
    return


@app.cell
def _(df_l2d):
    df_l2d_app = df_l2d[(df_l2d['status'] == 'APPROVED')].reset_index(drop=True)
    return (df_l2d_app,)


@app.cell
def _(df_l2d_app):
    df_l2d_app['status'].unique()
    return


@app.cell
def _(df_l2d_app):
    df_working = df_l2d_app.copy()
    return (df_working,)


@app.cell
def _(df, df_working):
    len(df_working) / len(df)
    return


@app.cell
def _(mo):
    mo.md(r"""At the end of working towards the data that we want to examine we have 48% of the orginal data left to work with. Given there is two level of transactions and each level has two transactions, this is not unexpected.""")
    return


@app.cell
def _(df_working):
    # Breakout the day of the week and the hour of the txn from the date element

    df_working['hour'] = df_working['date'].dt.hour
    df_working['weekday'] = df_working['date'].dt.day_of_week
    return


@app.cell
def _(df_working):
    day_of_week_dict = {0: '1_Mon', 1:'2_Tues', 2: '3_Wed', 3:'4_Thur', 4:'5_Fri', 5: '6_Sat', 6: '7_Sun' }

    df_working['weekday'] = df_working['weekday'].map(day_of_week_dict)
    return (day_of_week_dict,)


@app.cell
def _(df_working):
    df_working.head()
    return


@app.cell
def _(df_working, pd, sns):
    # Heat map of the sportsbook activities

    h_d_g =df_working[['hour', 'weekday', 'type']].groupby(['hour', 'weekday']).count().reset_index()

    working_hm = pd.pivot_table(h_d_g, values='type', index='weekday', columns='hour')

    sns.heatmap(working_hm)
    return h_d_g, working_hm


@app.cell
def _():
    # The heatmap shows greatest activity between 1400 - 1900 each day
    # Monday's greatest activity is from 1400 - 2300 and each day the time range tightens through to Sunday's range
    return


@app.cell
def _(df_working):
    df_working.dtypes
    return


@app.cell
def _(df_working):
    # Find and rank the top 25 spenders in the sportsbook

    df_summary_value = df_working.groupby('customer')['amount'].sum().reset_index()

    top_25_spenders = df_summary_value.sort_values(by='amount', ascending=False).head(25)
    print(top_25_spenders)
    return df_summary_value, top_25_spenders


@app.cell
def _(df_working):
    # Find and rank the top 25 most frequent depositors to the sportsbook

    df_summary_freq = df_working.groupby('customer')['status'].count().reset_index()

    top_25_depositors = df_summary_freq.sort_values(by='status', ascending=False).head(25)
    print(top_25_depositors)
    return df_summary_freq, top_25_depositors


@app.cell
def _(df_summary_freq):
    # See if there are any common entries in the top 25 lists
    # convert variable to df's

    df_freq_top_25 = df_summary_freq.sort_values(by='status', ascending=False)
    df_freq_top_25.head()

    return (df_freq_top_25,)


@app.cell
def _(df_summary_value):
    df_value_top_25 = df_summary_value.sort_values(by='amount', ascending=False)
    df_value_top_25.head()
    return (df_value_top_25,)


@app.cell
def _(df_freq_top_25, df_value_top_25):
    # Combine the lists to find any common customers

    common_customers = set(df_value_top_25['customer']).intersection(set(df_freq_top_25['customer']))

    print(common_customers)
    return (common_customers,)


@app.cell
def _(common_customers, df_freq_top_25, df_value_top_25):
    # Filter the dataframes to only include the common customers
    df_common_value = df_value_top_25[df_value_top_25['customer'].isin(common_customers)]
    df_common_frequency = df_freq_top_25[df_freq_top_25['customer'].isin(common_customers)]
    return df_common_frequency, df_common_value


@app.cell
def _(df_common_frequency, df_common_value, pd):
    # Merge the dataframes on 'Customer' column to get the combined information
    df_combined = pd.merge(df_common_value, df_common_frequency, on='customer')
    return (df_combined,)


@app.cell
def _(df_combined):
    df_combined.head(25)
    return


if __name__ == "__main__":
    app.run()
