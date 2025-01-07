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
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import LabelEncoder
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.graph_objects as go
    return (
        KMeans,
        LabelEncoder,
        MinMaxScaler,
        StandardScaler,
        dt,
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
def _(pd):
    df = pd.read_csv('online_sales_dataset.csv')
    return (df,)


@app.cell
def _(mo):
    mo.md(
        r"""
        Clean up the data. Drop NaN Customer ID, format data types, get rid of extra text etc in data,
        round the floats to 2 places for value columns
        """
    )
    return


@app.cell
def _(df):
    df.dropna(subset=['CustomerID'], inplace=True)
    return


@app.cell
def _(df):
    df['InvoiceNo'] = df['InvoiceNo'].astype('str')
    df['CustomerID'] = df['CustomerID'].astype('str')
    return


@app.cell
def _(df):
    df['StockCode'] = df['StockCode'].str.replace(r'^SKU_', '', regex=True)
    return


@app.cell
def _(df):
    df['CustomerID'] = df['CustomerID'].str.slice(0, -2)
    return


@app.cell
def _(df):
    df['UnitPrice'] = df['UnitPrice'].round(2)
    df['Discount'] = df['Discount'].round(2)
    df['ShippingCost'] = df['ShippingCost'].round(2)
    return


@app.cell
def _(df):
    df.head(10)
    return


@app.cell
def _():
    selected_columns = ['CustomerID', 'InvoiceNo', 'StockCode', 'Quantity',
                       'InvoiceDate', 'UnitPrice', 'ShippingCost',
                       'Discount', 'ReturnStatus']
    return (selected_columns,)


@app.cell
def _(df, selected_columns):
    df_working = df[selected_columns]
    return (df_working,)


@app.cell
def _(df_working):
    df_working.columns
    return


@app.cell
def _(df_working):
    df_working.rename(columns= {
        'CustomerID': 'customer',
        'InvoiceNo': 'invoice',
        'StockCode': 'sku',
        'Quantity': 'qty',
        'InvoiceDate': 'date',
        'UnitPrice': 'price',
        'ShippingCost': 'shipping',
        'Discount': 'disc',
        'ReturnStatus': 'status'
    }, inplace=True)
    return


@app.cell
def _(df_working):
    df_working.dtypes
    return


@app.cell
def _(df_working):
    df_working['netprice'] = df_working['price'] * df_working['qty']
    df_working['discprice'] = df_working['netprice'] - df_working['disc']
    df_working['totalprice'] = df_working['discprice'] + df_working['shipping']
    return


@app.cell
def _():
    selected_columns1 = ['customer', 'sku', 'totalprice', 'date', 'invoice', 'status']
    return (selected_columns1,)


@app.cell
def _(df_working, selected_columns1):
    df_status = df_working[selected_columns1]
    return (df_status,)


@app.cell
def _(df_status):
    df_status.head(2)
    return


@app.cell
def _(df_status):
    df_nr = df_status[df_status['status'] == 'Not Returned']
    df_nr.reset_index(drop=True, inplace=True)
    return (df_nr,)


@app.cell
def _(df, df_nr):
    # data left over after cleaning = 90.21%

    len(df_nr) / len(df)
    return


@app.cell
def _(df_nr):
    df_agg = df_nr.groupby(by='customer', as_index=False) \
            .agg(
                value = ('totalprice', 'sum'),
                freq = ('totalprice', 'nunique'),
                last = ('date', 'max')                
            )
    return (df_agg,)


@app.cell
def _(df_agg, pd):
    # set dtype for 'last' to datetime

    df_agg['last'] = pd.to_datetime(df_agg['last'])
    max_invoice_date = df_agg['last'].max()
    df_agg['recency'] = (max_invoice_date - df_agg['last']).dt.days
    return (max_invoice_date,)


@app.cell
def _(df_agg):
    df_agg['value'] = df_agg['value'].round(2)
    return


@app.cell
def _(df_agg):
    # trim the time component off the date/time stamp

    df_agg['lastdate'] = df_agg['last'].dt.date
    return


@app.cell
def _(df_agg):
    df_agg.head()
    return


@app.cell
def _(mo):
    mo.md("""Boxplot the features""")
    return


@app.cell
def _(df_agg, plt, sns):
    fig, ax = plt.subplots(ncols=3, figsize=(15, 5))

    sns.boxplot(data=df_agg['value'], color='skyblue', ax=ax[0])
    sns.boxplot(data=df_agg['freq'], color='orange', ax=ax[1])
    sns.boxplot(data=df_agg['recency'], color='salmon', ax=ax[2])
    return ax, fig


@app.cell
def _(mo):
    mo.md(
        """
        The value and frequency boxplots shows the influence of high outliers.
        The recency boxplot is useful.

        Produce a scatterplot of the features to check the scaling
        """
    )
    return


@app.cell
def _(df_agg, plt):
    from mpl_toolkits.mplot3d import Axes3D

    fig1 = plt.figure(figsize=(8, 8))
    ax1 = fig1.add_subplot(projection='3d')

    scatter = ax1.scatter(df_agg['value'], df_agg['freq'], df_agg['recency'])

    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    ax1.set_zlabel('Recency')
    ax1.set_title(' 3D Scatterplot of Aggregated, Non-Scaled Data')
    return Axes3D, ax1, fig1, scatter


@app.cell
def _(mo):
    mo.md(r"""Scaling is an issue. Try a log transformation to see if that can adjust the scaling.""")
    return


@app.cell
def _(df_agg):
    selected_columns2 = ['value', 'freq', 'recency']
    df_agg_log = df_agg[selected_columns2].copy()
    return df_agg_log, selected_columns2


@app.cell
def _(df_agg_log, np):
    df_agg_log['value'] = np.log1p(df_agg_log['value'])
    df_agg_log['freq'] = np.log1p(df_agg_log['freq'])
    df_agg_log['recency'] = np.log1p(df_agg_log['recency'])
    return


@app.cell
def _(mo):
    mo.md(r"""Scatterplot the log transformed data""")
    return


@app.cell
def _(df_agg_log, plt):
    fig2 = plt.figure(figsize=(8, 8))
    ax2 = fig2.add_subplot(projection='3d')

    scatter2 = ax2.scatter(df_agg_log['value'], df_agg_log['freq'], df_agg_log['recency'])

    ax2.set_xlabel('Value')
    ax2.set_ylabel('Frequency')
    ax2.set_zlabel('Recency')
    ax2.set_title(' 3D Scatterplot of Aggregated, Non-Scaled Data')
    return ax2, fig2, scatter2


@app.cell
def _(mo):
    mo.md(
        r"""
        This scatterplot shows 3 distinct 'slices' along the frequency axis at <1day, 1.5 days and <2days.
        Value and recency are more evenly distributed across their axes
        Redo the boxplots on the transformed data to visualize the features specifically
        """
    )
    return


@app.cell
def _(df_agg_log, plt, sns):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.boxplot(data=df_agg_log['value'], color='skyblue')
    plt.title('Value Boxplot LogTrans')
    plt.xlabel('Value')

    plt.subplot(1, 3, 2)
    sns.boxplot(data=df_agg_log['freq'], color='orange')
    plt.title('Frequency Boxplot LogTrans')
    plt.xlabel('Frequency')

    plt.subplot(1, 3, 3)
    sns.boxplot(data=df_agg_log['recency'], color='salmon')
    plt.title('Recency Boxplot LogTrans')
    plt.xlabel('Recency')
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        The boxplots show the distribution in Frequency is still severely influenced by high-end outliers.
        Value and Recency are less influenced but still noticeable impacted low low-end outliers.
        Try scaling the data
        """
    )
    return


@app.cell
def _(StandardScaler, df_agg_log):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_agg_log[['value', 'freq', 'recency']])
    scaled_data
    return scaled_data, scaler


@app.cell
def _(df_agg_log, pd, scaled_data):
    df_agg_log_scaled = pd.DataFrame(scaled_data, index=df_agg_log.index, columns=('value', 'freq', 'recency'))
    df_agg_log_scaled.head()
    return (df_agg_log_scaled,)


@app.cell
def _(mo):
    mo.md(
        r"""
        Redo the boxplots for the scaled data.
        Using StandardScaler did not seem to make a useful difference
        Repeat scaling on df_agg_log using MinMax scaling
        """
    )
    return


@app.cell
def _(df_agg_log_scaled, plt, sns):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.boxplot(data=df_agg_log_scaled['value'], color='skyblue')
    plt.title('Value Boxplot LogTransScaled')
    plt.xlabel('Value')

    plt.subplot(1, 3, 2)
    sns.boxplot(data=df_agg_log_scaled['freq'], color='orange')
    plt.title('Frequency Boxplot LogTransScaled')
    plt.xlabel('Frequency')

    plt.subplot(1, 3, 3)
    sns.boxplot(data=df_agg_log_scaled['recency'], color='salmon')
    plt.title('Recency Boxplot LogTransScaled')
    plt.xlabel('Recency')
    return


@app.cell
def _(MinMaxScaler, df_agg_log):
    scaler1 = MinMaxScaler()
    scaled_data1 = scaler1.fit_transform(df_agg_log[['value', 'freq', 'recency']])
    scaled_data1
    return scaled_data1, scaler1


@app.cell
def _(df_agg_log, pd, scaled_data1):
    df_agg_log_mmscaled = pd.DataFrame(scaled_data1, index=df_agg_log.index, columns=('value', 'freq', 'recency'))
    df_agg_log_mmscaled
    return (df_agg_log_mmscaled,)


@app.cell
def _(mo):
    mo.md("""Boxplot the MinMax scaled data""")
    return


@app.cell
def _(df_agg_log_mmscaled, plt, sns):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.boxplot(data=df_agg_log_mmscaled['value'], color='skyblue')
    plt.title('Value Boxplot LogTransMMScaled')
    plt.xlabel('Value')

    plt.subplot(1, 3, 2)
    sns.boxplot(data=df_agg_log_mmscaled['freq'], color='orange')
    plt.title('Frequency Boxplot LogTransMMScaled')
    plt.xlabel('Frequency')

    plt.subplot(1, 3, 3)
    sns.boxplot(data=df_agg_log_mmscaled['recency'], color='salmon')
    plt.title('Recency Boxplot LogTransMMScaled')
    plt.xlabel('Recency')
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        This result is not satisfactory either. 
        We can remove outliers +/- the IQR and see if the data that is left is enough to be useful
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""Remove the outliers beyond +/- 1.5x the Interquartile Range""")
    return


@app.cell
def _(df_agg):
    V_Q1 = df_agg['value'].quantile(0.25)
    V_Q3 = df_agg['value'].quantile(0.75)
    V_IQR = V_Q3 - V_Q1
    V_LO = V_Q1 - (1.5 * V_IQR)
    V_HO = V_Q3 + (1.5 * V_IQR)
    return V_HO, V_IQR, V_LO, V_Q1, V_Q3


@app.cell
def _(df_agg):
    F_Q1 = df_agg['freq'].quantile(0.25)
    F_Q3 = df_agg['freq'].quantile(0.75)
    F_IQR = F_Q3 - F_Q1
    F_LO = F_Q1 - (1.5 * F_IQR)
    F_HO = F_Q3 + (1.5 * F_IQR)
    return F_HO, F_IQR, F_LO, F_Q1, F_Q3


@app.cell
def _(df_agg):
    R_Q1 = df_agg['recency'].quantile(0.25)
    R_Q3 = df_agg['recency'].quantile(0.75)
    R_IQR = R_Q3 - R_Q1
    R_LO = R_Q1 - (1.5 * R_IQR)
    R_HO = R_Q3 + (1.5 * R_IQR)
    return R_HO, R_IQR, R_LO, R_Q1, R_Q3


@app.cell
def _(V_HO, V_LO, df_agg):
    df_value_ols = df_agg[(df_agg['value'] > V_HO) | (df_agg['value'] < V_LO)].copy()
    df_value_ols.describe()

    # 587 outlier values in this df
    return (df_value_ols,)


@app.cell
def _(F_HO, F_LO, df_agg):
    df_freq_ols = df_agg[(df_agg['freq'] > F_HO) | (df_agg['freq'] < F_LO)].copy()
    df_freq_ols.describe()

    # 6754 outliers in this df
    return (df_freq_ols,)


@app.cell
def _(R_HO, R_LO, df_agg):
    df_recency_ols = df_agg[(df_agg['recency'] > R_HO) | (df_agg['recency'] < R_LO)].copy()
    df_recency_ols.describe()

    # There are 0 outlier values in this df, this df can be excluded below
    return (df_recency_ols,)


@app.cell
def _(df_agg, df_freq_ols, df_value_ols):
    df_non_outliers = df_agg[(~df_agg.index.isin(df_value_ols)) & (~df_agg.index.isin(df_freq_ols))]
    return (df_non_outliers,)


@app.cell
def _(df_non_outliers):
    df_non_outliers

    # there are 32608 rows left in the non-outlier df
    return


@app.cell
def _(df, df_non_outliers):
    len(df_non_outliers) / len(df)

    # 72.77 % of data is left after the outliers are removed
    return


@app.cell
def _(mo):
    mo.md("""Redo the boxplots on the data with the value and frequency outliers removed""")
    return


@app.cell
def _(df_non_outliers, plt, sns):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.boxplot(data=df_non_outliers['value'], color='skyblue')
    plt.title('Value Boxplot Outliers Removed')
    plt.xlabel('Value')

    plt.subplot(1, 3, 2)
    sns.boxplot(data=df_non_outliers['freq'], color='orange')
    plt.title('Frequency Boxplot Outliers Removed')
    plt.xlabel('Frequency')

    plt.subplot(1, 3, 3)
    sns.boxplot(data=df_non_outliers['recency'], color='salmon')
    plt.title('Recency Boxplot Outliers Removed')
    plt.xlabel('Recency')
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        The value boxplot is better but there are still a large number of high value outliers
        The frequency boxplot is still squashed at the bottom of the plot indicating that most of values are in the low frequency range with more than a few high frequency values
        Try a log transformation on this reduced data set
        """
    )
    return


@app.cell
def _(df_non_outliers):
    df_non_outliers.columns

    selected_columns3 = ['value', 'freq', 'recency']

    df_nonoutliers_log = df_non_outliers[selected_columns3].copy()
    return df_nonoutliers_log, selected_columns3


@app.cell
def _(df_nonoutliers_log, np):
    df_nonoutliers_log['value'] = np.log1p(df_nonoutliers_log['value'])
    df_nonoutliers_log['freq'] = np.log1p(df_nonoutliers_log['freq'])
    df_nonoutliers_log['recency'] = np.log1p(df_nonoutliers_log['recency'])
    return


@app.cell
def _(df_nonoutliers_log, plt):
    # 3-D Plot the Log transformed dataframe
    fig3 = plt.figure(figsize=(8, 8))
    ax3 = fig3.add_subplot(projection='3d')

    scatter3 = ax3.scatter(df_nonoutliers_log['value'], df_nonoutliers_log['freq'], df_nonoutliers_log['recency'])

    ax3.set_xlabel('Value')
    ax3.set_ylabel('Frequency')
    ax3.set_zlabel('Recency')
    ax3.set_title(' 3D Scatterplot of Aggregated, Log Transformed Data')
    return ax3, fig3, scatter3


@app.cell
def _(mo):
    mo.md(
        r"""
        Using the dataframe with the outlier data removed has not made an appreciable difference in how the data looks. This makes using clustering a less than optimal exercise.
        Use the df_nr data to determine more basic data on customer behaviour
        """
    )
    return


@app.cell
def _(df_nr):
    df_summary = df_nr.copy()
    return (df_summary,)


@app.cell
def _(df_nr):
    df_nr.dtypes
    return


@app.cell
def _(df_summary, pd):
    # Convert date column to datetime
    df_summary['date'] = pd.to_datetime(df_summary['date'])
    return


@app.cell
def _(df_summary):
    # Group by customer and sum the totalprice
    df_summary_value = df_summary.groupby('customer')['totalprice'].sum().reset_index()
    return (df_summary_value,)


@app.cell
def _(df_summary_value):
    # Sort by totalprice in descending order to get top 20 frequent buyers
    top_25_spenders = df_summary_value.sort_values(by='totalprice', ascending=False).head(25)

    print(top_25_spenders)
    return (top_25_spenders,)


@app.cell
def _(df_summary_value):
    bottom_25_spenders = df_summary_value.sort_values(by='totalprice', ascending=False).tail(25)

    print(bottom_25_spenders)
    return (bottom_25_spenders,)


@app.cell
def _(df_summary):
    # Count the number of invoices per customer
    df_summary_frequency = df_summary.groupby('customer')['invoice'].count().reset_index()

    df_summary_frequency.head(5)
    return (df_summary_frequency,)


@app.cell
def _(df_summary_frequency):
    # Rename the column for better clarity
    df_summary_frequency1 = df_summary_frequency.rename(columns={'invoice': 'invoice_count'})
    return (df_summary_frequency1,)


@app.cell
def _(df_summary_frequency1):
    df_summary_frequency1.head()
    return


@app.cell
def _(df_summary_frequency1):
    # Sort by invoice_count in descending order to get top 20 frequent buyers
    top_25_freq_customers = df_summary_frequency1.sort_values(by='invoice_count', ascending=False).head(25)

    print(top_25_freq_customers)
    return (top_25_freq_customers,)


@app.cell
def _(df_summary_frequency1, df_summary_value):
    df_top_25_freq_customers = df_summary_frequency1.sort_values(by='invoice_count', ascending=False).head(25)
    df_top_25_spenders = df_summary_value.sort_values(by='totalprice', ascending=False).head(25)
    return df_top_25_freq_customers, df_top_25_spenders


@app.cell
def _(df_top_25_freq_customers, df_top_25_spenders):
    # Find the common entries in both top 25 lists = gold star customers

    # Convert 'customer' column to string to avoid any data type mismatch issues
    df_top_25_spenders['customer'] = df_top_25_spenders['customer'].astype(str)
    df_top_25_freq_customers['customer'] = df_top_25_freq_customers['customer'].astype(str)

    # Find the common customers using intersection
    common_customers = set(df_top_25_spenders['customer']).intersection(set(df_top_25_freq_customers['customer']))

    return (common_customers,)


@app.cell
def _():
    # If we wanted to merge the two top 25's on common customers: NOTE: there are many NaNs because only 4 entries are common

    # This isn't useful for this work since there is so little overlap between the two data sets
    # It could be useful in other contexts

    #df_common_value = df_top_25_spenders[df_top_25_spenders['customer'].isin(common_customers)]
    #df_common_freq = df_top_25_freq_customers[df_top_25_freq_customers['customer'].isin(common_customers)]

    # Merge the filtered dataframes on 'customer'
    #df_combined_common = pd.merge(df_common_value, df_common_freq, on='customer')

    #print(df_combined_common)

    return


if __name__ == "__main__":
    app.run()
