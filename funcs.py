import pandas as pd  
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame 
import seaborn as sns 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import os
import utils

expected_cols = ['year', 'region', 'wine', 'beer', 'vodka', 'champagne', 'brandy']
expected_null_cols = ['Feature', 'DataType', 'CountOfNonNulls', 'CountOfNulls',\
                                                'PercentOfNullsIinColumn', 'PercentOfNullsInData']

def read_data(path_str: str) -> pd.DataFrame:
    """Read a path string to a DataFrame.

    The string is read to a DataFrame and profiled for nulls.

    Parameters
    ----------
    path_str : str
        Path to csv file.
        
    Returns
    -------
    DataFrame    
    """

    if os.path.exists(path_str):
        # read data
        df = pd.read_csv(path_str)
        # make a copy of dataframe
        print("Making a copy of the dataframe\n")
        df_1 = df.copy()
        # drop duplicates
        df_final = df_1.drop_duplicates()
        # extract feature names
        df_cols = df_final.columns.tolist()
        
        print("Data consists of:\n")
        print("...........................\n")
        print("Rows: {}\n".format(len(df_final)))
        print("Columns: {}\n".format(len(df_cols)))
        print("...........................\n")

        # create empty lists to hold data types, null counts and null percentages
        data_types = []
        non_nulls = []
        nulls = []
        null_column_percent = []
        null_df_percent = []

        # check if columns in DataFrame are in expected columns
        if df_cols == expected_cols:
            # loop through columns and capture the variables above
            print("Extracting count and percentages of nulls and non nulls")
            for col in df_cols:
                    
                # extract null count
                null_count = df[col].isna().sum()
                nulls.append(null_count)
                    
                # extract non null count
                non_null_count = len(df) - null_count
                non_nulls.append(non_null_count)
                    
                # extract % of null in column
                col_null_perc = 100 * null_count/len(df)
                null_column_percent.append(col_null_perc)
                
                # append 0 to null percent o avoid ZeroDivisionError
                if null_count == 0:
                    null_df_percent.append(0)
                else:
                    # extract % of nulls out of total nulls in Dataframe
                    df_null_perc = 100 * null_count/df.isna().sum().sum()
                    null_df_percent.append(df_null_perc)
                    
                # capture data types
                data_types.append(df[col].dtypes)
        else:
            raise utils.UnexpectedDataFrame(df)
    
    else:
        raise utils.InvalidFilePath(path_str)
            
    # create zipped list with column names, data_types, nulls and non nulls
    lst_data = list(zip(df_cols, data_types, non_nulls, nulls, null_column_percent, null_df_percent))
    # create dataframe of zipped list
    df_nulls = pd.DataFrame(lst_data, columns = ['Feature', 'DataType', 'CountOfNonNulls', 'CountOfNulls',\
                                                'PercentOfNullsIinColumn', 'PercentOfNullsInData'])
    return df_final, df_cols, df_nulls

def plot_nulls(df_nulls):
    """Plot Count of Nulls of columns with nulls.

    The plot is done for columns with nulls in DataFrame.

    Parameters
    ----------
    df_nulls : DataFrame
        DataFrame with null data.
        
    Returns
    -------
    seaborn.barplot    
    """
    if isinstance(df_nulls, pd.DataFrame):

        # extract columns in df_nulls
        df_nulls_cols = df_nulls.columns.tolist()
        # check if columns in DataFrame are in expected columns
        if df_nulls_cols == expected_null_cols:
            # if sum of values in CountOfNulls is zero
            if df_nulls['CountOfNulls'].sum() == 0:
                print("There are zero nulls in the DataFrame.")
                print("No plot to display!")
            else:
                # filter for values of CountOfNulls greater than zero
                null_df = df_nulls.loc[df_nulls['CountOfNulls'] > 0]
                null_df.reset_index(drop = True, inplace = True)

                # create a barplot
                fig, ax = plt.subplots()
                fig.set_size_inches(12, 15)
                sns.set_context("poster", font_scale = .6, rc={"grid.linewidth": 0.6, "font.size":20, "axes.titlesize":20, "axes.labelsize":20})
                sns.barplot(x = 'CountOfNulls', y = 'Feature' , data = null_df.sort_values(by = 'CountOfNulls', ascending=False))
                plt.setp(ax.get_xticklabels(), rotation=90)
                plt.title('Count of Null Values in Dataset')
                plt.savefig('images/nullcount.png')
                plt.show()
        else:
            raise utils.UnexpectedDataFrame(df_nulls)
    else:
        raise utils.InvalidDataFrame(df_nulls)

def clean_and_transform(df):
    """Clean and transform data.

    Impute missing values for numerical and categorical data and, convert year to datatime object.

    Parameters
    ----------
    df : DataFrame
        DataFrame to clean and transform.        

    Returns
    -------
    DataFrame    
    """
    if isinstance(df, pd.DataFrame):
                
        
        # extract columns in dataframe
        cols = df.columns.tolist()
        # check if columns in DataFrame are in expected columns
        if cols == expected_cols:
            # loop through columns and fillna with mean, ffill and mode
            for col in cols:
                if df[col].isna().sum() == 0:
                    pass
                elif df[col].isna().sum() > 0:
                    if col == 'year':
                        df[col] = df[col].fillna(method = 'ffill')
                    elif df[col].dtypes == 'int64' or df[col].dtypes == 'int32' or df[col].dtypes == 'float64':
                        df[col] = df[col].fillna(df[col].mean())
                    elif df[col].dtypes == 'object':
                        df[col] = df[col].fillna(df[col].mode())
                        # strip leading and trailing spaces in text data
                        df[col] = df[col].apply(lambda x: x.strip())
        else:
            raise utils.UnexpectedDataFrame(df)
    else:
        raise utils.InvalidDataFrame(df)
    # convert year to datatime    
    df['year'] = pd.to_datetime(df['year'], format = '%Y')
    return df

def group_melt(df):
    """Group and melt a DataFrame.

    DataFrame is grouped by year and region and melted with year and region as ID variables.

    Parameters
    ----------
    df : DataFrame
        DataFrame to group and melt.        

    Returns
    -------
    DataFrame
    """

    # aggregate data by year and region
    df_grp = df.groupby(['year', 'region'], as_index = False)[['wine', 'beer', 'vodka', 'champagne', 'brandy']].mean()
    # melt data frame - wide to long
    df_melt = pd.melt(df_grp, id_vars = ['year', 'region'], value_vars = ['wine', 'beer', 'vodka', 'champagne', 'brandy'],\
                var_name = 'beverages', value_name = 'Sales per Capita')

    return df_melt



def plot_timeseries(df, **kwargs):
    """Plot  time series of sales.

    Time series plot of Mean Sales of Beverages per Region.

    Parameters
    ----------
    df : DataFrame
        
    **kwargs : dict, optional
        Keyword arguments for sales region and beverage type.        

    Returns
    -------
    seaborn.lineplot
    """
    sales_region = kwargs.get('sales_region', None)
    beverage = kwargs.get('beverage', None)
    
    if isinstance(df, pd.DataFrame):
        # check if columns in DataFrame are in expected columns
        if df.columns.tolist() == expected_cols:
            # extract melted dataframe
            df_melt = group_melt(df)
            # capture unique regions
            all_regions = df_melt['region'].unique().tolist()

            # if no keyword argument(s) is/are given
            if not kwargs:
                # plot time series for all regions and all beverages
                fig, ax = plt.subplots()
                fig.set_size_inches(15, 10)
                sns.set_context('poster', font_scale = 0.8, rc = {'grid.linewidth': 0.5, "font.size":20,"axes.titlesize":20,"axes.labelsize":20})
                sns.lineplot(data = df_melt, x = 'year', y = 'Sales per Capita', hue = 'beverages',\
                            style = 'beverages', markers = True)
                plt.title("Time Series of Mean Sales per Capita for all beverages")
                # plt.savefig("images/{}_{}_regions.png".format('allregions', 'allbeve'))
                plt.show()
            # single keyword arugumen and sales region
            elif len(kwargs) == 1 and sales_region:
                # check if sales region is in region list
                if sales_region in all_regions:
                    # filter for rows where region is sales region
                    df_plot = df_melt.loc[df_melt['region'] == sales_region]   
                    # plot time series for that regions Mean Sales of all Beverages 
                    fig, ax = plt.subplots()
                    fig.set_size_inches(15, 10)
                    sns.set_context('poster', font_scale = 0.8, rc = {'grid.linewidth': 0.5, "font.size":20,"axes.titlesize":20,"axes.labelsize":20})
                    sns.lineplot(data = df_plot, x = 'year', y = 'Sales per Capita', hue = 'beverages',\
                                style = 'beverages', markers = True)
                    plt.title("Time Series of Mean Sales per Capita of all Beverages in {}".format(sales_region))
                    # plt.savefig("images/{}_vs_allbevs.png".format(region))
                    plt.show()
                else:
                    raise utils.InvalidRegion(sales_region)
            # single keyword argument and beverage       
            elif len(kwargs) == 1 and beverage:
                # check if beverage is in expected columns
                if beverage in expected_cols:
                    # extract rows where beverages is= beverage
                    df_plot = df_melt.loc[df_melt['beverages'] == beverage]
                    # Plot time series for beverage for all Regions    
                    fig, ax = plt.subplots()
                    fig.set_size_inches(15, 10)
                    sns.set_context('poster', font_scale = 0.8, rc = {'grid.linewidth': 0.5, "font.size":20,"axes.titlesize":20,"axes.labelsize":20})
                    sns.lineplot(data = df_plot, x = 'year', y = 'Sales per Capita', hue = 'beverages',\
                                    style = 'beverages', markers = False)
                    plt.title("Time Series of Mean Sales per Capita of {} in all regions".format(beverage))
                    # plt.savefig("images/{}_vs_allregions.png".format(beverage))
                    plt.show()
                else:
                    raise utils.InvalidBeverage(beverage)
            # for region and beverage keyword arguments
            elif len(kwargs) == 2 and (beverage and sales_region):
                # check if beverage in expected columns and regions in unique regions
                if beverage in expected_cols and sales_region in all_regions:
                    # filter for beverage and region
                    df_plot = df_melt.loc[(df_melt['region'] == sales_region) & (df_melt['beverages'] == beverage)]
                    # plot time series for region and beverage    
                    fig, ax = plt.subplots()
                    fig.set_size_inches(15, 10)
                    sns.set_context('poster', font_scale = 0.8, rc = {'grid.linewidth': 0.5, "font.size":20,"axes.titlesize":20,"axes.labelsize":20})
                    sns.lineplot(data = df_plot, x = 'year', y = 'Sales per Capita', hue = 'beverages',\
                                    style = 'beverages', markers = False)
                    plt.title("Time Series of Mean Sales per Capita of {} in {}".format(beverage, sales_region))
                    plt.savefig("images/{}_{}.png".format(sales_region, beverage))
                    plt.show()
                elif beverage in expected_cols and sales_region not in all_regions:
                    raise utils.InvalidRegion(sales_region)
                elif sales_region in all_regions and beverage not in expected_cols:
                    raise utils.InvalidBeverage(beverage)
                elif beverage not in expected_cols and sales_region not in all_regions:
                    raise utils.BeverageRegionExceptions(beverage, sales_region)
        else:
            raise utils.UnexpectedDataFrame(df)        
    else:
        raise utils.InvalidDataFrame(df)

        

def box_plot(df):
    """Plot boxplot of distribution of sales.

    Boxplot is created to capture distribution of sales in the regions.

    Parameters
    ----------
    df : DataFrame        

    Returns
    -------
    seaborn.boxplot
    """
    
    if isinstance(df, pd.DataFrame):
        # extract melted data
        df_melt = group_melt(df)
        # create boxplot
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 10)
        sns.set_context('poster', font_scale = 0.8, rc = {'grid.linewidth': 0.5, "font.size":20,"axes.titlesize":20,"axes.labelsize":20})
        sns.boxplot(x = 'Sales per Capita', y = 'beverages', data = df_melt)
        sns.despine(offset = 20, trim = True)
        plt.title('Distribution of Mean Sales per Capita by Beverage')
        # plt.savefig("images/Distribution Plot of Sales by Beverage.png")
        plt.show()
    else:
        raise utils.InvalidDataFrame(df)

def bar_plot(df, n = 10, **kwargs):
    """Plot barplot of Regional Mean Sales.

    Barplot of Mean Sales of all beverages and per beverage.

    Parameters
    ----------
    df : DataFrame
        DataFrame to create barplot.
        
    n : integer
        Returns top 10 regions by Mean Sales if n is not given.
        (Default value = 10)
    **kwargs : dict, optional
        Keyword argument for beverage.        

    Returns
    -------
    seaborn.barplot
    """
    beverage = kwargs.get('beverage', None)
   
    if isinstance(df, pd.DataFrame):
        # check if columns in DataFrame are in expected columns
        if df.columns.tolist() == expected_cols:
            # group dataframe by region
            df_grp = df.groupby('region', as_index = False)[['wine', 'beer', 'vodka', 'champagne', 'brandy']].mean()
            # melt data frame - wide to long
            df_melt = pd.melt(df_grp, id_vars = 'region', value_vars = ['wine', 'beer', 'vodka', 'champagne', 'brandy'],\
                            var_name = 'beverages', value_name = 'Sales per Capita')
            # sort dataframe by Sales per Capita in descending order
            sort_df = df_melt.sort_values(by = ['Sales per Capita'], ascending = False)
            # return top n
            top_n = sort_df.iloc[:n, :]

            # if beverage is not given
            if not beverage:
                # plot barplot of top n regional sales
                fig, ax = plt.subplots()
                fig.set_size_inches(15, 10)
                sns.set_context('poster', font_scale = 0.8, rc = {'grid.linewidth': 0.5, "font.size":20,"axes.titlesize":20,"axes.labelsize":20})
                sns.barplot(x = 'Sales per Capita', y = 'region', hue = 'beverages', data = top_n)
                sns.despine(offset = 10, trim = True)
                plt.title('Top {} Regions by Mean Sales per Capita of all Beverages'.format(n))
                # plt.savefig("images/top_{}_regions.png".format(n))
                plt.show()
            
            # if beverage is given
            elif beverage:
                # check if beverage in expected columns
                if beverage in expected_cols:
                    # filter for beverage
                    sort_df_bev = df_melt.loc[df_melt['beverages'] == beverage]
                    # sort beverage in descending order
                    sort_df_bev = sort_df_bev.sort_values(by = ['Sales per Capita'], ascending = False)
                    top_n = sort_df_bev.iloc[:n, :]
                    fig, ax = plt.subplots()
                    fig.set_size_inches(15, 10)
                    sns.set_context('poster', font_scale = 0.8, rc = {'grid.linewidth': 0.5, "font.size":20,"axes.titlesize":20,"axes.labelsize":20})
                    sns.barplot(x = 'Sales per Capita', y = 'region', data = top_n)
                    sns.despine(offset = 10, trim = True)
                    plt.title('Top {} Regions by Mean Sales per Capita of {}'.format(n, beverage))
                    # plt.savefig("images/top_{}_regions.png".format(n))
                    plt.show()
                else:
                    raise utils.InvalidBeverage(beverage)
        else:
            raise utils.UnexpectedDataFrame(df)
    else:
        raise utils.InvalidDataFrame(df)

def corr_heatmap(df):
    """Plot Correlation heatmap.

    Parameters
    ----------
    df : DataFrame
        DataFrame with numerical values to make Correlation Heatmap        

    Returns
    -------
    seaborn.heatmap    
    """
    if isinstance(df, pd.DataFrame):
        if df.columns.tolist() == expected_cols:
            fig, ax = plt.subplots()
            fig.set_size_inches(15, 8)
            mask = np.triu(np.ones_like(df.corr(), dtype = bool))
            heatmap = sns.heatmap(df.corr(), mask = mask, vmin = -1, vmax = 1, annot  =True, cmap = 'GnBu')
            heatmap.set_title("Correlation Heatmap of Beverage Sales", fontdict = {'fontsize': 16}, pad = 15)
            plt.setp(ax.get_xticklabels(), rotation = 90)
            plt.setp(ax.get_yticklabels(), rotation = 0)
            # plt.savefig("images/dfcorr.png")
            plt.show()
        else:
            raise utils.UnexpectedDataFrame(df)
    else:
        raise utils.InvalidDataFrame(df)

def pivot_data(df, *beverages):
    """Pivot a DataFrame.

    DataFrame is pivoted with region as index and year as columns while beverage sales are the values.

    Parameters
    ----------
    df : DataFrame
        DataFrame to pivot
        
    *beverages : tuple
        List of beverages to use as values.        

    Returns
    -------
    DataFrame
    """
    # create a list with all beverages
    bev_list = ['wine', 'beer', 'vodka', 'champagne', 'brandy']
    # create a list object of beverage/s supplied.
    bevs = list(beverages)
    # for a single beverage
    if len(bevs) == 1:
        for bev in bevs:
            # check if beverage is in beverage list
            if bev in bev_list:
                # create a pivot with beverage values
                df_pivot = df.pivot(index='region', columns='year', values=bev).reset_index()
                df_pivot.columns.name = None
            else:
                raise utils.InvalidBeverage(bev)
    # for multiple beverages
    elif len(bevs) > 1:
        # check if all beverages supplied are in beverage list
        all_bevs = all(bev in bev_list for bev in bevs)
        if all_bevs:
            # create a pivot dataframe with beverages values
            df_pivot = df.pivot(index='region', columns='year', values=bevs).reset_index()
            df_pivot.columns.name = None
        else:
            invalid_bevs = []
            for bev in bevs:
                if bev not in bev_list:
                    invalid_bevs.append(bev)
            raise utils.InvalidBeverage(invalid_bevs)

    return df_pivot

def preprocess_data(df, *beverages):
    """Preprocess data for machine learning.

    Data is preprocessed by standardization and Cosine Similarity is calculated

    Parameters
    ----------
    df : DataFrame
        
    *beverages : tuple
        List of beverages.        

    Returns
    -------
    DataFrame, numpy.array    
    """
    # instantiate MinMaxScaler
    scaler = MinMaxScaler(feature_range = (0, 1))

    # extract pivoted data
    df_pivot_bev = pivot_data(df, *beverages)
    # slice for all year columns excluding region
    df_nums = df_pivot_bev.iloc[:, 1:]
    # scale sliced dataframe
    df_scaled = scaler.fit_transform(df_nums)

    # compute the cosine similarity
    cos_sim = cosine_similarity(df_scaled, df_scaled)
    
    return df_pivot_bev, cos_sim

def recommend_regions(df, region, *beverages, n = 10):
    """Recommend similar regions.

    Regions are recommended for a region with similarity in sales.

    Parameters
    ----------
    df : DataFrame
        DataFrame with beverage regional sales data.
        
    region : str
        Region to make recommendations for.

     *beverages : str, list
        List of beverages.

    n : integer
        Returns top 10 regions with similar sales if n is not given.
        (Default value = 10)
    

    Returns
    -------
    DataFrame.Series
    """
    if isinstance(df, pd.DataFrame):
        # check if columns in DataFrame are in expected columns
        if df.columns.tolist() == expected_cols:
            # extract pivoted data and cosine similarity scores
            df_pivot_bev, cos_sim = preprocess_data(df, *beverages)
            # check if region exists is list of regions
            if region in df['region'].unique().tolist():
                # create a series with region as index
                indices = pd.Series(df_pivot_bev.index, index = df_pivot_bev['region'])
                indices = indices.drop_duplicates()
                # get index corresponding to region
                idx = indices[region]
                # get pairwise similarity scores
                sig_scores = list(enumerate(cos_sim[idx]))
                # sort the regions
                sig_scores = sorted(sig_scores, key = lambda x: x[1], reverse = True)
                # scores of n most similar regions
                sig_scores = sig_scores[1:n+1]
                # region indices
                region_indices = [i[0] for i in sig_scores]
                # get n most similar regions
                top_n = df_pivot_bev['region'].iloc[region_indices]
            else:
                raise utils.InvalidRegion(region)
        else:
            raise utils.UnexpectedDataFrame(df)
    else:
        raise utils.InvalidDataFrame(df)
    
    return top_n



