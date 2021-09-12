import pandas as pd  
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

def read_data(path_str: str):
    """

    Parameters
    ----------
    path_str: str :
        

    Returns
    -------

    """
    if isinstance(path_str, str):
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

        data_types = []
        non_nulls = []
        nulls = []
        null_column_percent = []
        null_df_percent = []

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
            
            if null_count == 0:
                null_df_percent.append(0)
            else:
                # extract % of nulls out of total nulls in dataframe
                df_null_perc = 100 * null_count/df.isna().sum().sum()
                null_df_percent.append(df_null_perc)
                
            # capture data types
            data_types.append(df[col].dtypes) 
    
    else:
        raise Exception("{} is not a valid file path.".format(path_str))
            
    # create zipped list with column names, data_types, nulls and non nulls
    lst_data = list(zip(df_cols, data_types, non_nulls, nulls, null_column_percent, null_df_percent))
    # create dataframe of zipped list
    df_nulls = pd.DataFrame(lst_data, columns = ['Feature', 'DataType', 'CountOfNonNulls', 'CountOfNulls',\
                                                'PercentOfNullsIinColumn', 'PercentOfNullsInData'])
    return df_final, df_cols, df_nulls

def plot_nulls(df_nulls):
    """

    Parameters
    ----------
    df_nulls :
        

    Returns
    -------

    """
    if isinstance(df_nulls, pd.DataFrame):
        expected_cols = ['Feature', 'DataType', 'CountOfNonNulls', 'CountOfNulls',\
                                                'PercentOfNullsIinColumn', 'PercentOfNullsInData']
        df_nulls_cols = df_nulls.columns.tolist()
        if expected_cols == df_nulls_cols:
            if df_nulls['CountOfNulls'].sum() == 0:
                print("There are zero nulls in the DataFrame.")
                print("No plot to display!")
            else:
                null_df = df_nulls.loc[df_nulls['CountOfNulls'] > 0]
                null_df.reset_index(drop = True, inplace = True)

                fig, ax = plt.subplots()
                # the size of A4 paper lanscape
                fig.set_size_inches(12, 15)
                sns.set_context("poster", font_scale = .6, rc={"grid.linewidth": 0.6})
                sns.barplot(x = 'CountOfNulls', y = 'Feature' , data = null_df)
                plt.setp(ax.get_xticklabels(), rotation=90)
                plt.title('Count of Null Values in Dataset')
                plt.show()
        else:
            raise Exception("DataFrame columns should contain: \n".format(expected_cols))
    else:
        raise Exception("Object provided is not a valid DataFrame!")

def clean_and_transform(df):
    """

    Parameters
    ----------
    df :
        

    Returns
    -------

    """
    if isinstance(df, pd.DataFrame):
        # loop through columns and fillna with mean values
        # strip leading and trailing spaces in text data
        cols = df.columns.tolist()
        for col in cols:
            if df[col].isna().sum() == 0:
                pass
            elif df[col].isna().sum() > 0:
                if df[col].dtypes == 'int64' or df[col].dtypes == 'int32' or df[col].dtypes == 'float64':
                    df[col] = df[col].fillna(df[col].mean())
                elif df[col].dtypes == 'object':
                    df[col] = df[col].fillna(df[col].mode())
                    df[col] = df[col].apply(lambda x: x.strip())
    else:
        raise Exception("Object provided is not a valid DataFrame!")

    return df

def group_melt(df, *args):
    # aggregate data by year and region
    df_grp = df.groupby(list(args), as_index = False)[['wine', 'beer', 'vodka', 'champagne', 'brandy']].mean()
     # melt data frame - wide to long
    df_melt = pd.melt(df_grp, id_vars = args, value_vars = ['wine', 'beer', 'vodka', 'champagne', 'brandy'],\
                         var_name = 'beverages', value_name = 'Sales per Capita')
    df_melt['year'] = df_melt['year'].astype('int64')

    return df_grp, df_melt






def plot_timeseries(df_melt, *args):
    """

    Parameters
    ----------
    df_melt :
        
    *args :
        

    Returns
    -------

    """
    # plot time series for all regions and all beverages
    if 'all regions' in args and 'all beverages' in args:
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 10)
        sns.set_context('poster', font_scale = 0.5, rc = {'grid.linewidth': 0.5})
        sns.lineplot(data = df_melt, x = 'year', y = 'Sales per Capita', hue = 'beverages',\
                     style = 'beverages', markers = True)
        plt.title("Time Series of Mean Sales per Capita for all beverages")
        plt.savefig("images/{}_{}_regions.png".format('allregions', 'allbeve'))
        plt.show()
    
    else:
        for region in df_melt['region'].unique().tolist():
            # plot time series for a region and all beverages
            if region in args and 'all beverages' in args:
                df_plot = df_melt.loc[df_melt['region'] == region]    
                fig, ax = plt.subplots()
                fig.set_size_inches(15, 10)
                sns.set_context('poster', font_scale = 0.5, rc = {'grid.linewidth': 0.5})
                sns.lineplot(data = df_plot, x = 'year', y = 'Sales per Capita', hue = 'beverages',\
                            style = 'beverages', markers = True)
                plt.title("Time Series of Mean Sales per Capita of all Beverages in {}".format(region))
                plt.savefig("images/{}_vs_allbevs.png".format(region.strip()))
                plt.show()
            # plot time series for a region and a beverage
            for beverage in df_melt['beverages'].unique().tolist():
                if beverage in args:
                    df_plot = df_melt.loc[(df_melt['region'] == region) & (df_melt['beverages'] == beverage)]    
                    fig, ax = plt.subplots()
                    fig.set_size_inches(15, 10)
                    sns.set_context('poster', font_scale = 0.5, rc = {'grid.linewidth': 0.5})
                    sns.lineplot(data = df_plot, x = 'year', y = 'Sales per Capita', hue = 'beverages',\
                                style = 'beverages', markers = False)
                    plt.title("Time Series of Mean Sales per Capita of {} in {}".format(beverage, region))
                    plt.savefig("images/{}_{}.png".format(region.strip(), beverage.strip()))
                    plt.show()
                # plot time series for all region and a beverage
                elif 'all regions' in args:
                    df_plot = df_melt.loc[df_melt['beverages'] == beverage]    
                    fig, ax = plt.subplots()
                    fig.set_size_inches(15, 10)
                    sns.set_context('poster', font_scale = 0.5, rc = {'grid.linewidth': 0.5})
                    sns.lineplot(data = df_plot, x = 'year', y = 'Sales per Capita', hue = 'beverages',\
                                style = 'beverages', markers = False)
                    plt.title("Time Series of Mean Sales per Capita of {} in all regions".format(beverage))
                    plt.savefig("images/{}_vs_allregions.png".format(beverage.strip()))
                    plt.show()


def cat_plots(df_melt, *args, **kwargs):
    """

    Parameters
    ----------
    df_melt :
        
    *args :
        
    **kwargs :
        

    Returns
    -------

    """
    n = kwargs.get('n', None)
    if 'boxplot' in args:
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 10)
        sns.set_context('poster', font_scale = 0.5, rc = {'grid.linewidth': 0.5})
        sns.boxplot(x = 'Sales per Capita', y = 'beverages', data = df_melt)
        sns.despine(offset = 20, trim = True)
        plt.title('Distribution of Mean Sales per Capita by Beverage')
        plt.savefig("images/Distribution Plot of Sales by Beverage.png")
        plt.show()
    for beverage in df_melt['beverages'].unique().tolist():
        if n and beverage in args:
            sort_df = df_melt.sort_values(by = ['Sales per Capita'], ascending = False)
            top_n = sort_df.iloc[:n, :]
            fig, ax = plt.subplots()
            fig.set_size_inches(15, 10)
            sns.set_context('poster', font_scale = 0.5, rc = {'grid.linewidth': 0.5})
            sns.barplot(x = 'Sales per Capita', y = 'region', hue = 'beverages', data = top_n)
            sns.despine(offset = 10, trim = True)
            plt.title('Top {} Regions by Mean Sales per Capita'.format(n))
            plt.savefig("images/top_{}_regions.png".format(n))
            plt.show()

def corr_heatmap(df):
    """

    Parameters
    ----------
    df :
        

    Returns
    -------

    """
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 8)
    mask = np.triu(np.ones_like(df.corr(), dtype = bool))
    heatmap = sns.heatmap(df.corr(), mask = mask, vmin = -1, vmax = 1, annot  =True, cmap = 'GnBu')
    heatmap.set_title("Correlation Heatmap of Beverage Sales", fontdict = {'fontsize': 16}, pad = 15)
    plt.setp(ax.get_xticklabels(), rotation = 90)
    plt.setp(ax.get_yticklabels(), rotation = 0)
    plt.savefig("images/dfcorr.png")
    plt.show()


def preprocess_data(df):
    """

    Parameters
    ----------
    df :
        

    Returns
    -------

    """
    # instantiate MinMaxScaler
    scaler = MinMaxScaler(feature_range = (0, 1))
        
    # aggregate data by region
    df_grp = df.groupby('region', as_index = False)[['wine', 'beer', 'vodka', 'champagne', 'brandy']].mean()

    # create dataframe of transformed features
    df_grp[['wine', 'beer', 'vodka', 'champagne', 'brandy']] = scaler.fit_transform(df_grp[['wine', 'beer', 'vodka', 'champagne', 'brandy']])
    
    # extract numerical data
    df_nums = df_grp[['wine', 'beer', 'vodka', 'champagne', 'brandy']]

    # extract numerical columns names
    df_num_cols = df_nums.columns.tolist()
    
    # extract all columns
    cols_of_df = df.columns.tolist()

    # compute the cosine similarity
    cos_sim = cosine_similarity(df_nums, df_nums)
    
    return df_grp, cols_of_df, df_nums, df_num_cols, cos_sim

def recommend_regions(df_grp, region, n, cos_sim):
    """

    Parameters
    ----------
    df_grp :
        
    region :
        
    n :
        
    cos_sim :
        

    Returns
    -------

    """

    indices = pd.Series(df_grp.index, index = df_grp['region'])
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
    top_n = df_grp['region'].iloc[region_indices]
    
    return top_n



