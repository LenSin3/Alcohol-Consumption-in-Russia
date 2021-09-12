import pandas as pd    

def read_data(path_str: str):
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
        