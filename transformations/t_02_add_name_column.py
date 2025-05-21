import pandas as pd

def transform(df, input_df):
    """
    Adds a name column to the dataframe by combining first_name and last_name from input data.
    
    Args:
        df (pd.DataFrame): The current transformed dataframe
        input_df (pd.DataFrame): The original input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with name column added
    """
    # Create a copy of the current dataframe
    result_df = df.copy()
    
    # Get first name and last name from input data
    first_names = input_df['first_name']
    last_names = input_df['last_name']
    
    # Combine first and last name, handling any missing values
    full_names = first_names.fillna('') + ' ' + last_names.fillna('')
    full_names = full_names.str.strip()
    
    # Add the name column
    result_df['name'] = full_names
    
    return result_df
