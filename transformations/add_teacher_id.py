import pandas as pd

def transform(df):
    """
    Creates a new dataframe with only the teacher_id column.
    
    Args:
        df (pd.DataFrame): Input dataframe (not used in this transformation)
        
    Returns:
        pd.DataFrame: New dataframe with only teacher_id column
    """
    # Create a new dataframe with just the teacher_id column
    result_df = pd.DataFrame({'teacher_id': [''] * len(df)})
    
    return result_df
