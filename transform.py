import os
import pandas as pd
from transformations.add_teacher_id import transform as add_teacher_id
from transformations.add_name_column import transform as add_name_column
from transformations.infer_subject import transform as infer_subject
from transformations.add_headline_column import transform as add_headline_column
from transformations.add_teacher_bio import transform as add_teacher_bio
from transformations.add_empty_columns import transform as add_empty_columns
from transformations.add_curriculum_experience import transform as add_curriculum_experience
from transformations.calculate_teaching_experience_ai import transform as calculate_teaching_experience

def load_transformations():
    """
    Returns a list of transformation functions in the order they should be applied.
    """
    return [
        add_teacher_id,
        add_name_column,
        infer_subject,
        add_headline_column,
        add_teacher_bio,
        add_empty_columns,
        add_curriculum_experience,
        calculate_teaching_experience,
        # Add more transformation functions here as we create them
    ]

def process_file(input_file, output_file):
    """
    Processes the input file through all transformations and saves the result.
    
    Args:
        input_file (str): Path to the input CSV file
        output_file (str): Path where the output CSV file will be saved
    """
    # Read the input CSV
    print(f"Reading input file: {input_file}")
    input_df = pd.read_csv(input_file)
    
    # Initialize with empty dataframe
    df = pd.DataFrame()
    
    # Get all transformations
    transformations = load_transformations()
    
    # Apply each transformation
    for i, transform_func in enumerate(transformations, 1):
        print(f"Applying transformation {i}: {transform_func.__name__}")
        df = transform_func(df, input_df)
    
    # Save the result
    print(f"Saving output to: {output_file}")
    df.to_csv(output_file, index=False)
    print(f"Successfully processed {len(df)} records")

if __name__ == "__main__":
    input_file = "inputv2.csv"
    output_file = "output.csv"
    process_file(input_file, output_file)
