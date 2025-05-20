import os
import pandas as pd
from transformations.add_teacher_id import transform as add_teacher_id

def load_transformations():
    """
    Returns a list of transformation functions in the order they should be applied.
    """
    return [
        add_teacher_id,
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
    df = pd.read_csv(input_file)
    
    # Get all transformations
    transformations = load_transformations()
    
    # Apply each transformation
    for i, transform_func in enumerate(transformations, 1):
        print(f"Applying transformation {i}: {transform_func.__name__}")
        df = transform_func(df)
    
    # Save the result
    print(f"Saving output to: {output_file}")
    df.to_csv(output_file, index=False)
    print(f"Successfully processed {len(df)} records")

if __name__ == "__main__":
    input_file = "inputv2.csv"
    output_file = "output.csv"
    process_file(input_file, output_file)
