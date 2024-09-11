"""
This module contains utility functions for reading and processing multiple CSV files into a single DataFrame, 
as well as preparing and displaying model evaluation metrics in recommendation systems. The functions are designed 
to assist in efficiently importing data for model training and evaluation, creating formatted DataFrames from metric 
dictionaries, and presenting them in a user-friendly format.

The module is structured as follows:

1. IMPORT LIBRARIES:
   - Utilities for file handling and regular expression operations (os, re)
   - Data manipulation library (Pandas)
   - Data visualization in Jupyter notebooks (IPython.display)

2. FUNCTIONS:
   - read_all_csv_files: Reads all CSV files in a specified folder, sorts them in natural order, 
     and concatenates them into a single DataFrame.
   - create_metrics_df: Creates a formatted DataFrame from a dictionary of metrics, ensuring consistency 
     and readability of evaluation results for different algorithms.
   - prepare_and_display_metrics: Prepares and displays model evaluation metrics in a formatted DataFrame, 
     providing a concise summary of model performance metrics with a custom title.
"""

# ---------------------------------------------------------
# 1. IMPORT LIBRARIES
# ---------------------------------------------------------


# Import utilities for file handling and regular expression operations
import os
import re

# Import libraries for data manipulation
import pandas as pd

# Import libraries for data visualization in Jupyter notebooks
from IPython.display import display, Markdown


# ---------------------------------------------------------
# 2. FUNCTIONS
# ---------------------------------------------------------


# Function to read all CVS files in a given folder in natural order into one single DataFrame
def read_all_csv_files(
    folder_path: str, header: int | list[int] | None = None
) -> pd.DataFrame:
    """
    Reads all CSV files in a given folder, sorts them in natural order,
    and concatenates them into a single DataFrame.

    Parameters:
    - folder_path (str): The path to the folder containing CSV files.
    - header (int | list[int] | None, optional): Row number(s) to use as the column names, and the start of the data.
      If None, it assumes no headers in the CSV files (default is None).

    Returns:
    - pd.DataFrame: A concatenated DataFrame containing data from all the CSV files.
    """

    # List all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    # Function to sort filenames with numbers
    def natural_sort_key(s):
        # Extract numbers from the string and return a list of integers and text parts
        return [
            int(text) if text.isdigit() else text.lower()
            for text in re.split(r"(\d+)", s)
        ]

    # Sort the files using the natural sort key
    csv_files.sort(key=natural_sort_key)

    # Read each CSV file into a DataFrame
    df_list = []
    display(Markdown(f"**Reading CSV Files**"))

    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path, header=header)
        df_list.append(df)
        print(f"Imported: {file_path}")

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"\nData import successful. Total files read: {len(csv_files)}")

    return combined_df


# Function to prepare and display metrics
def create_metrics_df(metrics: dict, algo_names: list) -> pd.DataFrame:
    """
    Creates a DataFrame from a dictionary of metrics.

    Parameters:
    - metrics (dict): Dictionary containing algorithm names and associated metrics
    - algo_names (list): List of algorithm names from the Surprise library

    Returns:
    - DataFrame: A formatted DataFrame with the combined metrics.
    """
    # Convert dictionary to DataFrame
    metrics_df = pd.DataFrame(metrics).T

    # Check if the number of rows matches the number of algorithm names
    if len(metrics_df) != len(algo_names):
        raise ValueError(
            "The number of algorithm names does not match the number of rows in the metrics dictionary."
        )

    # Insert algorithm and separator columns
    metrics_df.insert(0, "Algo", algo_names)
    metrics_df.insert(1, "|", ["|"] * len(metrics_df))

    return metrics_df


# Function to prepare and display metrics
def prepare_and_display_metrics(metrics: dict, algo_names: list, title: str) -> None:
    """
    Prepare and display model evaluation metrics in a formatted DataFrame.

    Parameters:
    - metrics (dict): Dictionary containing algorithm names and associated metrics
    - algo_names (list): List of algorithm names from the Surprise library
    - title (str): The title to display above the DataFrame.

    Returns:
    - None: This function displays the title and DataFrame of metrics.
    """

    # Create DataFrame of algorithm instances of the Surprise library and metrics
    df = create_metrics_df(metrics, algo_names)

    # Display DataFrame
    display(Markdown(f"**{title}**"))
    display(df)


# ---------------------------------------------------------
# ---------------------------------------------------------
