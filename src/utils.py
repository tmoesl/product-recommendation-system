"""
This module contains utility functions for preparing and displaying model evaluation metrics
in recommendation systems. The functions assist in creating formatted DataFrames from metric
dictionaries and displaying them in a user-friendly format, enhancing the interpretability
and presentation of model performance results.

The module is structured as follows:

1. IMPORT LIBRARIES:
   - Data manipulation library (Pandas)
   - Data visualization in Jupyter notebooks (IPython.display)

2. FUNCTIONS:
   - create_metrics_df: Creates a formatted DataFrame from a dictionary of metrics, ensuring consistency 
     and readability of evaluation results for different algorithms.
   - prepare_and_display_metrics: Prepares and displays model evaluation metrics in a formatted DataFrame, 
     providing a concise summary of model performance metrics with a custom title.
"""

# ---------------------------------------------------------
# 1. IMPORT LIBRARIES
# ---------------------------------------------------------


# Import libraries for data manipulation
import pandas as pd

# Import libraries for data visualization in Jupyter notebooks
from IPython.display import display, Markdown


# ---------------------------------------------------------
# 2. FUNCTIONS
# ---------------------------------------------------------


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
