"""
utils.py

This module contains utility functions for reading and processing multiple CSV files into a single DataFrame, 
as well as preparing and displaying model evaluation metrics in recommendation systems. The functions are designed 
to assist in efficiently importing data for model training and evaluation, randomly selecting user-item interactions, creating formatted DataFrames from metric 
dictionaries, and presenting them in a user-friendly format.

The module is structured as follows:

1. IMPORT LIBRARIES:
    - Utilities for file handling and regular expression operations (os, re)
    - Data manipulation library (random, Pandas)
    - Data visualization in Jupyter notebooks (IPython.display)

2. FUNCTIONS:
    - read_all_csv_files: Reads all CSV files in a specified folder, sorts them in natural order, and concatenates them into a single DataFrame.
    - select_interactions: Randomly selects a subset of users and their interactions and non-interactions from the training set.
    - display_eval_metrics: Displays model evaluation metrics in DataFrame format, providing a concise summary of predictive and ranking metrics for both training and test sets.
    - create_metrics_df: Creates a formatted DataFrame from a dictionary of metrics, ensuring consistency and readability of evaluation results for different algorithms.
    - prepare_and_display_metrics: Prepares and displays model evaluation metrics in a formatted DataFrame, providing a concise summary of model performance metrics with a custom title.
"""

# ---------------------------------------------------------
# 1. IMPORT LIBRARIES
# ---------------------------------------------------------


# Import utilities for file handling and regular expression operations
import os
import re

# Import libraries for data manipulation
import random
import pandas as pd

# Import libraries for data visualization in Jupyter notebooks
from IPython.display import Markdown, display


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


# Function to select random interactions and non-interactions for a subset of users
def select_interactions(
    trainset, num_users: int = 2, num_products: int = 2, seed: int = 42
) -> tuple:
    """
    Randomly select a subset of users and their interactions from the training set,
    along with a set of non-interacted products for comparison.

    Parameters:
    - trainset: The training set containing user-item interactions, provided by the Surprise library.
    - num_users (int, optional): Number of users to randomly select (default is 2).
    - num_products (int, optional): Number of products to select per user (default is 2).
    - seed (int, optional): Seed for random number generation to ensure reproducibility (default is 42).

    Returns:
    - tuple: A tuple containing dictionaries of random interactions and non-interactions for a subset of user IDs.
        - user_interactions (dict): A dictionary with user IDs as keys and a list of randomly selected (product, rating) tuples as values.
        - user_non_interactions (dict): A dictionary with user IDs as keys and a list of randomly selected non-interacted products as values.
    """

    # Set the random seed for reproducibility
    random.seed(seed)

    # Randomly select a few users from the trainset
    random_inner_uids = random.sample(trainset.all_users(), num_users)

    # Initialize dictionaries to store user interactions and all interacted products
    user_interactions = {}
    user_interacted_products = {}
    user_non_interactions = {}  # If you plan to use this later

    # Get the set of all product IDs in the trainset
    all_products = set(trainset.to_raw_iid(item) for item in trainset.all_items())

    # Iterate over the selected users in the trainset
    for inner_uid in random_inner_uids:
        # Convert inner user ID to raw user ID
        raw_user_id = trainset.to_raw_uid(inner_uid)

        # Get the list of (item_inner_id, rating) tuples for this user
        user_ratings = trainset.ur[inner_uid]

        # Convert item_inner_id to raw item IDs and store the interactions
        user_interacted_products[raw_user_id] = [
            (trainset.to_raw_iid(item_inner_id), rating)
            for item_inner_id, rating in user_ratings
        ]

        # Randomly select a few interacted products
        user_interactions[raw_user_id] = random.sample(
            user_interacted_products[raw_user_id], num_products
        )

        # Get the set of interacted product IDs for the current user
        interacted_products = set(
            item_id for item_id, _ in user_interacted_products[raw_user_id]
        )

        # Identify non-interacted products by subtracting interacted ones from all products
        non_interacted_products = list(all_products - interacted_products)

        # Randomly select non-interacted products
        if len(non_interacted_products) >= num_products:
            user_non_interactions[raw_user_id] = random.sample(
                non_interacted_products, num_products
            )

    # Display the user interactions and non-interactions
    display(Markdown(f"**User Interactions**"))
    display(user_interactions)

    display(Markdown(f"**User Non-Interactions**"))
    display(user_non_interactions)

    return user_interactions, user_non_interactions


# Function to display evaluation metrics for the recommender class evaluate instance
def display_eval_metrics(
    train_pred_met, train_rank_met, test_pred_met, test_rank_met
) -> None:
    """
    Display the evaluation metrics in DataFrame format.

    Parameters:
    - train_pred_met: Predictive metrics for the training set.
    - train_rank_met: Ranking metrics for the training set.
    - test_pred_met: Predictive metrics for the test set.
    - test_rank_met: Ranking metrics for the test set.
    """
    pred_metrics_df = pd.DataFrame(
        {"Trainset": train_pred_met, "Testset": test_pred_met}
    )
    rank_metrics_df = pd.DataFrame(
        {"Trainset": train_rank_met, "Testset": test_rank_met}
    )

    display(Markdown(f"**Predictive Quality Metrics**"))
    display(pred_metrics_df)
    display(Markdown(f"**Ranking Quality Metrics**"))
    display(rank_metrics_df)


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
