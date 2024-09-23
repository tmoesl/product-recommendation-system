"""
model_eval_functions.py

This module contains custom functions for model evaluation, recommendation, and analysis 
in recommendation systems. The functions support various tasks such as calculating ranking and quality metrics, 
running grid searches, and generating personalized product recommendations. 
These functions are designed to enhance data science workflows by integrating with standard libraries 
such as NumPy, Pandas, and the Surprise library for recommendation algorithms.

The module is structured as follows:

1. IMPORT LIBRARIES:
   - Data manipulation and analysis libraries (NumPy, Pandas, collections)
   - Data visualization libraries in Jupyter notebooks (IPython.display)
   - Model evaluation and recommendation libraries (Surprise)

2. FUNCTIONS:
   - calculate_predictive_metrics: Calculates predictive quality metrics like Precision@K, Recall@K, and F1 Score@K.
   - calculate_ranking_metrics: Computes ranking quality metrics such as MRR, MAP, and Hit Rate@K.
   - evaluate_model: Evaluates a recommendation model using both predictive and ranking metrics.
   - get_recommendations: Generates top N product recommendations for a user based on a trained model.
   - baseline_gs: Runs GridSearchCV for multiple algorithms with specified parameter grids.
"""

# ---------------------------------------------------------
# 1. IMPORT LIBRARIES
# ---------------------------------------------------------


# Import libraries for data manipulation and analysis
from collections import defaultdict

import numpy as np
import pandas as pd

# Import libraries for data visualization
from IPython.display import Markdown, display

# Import libraries for model evaluation
from surprise import accuracy
from surprise.model_selection import GridSearchCV

# ---------------------------------------------------------
# 2. FUNCTIONS
# ---------------------------------------------------------


#  Function to calculate predictive quality metrics for a recommendation model
def calculate_predictive_metrics(
    user_est_true: dict, k: int = 10, threshold: float = 3.5
) -> dict:
    """
    Calculate predictive quality metrics for a recommendation model, including Precision@K, Recall@K, and F1 Score@K.

    Parameters:
    - user_est_true (dict): Dictionary mapping user IDs to a list of (estimated rating, true rating) tuples.
    - k (int, optional): The number of top recommendations to consider (default is 10).
    - threshold (float, optional): The rating threshold to consider an item as relevant (default is 3.5).

    Returns:
    - A dictionary containing:
      - Precision@K: The proportion of recommended items in the top K that are relevant.
      - Recall@K: The proportion of relevant items that are recommended in the top K.
      - F1 Score@K: The harmonic mean of Precision@K and Recall@K.
    """

    # Initialize dictionaries to store precision and recall for each user
    precisions = {}
    recalls = {}

    # Recall@K: Proportion of relevant items that are recommended (0 if n_rel is 0)
    # Precision@K: Proportion of recommended items that are relevant (0 if n_rec_k is 0)

    for uid, user_ratings in user_est_true.items():
        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for _, true_r in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for est, _ in user_ratings[:k])
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold))
            for est, true_r in user_ratings[:k]
        )

        # Calculate Precision@K for the user
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

        # Calculate Recall@K for the user
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    # Compute overall Precision@K and Recall@K by averaging across all users.
    precision = round(sum(prec for prec in precisions.values()) / len(precisions), 3)
    recall = round(sum(rec for rec in recalls.values()) / len(recalls), 3)

    # Derive F1 Score@K, representing the harmonic mean of precision and recall.
    f1 = (
        round((2 * precision * recall) / (precision + recall), 3)
        if (precision + recall) != 0
        else 0
    )

    # Return all metrics in a dictionary
    return {
        "Precision@K": precision,
        "Recall@K": recall,
        "F1 Score@K": f1,
    }


#  Function to calculate ranking quality metrics for a recommendation model
def calculate_ranking_metrics(
    user_est_true: dict, k: int = 10, threshold: float = 3.5
) -> dict:
    """
    Calculate ranking quality metrics for a recommendation model, including MRR, MAP, and Hit Rate@K.

    Parameters:
    - user_est_true (dict): Dictionary mapping user IDs to a list of (estimated rating, true rating) tuples.
    - k (int, optional): The number of top recommendations to consider (default is 10).
    - threshold (float, optional): The rating threshold to consider an item as relevant (default is 3.5).

    Returns:
    - A dictionary containing:
      - MRR: Mean Reciprocal Rank, the average rank of the first relevant item for each user.
      - MAP: Mean Average Precision, the average of the precision scores for all relevant items.
      - Hit Rate@K: The proportion of users for whom at least one relevant item appears in the top K recommendations.
    """

    # Initialize sums for MRR, MAP, and count for Hit Rate
    mrr_sum = 0
    map_sum = 0
    hit_rate_count = 0

    # Get the total number of users
    num_users = len(user_est_true)

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Calculate MRR for the user
        for rank, (est, true_r) in enumerate(user_ratings[:k], start=1):
            if true_r >= threshold:
                mrr_sum += 1 / rank
                break

        # Calculate MAP for the user
        hits = 0
        avg_precision = 0
        for rank, (est, true_r) in enumerate(user_ratings[:k], start=1):
            if true_r >= threshold:
                hits += 1
                avg_precision += hits / rank

        if hits > 0:
            map_sum += avg_precision / hits

        # Calculate Hit Rate for the user
        if hits > 0:
            hit_rate_count += 1

    # Calculate the average of all metrics
    mrr = round(mrr_sum / num_users, 3)
    map_score = round(map_sum / num_users, 3)
    hit_rate = round(hit_rate_count / num_users, 3)

    # Return all metrics in a dictionary
    return {
        "MRR": mrr,
        "MAP": map_score,
        "Hit Rate@K": hit_rate,
    }


# Function to evaluate a recommendation model using both predictive quality metrics
def evaluate_model(predictions: list, k: int = 10, threshold: float = 3.5) -> tuple:
    """
    Evaluate a recommendation model using both predictive quality metrics (RMSE, Precision@K, Recall@K, F1 Score@K)
    and ranking quality metrics (MRR, MAP, Hit Rate@K).

    Parameters:
    - predictions (list): List of predictions made by the recommendation model (based on trainset or testset).
    - k (int, optional): The number of top recommendations to consider (default is 10).
    - threshold (float, optional): The rating threshold to consider an item as relevant (default is 3.5).

    Returns:
    - tuple: A tuple containing dictionaries of predictive metrics and ranking metrics.
    """

    # Compute RMSE to measure the average prediction error of the recommendation model
    rmse = round(accuracy.rmse(predictions, verbose=False), 3)

    # Initialize a dictionary to store the estimated and true ratings for each user
    user_est_true = defaultdict(list)

    # Populate the dictionary with predictions: mapping user ID to a list of (estimated rating, true rating) tuples
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    # Calculate predictive quality metrics (Precision@K, Recall@K, F1 Score@K)
    predictive_metrics = calculate_predictive_metrics(user_est_true, k, threshold)

    # Add RMSE to the predictive metrics
    predictive_metrics = {"RMSE": rmse, **predictive_metrics}

    # Calculate ranking quality metrics (MRR, MAP, Hit Rate@K)
    ranking_metrics = calculate_ranking_metrics(user_est_true, k, threshold)

    # Return all scores
    return predictive_metrics, ranking_metrics


# Function to obtain the top N products for a given user based on the specified recommendation model
def get_recommendations(
    data: pd.DataFrame, algo, user_id: str, top_n: int = None
) -> pd.DataFrame:
    """
    This function recommends top_n products for a given user based on the specified algorithm.

    Parameters:
    - data (pd.DataFrame): The input DataFrame containing user-product interactions, with columns 'user_id', 'prod_id', and 'rating'.
    - algo: The trained recommendation algorithm from the Surprise library to use for predicting ratings.
    - user_id (str): The unique identifier of the user for whom the recommendations are to be generated.
    - top_n (int, optional): The number of top recommendations to generate for the given user (default is None).

    Returns:
    - pd.DataFrame: A DataFrame containing the top N recommended products for the given 'user_id', including estimated ratings and additional details.
    """

    # Initialize list to store recommendations
    recommendations = []

    # Create an user item interactions matrix
    user_item_interactions_matrix = data.pivot(
        index="user_id", columns="prod_id", values="rating"
    )

    # Extracte those product ids which the user_id has not interacted yet
    non_interacted_products = user_item_interactions_matrix.loc[user_id][
        user_item_interactions_matrix.loc[user_id].isnull()
    ].index.tolist()

    # Loop through each of the product ids which user_id has not interacted yet
    for item_id in non_interacted_products:

        # Predict the ratings for those non interacted product ids by this user
        prediction = algo.predict(user_id, item_id)
        est = prediction.est
        details = prediction.details

        # Append the predicted ratings
        recommendations.append((item_id, est, details))

    # Rank products by descending predicted ratings
    recommendations.sort(key=lambda x: x[1], reverse=True)

    # Set default for top_n to the number of unique products if not specified
    if top_n is None:
        top_n = len(non_interacted_products)

    # Convert recommendations to DataFrame and round predicted ratings
    recommendations_df = pd.DataFrame(
        recommendations[:top_n],
        columns=["prod_id", "estimated_ratings", "details"],
    ).assign(estimated_ratings=lambda df: df["estimated_ratings"].round(2))

    # Return a DataFrame of the top n recommended products, including predicted ratings.
    return recommendations_df


# Function to run baseline gridsearch for multiple algorithms with specified parameter grids
def baseline_gs(
    data: pd.DataFrame,
    algos: dict,
    param_grids: dict,
    measures: list = ["rmse"],
    cv: int = 5,
):
    """
    Run GridSearchCV for multiple algorithms with specified parameter grids for baseline investigation.

    Parameters:
    - data (Dataset): A Surprise library dataset containing user-item interactions.
    - algos: Dictionary of algorithm instances and classes from the Surprise library.
    - param_grids: Dictionary of parameter grids for each algorithm.
    - measures: List of performance measures to evaluate (default is ["rmse"]).
    - cv: Number of cross-validation folds (default is 5).

    Returns:
    - tuple: A tuple containing the name of the best algorithm and its best parameters based on RMSE.
    """

    # Initialize variables to track the best model
    best_rmse = float("inf")  # Initialize the best RMSE to a large value
    best_model_name = None
    best_model_params = None

    # Initialize dictionary to store the models
    models = {}

    for name, algo in algos.items():
        # Retrieve the parameter grid for the current algorithm
        param_grid = param_grids.get(name, {})

        # Initialize GridSearchCV for the current algorithm
        model_gs = GridSearchCV(algo, param_grid, measures=measures, cv=cv, n_jobs=-1)
        model_gs.fit(data)
        models[name] = model_gs

        # Update the best model if the current model's RMSE is lower
        current_rmse = model_gs.best_score["rmse"]
        if current_rmse < best_rmse:
            best_rmse = current_rmse
            best_model_name = name
            best_model_params = model_gs.best_params["rmse"]

    for name in algos.keys():
        # Display results for the current algorithm
        display(Markdown(f"**{name}:**"))
        print(f'RMSE: {round(models[name].best_score["rmse"], 3)}')
        print(f'Parameters: {models[name].best_params["rmse"]}')

    # Display the best baseline model selection
    width = 100
    formatted_message = f" BASELINE MODEL SELECTION: {best_model_name} ".center(
        width, "-"
    )
    display(Markdown(f"**{formatted_message}**"))

    return best_model_name, best_model_params


# ---------------------------------------------------------
# ---------------------------------------------------------
