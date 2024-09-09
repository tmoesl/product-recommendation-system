"""
This module contains custom functions for model evaluation, recommendation, and analysis 
in recommendation systems. The functions support various tasks such as calculating ranking and quality metrics, 
performing cross-validation, running grid search, and generating personalized product recommendations. 
These functions are designed to enhance data science workflows by integrating with standard libraries 
such as NumPy, Pandas, and the Surprise library for recommendation algorithms.

The module is structured as follows:

1. IMPORT LIBRARIES:
   - Data manipulation and analysis libraries (NumPy, Pandas, collections, random)
   - Data visualization libraries in Jupyter notebooks (IPython.display)
   - Model evaluation and recommendation libraries (Surprise)

2. FUNCTIONS:
   - top_n_products: Retrieves top N products based on Bayesian average rating.
   - calculate_predictive_metrics: Calculates predictive quality metrics like Precision@K, Recall@K, and F1 Score@K.
   - calculate_ranking_metrics: Computes ranking quality metrics such as MRR, MAP, and Hit Rate@K.
   - evaluate_model: Evaluates a recommendation model using both predictive and ranking metrics.
   - get_recommendations: Generates top N product recommendations for a user based on a trained model.
   - cross_val: Performs cross-validation on the provided dataset using a specified algorithm.
   - baseline_gs: Runs GridSearchCV for multiple algorithms with specified parameter grids.
   - select_interactions: Randomly selects a subset of users and their interactions and non-interactions from the training set.
"""

# ---------------------------------------------------------
# 1. IMPORT LIBRARIES
# ---------------------------------------------------------


# Import libraries for data manipulation and analysis
import numpy as np
import pandas as pd
from collections import defaultdict
import random

# Import libraries for data visualization
from IPython.display import display, Markdown

# Import libraries for model evaluation
from surprise import accuracy
from surprise.model_selection import GridSearchCV, cross_validate


# ---------------------------------------------------------
# 2. FUNCTIONS
# ---------------------------------------------------------


# Function to get the top n products based on the highest Bayesion average rating, filtered by a minimum interactions
def top_n_products(data: pd.DataFrame, n: int = 10, threshold: int = 0) -> pd.DataFrame:
    """
    Get the top n products based on the highest Bayesian average rating, filtered by a minimum number of interactions.

    Parameters:
    - data (pd.DataFrame): The input DataFrame containing product information.
    - n (int): The number of top products to return.
    - threshold (int, optional): Minimum number of interactions required for a product to be considered. Default is 0.

    Returns:
    - pd.DataFrame: A DataFrame containing the top n products sorted by their Bayesian average ratings.
    """

    # Filter products that meet the minimum interaction threshold to ensure sufficient data reliability.
    recommendations = data[data["cnt_rating"] > threshold]

    # Sort products by Bayesian average rating to prioritize items with a balance of high ratings and adequate interaction count.
    recommendations = recommendations.sort_values(
        by=["bayesian_avg"], ascending=[False]
    )

    # Extract the top n products after applying the rating and interaction filters.
    return recommendations["bayesian_avg"].nlargest(n).round(2).reset_index()


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
    data: pd.DataFrame, algo, user_id: str, top_n: int = 5
) -> pd.DataFrame:
    """
    This function recommends top_n products for a given user based on the specified algorithm.

    Parameters:
    - data (pd.DataFrame): The input DataFrame containing user-product interactions, with columns 'user_id', 'prod_id', and 'rating'.
    - user_id (str): The unique identifier of the user for whom the recommendations are to be generated.
    - top_n (int, optional): The number of top recommendations to generate for the given user (default is 5).
    - algo: The trained recommendation algorithm from the Surprise library to use for predicting ratings.

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
        est = algo.predict(user_id, item_id).est
        details = algo.predict(user_id, item_id).details

        # Append the predicted ratings
        recommendations.append((item_id, est, details))

    # Rank products by descending predicted ratings.
    recommendations.sort(key=lambda x: x[1], reverse=True)

    # Convert recommendations to DataFrame and round predicted ratings
    recommendations_df = pd.DataFrame(
        recommendations[:top_n],
        columns=["prod_id", "estimated_ratings", "details"],
    ).assign(estimated_ratings=lambda df: df["estimated_ratings"].round(2))

    # Return a DataFrame of the top n recommended products, including predicted ratings.
    return recommendations_df


# Function to perform cross validation on a dataset for specified algorithm and metrics
def cross_val(algo, data, measures=["RMSE", "MAE"], cv: int = 6) -> dict:
    """
    Perform cross-validation on the provided dataset using a specified algorithm and evaluation metrics.

    Parameters:
    - algo: The algorithm instance from the Surprise library to use for model training and evaluation.
    - data (Dataset): A Surprise library dataset containing user-item interactions.
    - measures (list, optional): List of performance measures to evaluate during cross-validation (default is ["RMSE", "MAE"]).
    - cv (int, optional): Number of cross-validation folds (default is 6).

    Returns:
    - dict: A dictionary containing cross-validation results for each evaluation measure.
    """

    # Execute cross-validation with results printed via verbose=True
    cval = cross_validate(
        algo=algo.model, data=data, measures=measures, cv=cv, n_jobs=-1, verbose=True
    )
    return cval


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

    return user_interactions, user_non_interactions


# ---------------------------------------------------------
# ---------------------------------------------------------
