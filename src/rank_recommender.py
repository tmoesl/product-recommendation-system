"""
rank_recommender.py

Rank-Based Recommendation System Module

This module provides a class `RankRecommendationSystem` using Bayesian or arithmetic average scoring methods.
It supports computing scores, evaluating model performance, and generating top-N recommendations for all users.

Classes:
    InvalidMethodError: Custom exception for invalid scoring method.
    RankRecommendationSystem: Manages rank-based recommendation systems, computes scores, evaluates model performance, and generates top-N recommendations.

RankRecommendationSystem Methods:
    __init__: Initializes the RankRecommendationSystem with the selected scoring method.
    compute_scores: Computes rank-based scores using popularity, simple averages, and Bayesian averages.
    evaluate: Evaluates the trained model on training and test sets, and displays performance metrics.
    recommend: Generates and displays top N recommendations for all users.
    _calculate_predictions: Calculates predictions for the test set using the rank-based model.
"""

# ---------------------------------------------------------
# IMPORT LIBRARIES
# ---------------------------------------------------------

# Import required libraries
from typing import Dict, List, Optional, Tuple

import pandas as pd
from IPython.display import Markdown, display
from surprise import Dataset
from surprise.prediction_algorithms.predictions import Prediction

# Import modules and required functions from the src directory
from src.model_eval_functions import evaluate_model
from src.utils import display_eval_metrics

# ---------------------------------------------------------
# CLASS DEFINITIONS
# ---------------------------------------------------------


# Class for error handling
class InvalidMethodError(Exception):
    """Custom exception for invalid scoring method."""

    pass


# Class for managing rank based recommendation systems
class RankRecommendationSystem:
    """
    A rank-based recommender system using Bayesian or arithmetic average scoring methods.

    This class provides functionalities to compute scores, evaluate model performance on training and test sets,
    and generate top-N product recommendations for users based on the selected scoring method.
    """

    __slots__ = (
        "method",
        "algo_name",
        "key_column",
        "scores",
    )

    # Initialize the recommender system with the selected scoring method
    def __init__(self, method: str = "bayesian"):
        """
        Initialize the RankRecommendationSystem.

        Args:
            method (str, optional): The scoring method to use. Defaults to "bayesian".

        Raises:
            InvalidMethodError: If the provided method is not "bayesian" or "average".
        """
        self.method = method.lower()
        if self.method not in ["bayesian", "average"]:
            raise InvalidMethodError("Method must be either 'bayesian' or 'average'.")

        self.algo_name: str = "BAvg" if self.method == "bayesian" else "Avg"
        self.key_column: str = (
            "bayesian_rating" if self.method == "bayesian" else "avg_rating"
        )
        self.scores: Optional[pd.DataFrame] = None

    # Function to compute scores using popularity, simple averages and Bayesian averages
    def compute_scores(
        self, data: pd.DataFrame, gb_feature: str, filter_feature: str
    ) -> None:
        """
        Computes rank-based scores using popularity, simple averages and Bayesian averages.

        Args:
            data (pd.DataFrame): Input DataFrame containing user-product interactions ('user_id', 'prod_id', 'rating').
            gb_feature (str): Feature to group by (e.g., 'prod_id').
            filter_feature (str): Feature to compute scores on (e.g., 'rating').
        """
        # Calculate the count and average of ratings for each product
        model_scores = data.groupby(gb_feature)[filter_feature].agg(
            cnt_rating="count", avg_rating="mean"
        )

        # Define the global mean (average rating across all products)
        global_rating = data[filter_feature].mean()

        # Set the hyperparameter (average number of ratings across all products)
        kn = model_scores["cnt_rating"].mean()

        # Calculate the Bayesian Average for each product
        model_scores["bayesian_rating"] = (
            global_rating * kn + model_scores["cnt_rating"] * model_scores["avg_rating"]
        ) / (kn + model_scores["cnt_rating"])

        self.scores = model_scores

    # Function to evaluate the model on the training and test sets, and display performance metrics
    def evaluate(
        self, trainset: Dataset, testset: Dataset, k: int = 10, th: float = 3.5
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Evaluate the trained model on training and test sets, and display performance metrics.

        Args:
            trainset (Dataset): Training dataset.
            testset (Dataset): Test dataset.
            k (int): Number of recommendations to generate. Defaults to 10.
            th (float): Rating threshold. Defaults to 3.5.

        Returns:
            Tuple[Dict[str, float], Dict[str, float]]: A tuple containing dictionaries of train and test metrics.

        Raises:
            ValueError: If scores have not been computed yet.
        """
        if self.scores is None:
            raise ValueError(
                "Scores have not been computed. Call compute_scores() first."
            )

        # Convert the key column to a dictionary and calculate the global rating
        model_dict = self.scores[self.key_column].to_dict()
        global_rating = self.scores[self.key_column].mean()

        # Compute predictions using the rank-based model
        train_predictions = self._calculate_predictions(
            trainset.build_testset(), model_dict, global_rating
        )
        test_predictions = self._calculate_predictions(
            testset, model_dict, global_rating
        )

        # Evaluate the rank-based predictions on both sets
        train_pred_met, train_rank_met = evaluate_model(
            train_predictions, k=k, threshold=th
        )

        test_pred_met, test_rank_met = evaluate_model(
            test_predictions, k=k, threshold=th
        )

        # Combine and display results
        display_eval_metrics(
            train_pred_met,
            train_rank_met,
            test_pred_met,
            test_rank_met,
        )

        return {**train_rank_met, **train_pred_met}, {**test_rank_met, **test_pred_met}

    # Function to generate and display top N recommendations for all users
    def recommend(self, top_n: Optional[int] = None, threshold: int = 0) -> None:
        """
        Get the top N products based on the highest Bayesian or average rating, filtered by a minimum number of interactions.

        Args:
            top_n (Optional[int], optional): The number of top N products to return. Defaults to None.
            threshold (int, optional): Minimum number of interactions required for a product to be considered. Defaults to 0.

        Raises:
            ValueError: If scores have not been computed yet.
        """
        if self.scores is None:
            raise ValueError(
                "Scores have not been computed. Call compute_scores() first."
            )

        # Filter products that meet the minimum interaction threshold
        recommendations = self.scores[self.scores["cnt_rating"] > threshold]

        # Extract the top N products after applying the rating and interaction filters
        top_items = (
            recommendations[self.key_column].nlargest(top_n).round(2).reset_index()
        )
        top_items.columns = ["prod_id", "estimated_ratings"]

        # Display the recommendations
        display(Markdown("**Recommendations for All Users**"))
        display(top_items)

    # Helper function to compute predictions
    @staticmethod
    def _calculate_predictions(
        data: Dataset, model_dict: Dict[str, float], global_rating: float
    ) -> List[Prediction]:
        """
        Calculate predictions for the test set using the rank-based model.

        Args:
            data (Dataset): Training or test dataset.
            model_dict (Dict[str, float]): Dictionary containing item_id to rank score mappings.
            global_rating (float): Global average rating to use as a fallback.

        Returns:
            List[Prediction]: List of rank-based predictions.
        """
        predictions = []
        for user_id, item_id, true_r in data:
            # Retrieve the rank score for the item, defaulting to the global average if not found
            score = model_dict.get(item_id, global_rating)
            predictions.append((user_id, item_id, true_r, score, {}))

        return predictions


# ---------------------------------------------------------
# END OF SCRIPT
# ---------------------------------------------------------
