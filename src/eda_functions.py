"""
eda_functions.py

This module contains custom functions for data analysis and visualization,
including statistical summaries and plotting tools. 
These functions are designed for use in data science workflows and are 
integrated with standard libraries such as NumPy, Pandas, Math, Matplotlib, and Seaborn.

The module is structured as follows:

1. IMPORT LIBRARIES:
   - Data manipulation libraries (NumPy, Pandas, Math)
   - Data visualization libraries (Matplotlib, Seaborn)

2. SET STANDARDIZED VARIABLES:
   - Standard color palette for plotting (Seaborn Set2)
   - Standard rating order and custom color palette for bar plots

3. FUNCTIONS:
   - feature_stats: Calculates counts and percentage distribution of unique values.
   - annotate_bars: Adds annotations to bar plots with counts or percentages.
   - barplot: Plots a bar chart for a categorical variable with optional annotations.
   - plot_interaction_histplot: Plots histogram for distribution of user-item interactions.
   - plot_interaction_boxplot: Plots boxplot for distribution of user-item interactions.
   - nlargest_entries: Retrieves and displays top N entries with highest counts.
   - barplot_subplots: Creates subplots of bar plots for specified features.
"""

# ---------------------------------------------------------
# 1. IMPORT LIBRARIES
# ---------------------------------------------------------

# Import libraries for data manipulation
import numpy as np
import pandas as pd
import math

# Import libraries for data visualization
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------
# 2. SET STANDARDIZED VARIABLES
# ---------------------------------------------------------

# Set standard color palette for plotting
palette_color = sns.color_palette("Set2")

# Set standard rating order and create custome color palette
rating_order = [1.0, 2.0, 3.0, 4.0, 5.0]
custom_palette = {
    str(rating): color for rating, color in zip(rating_order, palette_color)
}


# ---------------------------------------------------------
# 3. FUNCTIONS
# ---------------------------------------------------------
# Function to calculate the distribution of a categorical variable
def feature_stats(data: pd.DataFrame, feature: str) -> None:
    """
    Calculate the counts and percentage distribution of unique values in a specified feature.

    Parameters:
    - data (DataFrame): The input DataFrame.
    - feature (str): The column name for which the statistics are calculated.

    Returns:
    - None: Prints the distribution statistics to the console.
    """
    rating_percentage_df = (
        data[feature]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .reset_index()
        .rename(columns={"index": feature, "proportion": "percentage"})
    )
    print(f"{feature.capitalize()} Distribution")
    print("-" * 30)
    print(rating_percentage_df.sort_values(by=feature).to_string(index=False))
    print("-" * 30)


# Function to annotate each bar in barplot
def annotate_bars(ax, total: int, perc: bool = False) -> None:
    """
    Annotate each bar in a bar plot with either percentage or count labels.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes object containing the plot.
    - total (int): The total number of observations, used for calculating percentages.
    - perc (bool, optional): If True, annotate bars with percentages; otherwise, use counts. Default is False.

    Returns:
    - None: Modifies the plot in place by adding annotations.
    """
    # Annotate each bar by percentage or count
    for container in ax.containers:
        labels = [
            f"{100 * height / total:.1f}%" if perc else f"{height:.0f}"
            for height in container.datavalues
        ]
        ax.bar_label(container, labels=labels, label_type="edge", fontsize=10)


# Function to plot categorical variables with barplot and optional relative percentage annotations
def barplot(data: pd.DataFrame, feature: str, perc: bool = False) -> None:
    """
    Plot a bar chart for a categorical variable and display its distribution statistics.

    Parameters:
    - data (pd.DataFrame): The input DataFrame containing the data.
    - feature (str): The column name of the categorical variable to plot.
    - perc (bool, optional): If True, annotate bars with percentages; otherwise, use counts. Default is False.

    Returns:
    - None: Displays the bar plot and prints the mean and median of the feature.
    """
    # Display the distribution statistics for the feature
    feature_stats(data, feature)

    # Compute mean and median
    fmean = data[feature].mean()
    fmedian = data[feature].median()

    fig = plt.figure(figsize=(14, 8))
    total = len(data[feature])

    # Plot countplot
    sns.countplot(
        data=data,
        x=feature,
        palette=custom_palette,
        order=rating_order,
    )

    # Add title
    ax = plt.gca()
    formatted_title = feature.replace("_", " ").title()
    ax.set_title(f"Distribution of {formatted_title}")

    # Annotate each bar by percentage or count
    annotate_bars(ax, total, perc)

    # Adjust layout and show the plot
    plt.tight_layout(pad=2.0)
    plt.show()

    # Print the key statistics
    print(f"Mean: {np.round(fmean, 2)}, Median: {np.round(fmedian, 2)}")


# Function to plot interaction data distribution with histplot
def plot_interaction_histplot(
    data: pd.DataFrame, groupby_feature: str, select_feature: str, ax
) -> None:
    """
    Plot a histogram for the distribution of user-item interactions and annotate it with key statistics.

    Parameters:
    - data (pd.DataFrame): The input DataFrame containing the interaction data.
    - groupby_feature (str): The column name to group data by (e.g., users or items).
    - select_feature (str): The column name representing the feature to be counted for each group.
    - ax (matplotlib.axes.Axes): The matplotlib Axes object where the plot will be drawn.

    Returns:
    - None: Displays the histogram plot with mean, median, and max annotations.
    """
    # Finding user-item interactions distribution
    count_interactions = data.groupby(groupby_feature)[select_feature].count()

    # Calculate mean and median
    mean_val = count_interactions.mean()
    median_val = count_interactions.median()
    max_val = count_interactions.max()

    # Calculate xlim rounded up to the next tick
    xlim = np.ceil(max_val / 10) * 10

    # Plot histplot
    sns.histplot(
        count_interactions,
        kde=True,
        color="green",
        edgecolor="darkgreen",
        linewidth=1.0,
        ax=ax,
    )
    # Add limits, yscale (square root), titel, and label
    ax.set_xlim(0, xlim)
    ax.set_yscale("function", functions=(np.sqrt, lambda x: x**2))
    formatted_title = groupby_feature.replace("_", " ").title()
    ax.set_title(f"Distribution of {formatted_title} Interactions")
    ax.set_xlabel("Number of Interactions")

    # Adding lines for max, mean, and median values
    ax.axvline(
        max_val,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Max: {max_val:.2f}",
    )

    ax.axvline(
        mean_val,
        color="blue",
        linestyle="-",
        linewidth=1.5,
        label=f"Mean: {mean_val:.2f}",
    )

    ax.axvline(
        median_val,
        color="red",
        linestyle="-",
        linewidth=1.5,
        label=f"Median: {median_val:.2f}",
    )

    # Calculate the x-coordinate for the legend
    legend_x = max_val / xlim - 0.005

    # Add legend
    ax.legend(loc="upper right", bbox_to_anchor=(legend_x, 1))


# Function to plot interaction data distribution with boxplot
def plot_interaction_boxplot(
    data: pd.DataFrame, groupby_feature: str, select_feature: str, ax
) -> None:
    """
    Plot a boxplot for the distribution of user-item interactions and annotate it with statistical information.

    Parameters:
    - data (pd.DataFrame): The input DataFrame containing the interaction data.
    - groupby_feature (str): The column name to group data by (e.g., users or items).
    - select_feature (str): The column name representing the feature to be counted for each group.
    - ax (matplotlib.axes.Axes): The matplotlib Axes object where the plot will be drawn.

    Returns:
    - None: Displays the boxplot with statistical annotations.
    """
    # Finding user-item interactions distribution
    count_interactions = data.groupby(groupby_feature)[select_feature].count()

    # Adding statistical annotations
    max_val = count_interactions.max()

    # Calculate xlim rounded up to the next tick
    xlim = np.ceil(max_val / 10) * 10

    # Plot boxplot
    sns.boxplot(
        x=count_interactions,
        ax=ax,
        color="#FFC877",
        medianprops=dict(visible=True, color="red"),
        flierprops=dict(marker="o", markersize=3),
    )
    # Add label and limits
    ax.set_xlim(0, xlim)
    ax.set_xlabel("Number of Interactions")


# Function to calculate and print the top 10 entries with the highest counts
def nlargest_entries(data: pd.DataFrame, feature: str, n: int = 10) -> pd.DataFrame:
    """
    Retrieve and display the top N entries with the highest counts for a specified feature.

    Parameters:
    - data (pd.DataFrame): The input DataFrame containing the data.
    - feature (str): The column name for which the top entries are to be determined.
    - n (int, optional): The number of top entries to return. Defaults to 10.

    Returns:
    - pd.DataFrame: A DataFrame containing the top N entries and their counts.
    """
    top_entries = data[feature].value_counts().nlargest(10)
    top_entries_df = top_entries.reset_index()  # Converts Series to DataFrame
    top_entries_df.columns = [feature, "count"]  # Rename columns of DataFrame

    print(f"Top {n} Entries")
    print("-" * 30)
    print(top_entries_df.to_string(index=False))
    print("-" * 30)

    return top_entries


# Function to plot categorical variables with barplot and optional relative percentage annotations
def barplot_subplots(data: pd.DataFrame, features: list, perc: bool = False) -> None:
    """
    Plot subplots of barplots for a specified feature and its associated ratings.

    Parameters:
    - data (pd.DataFrame): The input DataFrame containing the data.
    - features (list): A list of two elements; the first element is the feature to filter by,
      and the second element is the categorical feature to plot.
    - perc (bool, optional): Whether to annotate bars by percentage (True) or count (False).
      Defaults to False.

    Returns:
    - None: This function only plots the barplot subplots.
    """
    # Extract the top 10 entries based on the total number of ratings
    top_entries = nlargest_entries(data, features[0])
    top_entries = top_entries.index

    # Determine the number of columns and rows for the subplot
    num_ids = len(top_entries)
    num_cols = int(math.ceil(math.sqrt(num_ids))) if num_ids != 3 else 3
    num_rows = int(math.ceil(num_ids / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))
    axes = axes.flatten() if num_ids > 1 else [axes]

    for i, id in enumerate(top_entries):
        data_filtered = data[data[features[0]] == id]
        total = len(data_filtered)

        ax = axes[i]
        sns.countplot(
            data=data_filtered,
            x=features[1],
            ax=ax,
            palette=custom_palette,
            order=rating_order,
        )
        ax.set_title(f"{features[0]}: {id}")

        # Annotate each bar by percentage or count
        annotate_bars(ax, total, perc)

    # Hide any unused subplots
    for j in range(num_ids, num_rows * num_cols):
        fig.delaxes(axes[j])

    # Adjust layout and show the plot
    plt.tight_layout(pad=2.0)
    plt.show()
