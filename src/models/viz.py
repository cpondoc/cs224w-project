"""
Visualization of different experiments for our tutorial.
"""

import matplotlib.pyplot as plt
import typing

def create_bar_graph(labels, values, title='Accuracies of Different Models + Tasks', x_label='', y_label='', file_name=None):
    """
    Create a bar graph with the given labels and values.
    """
    for x, y in zip(labels, values):
        plt.bar(x, y, width=0.4)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

def plot_roc_auc_vs_layers(
    layers_list: typing.List[int],
    roc_auc_scores: typing.List[float],
    title: str = "Test ROC AUC vs Number of Layers",
) -> None:
    """
    Create a line plot of ROC AUC scores against number of layers.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(
        layers_list,
        roc_auc_scores,
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=8,
    )
    plt.title(title, fontsize=16)
    plt.xlabel("Number of Layers", fontsize=14)
    plt.ylabel("Test ROC AUC", fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(layers_list)
    plt.tight_layout()
    plt.show()
