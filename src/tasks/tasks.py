"""
Code to handle preprocessing of data + task information for Relbench.

Modeled after `train_model.ipynb` tutorial in the Relbench GitHub repository.
"""

from relbench.datasets import get_dataset
from relbench.modeling.utils import get_stype_proposal
from relbench.tasks import get_task


def initialize_task(dataset_name, task):
    """
    Returns associated information with a task in Relbench.
    """
    dataset = get_dataset(dataset_name, download=True)
    task_obj = get_task(dataset_name, task, download=True)

    train_table = task_obj.get_table("train")
    val_table = task_obj.get_table("val")
    test_table = task_obj.get_table("test")

    return dataset, task_obj, train_table, val_table, test_table


def db_to_graph(dataset):
    """
    Turn database data into a graph for Relbench.
    """
    db = dataset.get_db()
    col_to_stype_dict = get_stype_proposal(db)
    return db, col_to_stype_dict
