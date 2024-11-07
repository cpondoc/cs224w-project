"""
Defines the dataset, tasks, and table splits for each task for F1.

Modeled after `train_model.ipynb` tutorial in the Relbench GitHub repository.
"""
from relbench.datasets import get_dataset
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