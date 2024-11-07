"""
Helps to set up a `NeighborLoader` for each of the Relbench tasks.
"""

from relbench.modeling.graph import get_node_train_table_input
from torch_geometric.loader import NeighborLoader


def get_loader(train_table, val_table, test_table, task, data):
    """
    Creates a `NeighborLoader` object for each of the train, test, and val splits.
    """
    loader_dict = {}

    for split, table in [
        ("train", train_table),
        ("val", val_table),
        ("test", test_table),
    ]:
        table_input = get_node_train_table_input(
            table=table,
            task=task,
        )
        entity_table = table_input.nodes[0]
        loader_dict[split] = NeighborLoader(
            data,
            num_neighbors=[128 for _ in range(2)],
            time_attr="time",
            input_nodes=table_input.nodes,
            input_time=table_input.time,
            transform=table_input.transform,
            batch_size=512,
            temporal_strategy="uniform",
            shuffle=split == "train",
            num_workers=0,
            persistent_workers=False,
        )

    return loader_dict, entity_table
