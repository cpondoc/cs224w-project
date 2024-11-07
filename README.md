# Exploring the Impacts of Architecture and Scale on GNN Performance on Relational Data

By: Joseph Guman, Atindra Jha, and Christopher Pondoc

## Abstract

For our CS 224W project, our team will evaluate the performance of graph neural network (GNN) methods on relational databases using RelBench. In particular, we’ll use the open-source implementation of Relational Deep Learning (RDL) as a launching pad and explore several choices around architecture and scale to optimize performance. During this process, we would also like to extend the existing RelBench package to enhance the user experience by allowing users to add their own SQL datasets to the pipeline for easy training and evaluation. The aim is to spark more excitement and investigations into the intersection of GNNs and relational data by adopting RelBench and PyG. 

## Installation

Official tutorial on training a model just doesn't work. We can make a PR around trying to get it to work, but general steps are [here](https://github.com/pyg-team/pytorch_geometric/discussions/9143):

```bash
pip install torch==2.4.0
pip install torch-geometric torch-sparse torch-scatter torch-cluster torch-spline-conv pyg-lib -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
pip install pytorch_frame
pip install relbench
```