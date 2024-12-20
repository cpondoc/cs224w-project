# Exploring the Impacts of GNN Architecture and Scale on Learning from Relational Data

By: Joseph Guman, Atindra Jha, and Christopher Pondoc

## Abstract

Our project involves evaluating the performance of graph neural network (GNN) methods on relational databases using RelBench. In particular, we’ll use the open-source implementation of Relational Deep Learning (RDL) as a launching pad and explore several choices around architecture and scale to optimize performance. The aim is to spark more excitement and investigations into the intersection of GNNs and relational data by adopting RelBench and PyG. 

## Structure

The core tutorial is currently in `tutorial.ipynb`. Our intention is that the tutorial will only focus on helping the user explore the different ablation axes -- choice of task, architecture, scale, etc. -- while abstracting away all of the nitty-gritty of Relbench. The code for the latter is in `src`.

We also ran some other experiments on larger datasets. The successful ones can be found in `experiments/`. `old/` contains some older, unsuccessful experiments (i.e., because we were unable to load in all of the data from RelBench).

## Progress

See relevant documents in the `reports` folder.

## Installation

If building our code from scratch, you can set it up just like any normal project. We specifically use Python 3.9.20 for our installation:

```bash
python3 -m venv env
pip install -r requirements.txt
```

If setting up on Colab, note that the official tutorial from Relbench on training a model doesn't work. We will be filing a PR around trying to get it to work, but general steps are [here](https://github.com/pyg-team/pytorch_geometric/discussions/9143):

```bash
!pip install torch==2.4.0
!pip install torch-geometric torch-sparse torch-scatter torch-cluster torch-spline-conv pyg-lib -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
!pip install pytorch_frame
!pip install relbench
!pip install sentence_transformers
!pip install lightgbm
!pip install optuna
!pip install matplotlib
```