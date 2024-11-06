# CS 224W -- Working on RelBench

## Installation

Official tutorial on training a model just doesn't work. We can make a PR around trying to get it to work, but general steps are [here](https://github.com/pyg-team/pytorch_geometric/discussions/9143):

```bash
pip install torch==2.4.0
pip install torch-geometric torch-sparse torch-scatter torch-cluster torch-spline-conv pyg-lib -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
pip install pytorch_frame
pip install relbench
```