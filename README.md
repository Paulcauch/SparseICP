# Sparse Iterative Closest Point Implementation
As part of a work for the "Point Cloud and 3D modelization" from the IASD/MVA course at Les Mines.

This repository contains an implementation in Python and an analysis report of the Sparse Iterative Closest Point (SICP) algorithm, as introduced in the paper:
```
[Sparse Iterative Closest Point] 
Sofien Bouaziz, Andrea Tagliasacchi, Mark Pauly  
Symposium on Geometry Processing 2013, Computer Graphics Forum
```

## Features

- Implementation of ICP for point-to-point and point-to-plane correspondences.
- Implementation of re-weight ICP for point-to-point correspondences. 
- Implementation of ICP with correspondences pruning for point-to-point correspondences.
- Implementation of the Sparse ICP algorithm for point-to-point and point-to-plane correspondences.
- Examples with popular 3D scan datasets (e.g., bunny and owl models).
- Utilities for data preprocessing.
- Visualization tools to inspect the alignment results and convergence behavior.

## Running a test :

To run the SICP algorithm on a specific dataset, use the main.py script with appropriate arguments. For example, to run the SICP algorithm on the owl dataset with parameter p=1.0 and 30 iterations:
```
python main.py --test owls_plane --p 1 --ite 30
```

