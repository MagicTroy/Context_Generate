# Context_Generate

This repository used for generating context information based on Deep Learning tehcniques

This version aim to concatenate auxiliary values, user ids, and item ids as feeding data into model.

All of them are using one hot to represents, which all generate by tensorflow.

The difference between this version with "concatenation" one is this version trained a single model for a particular user, while the previous one train all info in total.

The results of this version seems over-fitting.


