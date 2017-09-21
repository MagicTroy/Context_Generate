# Context_Generate

This repository used for generating context information based on Deep Learning tehcniques

This branch is basically concatenate auxiliary info, such as beer/hotel ratings, which is also the basic GCN model.

It read the beeradvocate dataset directly, then sample the dataset by given a user splitting by a fraction number. After that, it generate one hot vector on each characters on the sample by using numpy

Then feeding those data into a lstm rnn network.