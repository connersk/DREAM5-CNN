# DREAM5-CNN
MIT 6.867 course project from Fall 2017, co-authored with Ellen Zhong (@zhonge)

Implemented convolutional neural networks (CNN) to predict transcription factor (TF) binding affinities, using data from the DREAM5 challenge https://www.synapse.org/#!Synapse:syn2887863/wiki/72185.

Performed architecture (filter width, number of feature detectors, number of nodes in the convolutational layer) and parameter (dropout rate, regularization strength) sweeps to develop an optimized network.

Developed a regression CNN for predicting transcription factor binding with improved Pearson's correlation, AUC, and average precision over the DeepBind CNN architecture (https://www.nature.com/articles/nbt.3300).

Developed classification CNNs to predict whether a given TF is a low or high affinity binder. As this is a highly unbalanced classification problem (high affinity binders are very rare), we implemented three techniques for dealing with unbalanced classification: 1) oversampling data in the minority class 2) undersampling data in the majority class (low binder) data points 3) Using a loss function that was weighted by the number of data points in each class.
