# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains marketing data about individuals. It is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. The dataset contains inputs such as profession, job, marital status, education, period, duration, campaign information and so on. The classification goal is to predict whether the client will subscribe to a bank term deposit where output can be yes or no (column y).

The best performing model on the data using Azure's AutoML is VotingEnsemble classifier. With Azure's AutoML, Ireceived an accuracy score of 0.917 within just 12 iterations. The best performance with the HyperDrive option is the accuracy of 0.9160.


## Scikit-learn Pipeline
Following were the major steps carried out to create the Scikit-learn Pipeline:
1) Creation of TabularDataset using TabularDatasetFactory.[ dataset here](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv)
2) The raw data obtained is thus preprocessed in training ('train.py') script. Some of the processes performed are:
    a) Dropping null values
    b) Using one-hot-encoding on some features
    c) Separating the target column ('y')
3) Then the processed data is split into train and test sets.
4) From the HyperDrive runs, training the logistic regression model using arguments .
5) Calculating the accuracy score.


The classification method used here is logistic regression. It uses a fitted logistic function and a threshold. The parameters available within the 'train.py' script are C which is the Regularization while max_iter which is the maximum number of iterations.

**Parameter Sampler**

Azure supports three types of parameter sampling techniques- Random sampling,Grid sampling and Bayesian sampling. I decided to choose random parameter sampling  because it is faster and supports early termination of low-performance runs. RandomParameterSampling supports discrete and continuous hyperparameters.  Compared to the other techniques available, it does not require pre-defined values (like grid search) and can make full use of all the available nodes (unlike bayesian parameter sampling). It is preferable to gridsearch as it is highly unlikely that the specified values in gridsearch are optimal, while there is a chance that the ones obtained randomly are closer to ideal values.

**Early Stopping Policy**

I selected the BanditPolicy stopping policy because it terminates runs where the primary metric is not within the slack factor compared to the best run. This helps in ignoring runs which we won't result in the best run, resulting in saving time as well as resources for the experiment.


## AutoML

It was a classification task, so the primary metric that was to be maximized was set to 'accuracy'. We provided the cleaned version of the data, and number of iterations was set to a small value(12 in this project). The best model selected by autoML was a voting ensemble (91.7% accurate) which was slightly better than the score achieved using HyperDrive.

## Pipeline comparison

The two models performed very similarly in terms of accuracy, with the hyperdive model achieving accuracy of 0.916 and the autoML model achieving accuracy of 0.917. The difference in accuracy could be slight variations in the cross-validation process. The pipelines use the same data cleansing process, however autoML tests a number of scalers in combination with models, adding a preprocessing step prior to model training. 

Architecturally, the models are quite different. Logistic regression (91.6% accurate; tuned with hyperdrive) effectively makes use of a fitted logistic function to carry out binary classification. The voting ensemble classifier (91.7% accurate; selected via autoML) makes use of a number of classifiers and, in this case, averages the class probabilities of each classifier to make a prediction.


## Future work

With HyperDrive, we can increase the maximum total runs allowing us to go through more hyperparameter options. Also, with AutoML, running for much longer time allows us to go through more models supported by AutoML for classification. Trying more models will help us find the best model for the problem in hand.  Using different parameter sampling techniques and tuning the arguments of the BanditPolicy can also prove beneficial.

## Proof of cluster clean up
https://github.com/NikitaMahajan19/MachineLearning/blob/master/cluster.JPG

