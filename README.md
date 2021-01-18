# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
**In 1-2 sentences, explain the problem statement: e.g "This dataset contains data about... we seek to predict..."**
This dataset contains marketing campaign data for a financial institution. The goal is to predict whether they will subscribe for term deposit. The data contains age, job, marital status, education, housing loan, personal loan, and poutcome which can be used train the model.

**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**
The best performing model was the automl model with ID AutoML_27526f0b-7b45-40f0-a412-281414049f7b_14 with an accuracy of 0.9172617023688088 and the algorithm used was VotingEnsemble. The hyperdrive run HD_22dae0c9-2f58-4634-a8f1-3a664aa97c93_10 that was created using hyperdrive parameters and Scikit-learn pipeline. 


## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

**Data**
The data is normalized and cleaned by updating the values in columns(independent variables) from words to binary 0/1 where ever applicable. This helps the the model to run efficiently. The dats is also being split into 80/20 so the model can be tested using predict objects.

**HyperParameter config**
**What are the benefits of the parameter sampler you chose?**
RandomParameterSampling with discrete values is being used. RandomParameterSampling is faster and supports early termination of low-performance runs. GridParameterSampling can be used when budget is not an issue to exhaustively search over the search space or BayesianParameterSampling to explore the hyperparameter space. 

C --> Regularization value. Smaller values have more regularization. max_iter --> Maximum number of iterations.
Below are my config settings:

ps = RandomParameterSampling( 
    {
        "--max_iter": choice(10,50,100,150,200)
        ,"--C": choice(0.001,0.01,0.1,1,1.25,1.5)
    }
)

**What are the benefits of the early stopping policy you chose?**
An early stopping policy is being used to automatically terminate poorly performing runs. This will help improve compute efficiency. Any run that doesn't fall within the slack factor of the evaluation metric with respect to the best performing run will be terminated. 
policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1)

hyperdrive_config = HyperDriveConfig(run_config=src,
    hyperparameter_sampling=ps,
    policy=policy,
    primary_metric_name="Accuracy",
    primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
    max_total_runs=100,
    max_concurrent_runs=4)



**Classification Algorithm:**
Since we are trying to determine if teh individual will subscribe for short term deposit, i.e. binary outcome, logistic regression is being used as teh classification algorithm.


## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

Below is the configuration for the AutoML run:

automl_config = AutoMLConfig(
    compute_target=compute_target,
    experiment_timeout_minutes=20,
    task='classification',
    primary_metric="accuracy",
    training_data=train_data,
    label_column_name="y",
    n_cross_validations=3
)

**Begin - Update based on review 1/18/2021 Round 2 Set 1**
**experiment_timeout_minutes=20**
The experiment will continue to run for 20 minutes and exit. Given teh amount of data 20 minute should be more than enough to generate a model.

**task=classification**
This defines the experiment type which in this case is classification(logistic regression).

**primary_metric="accuracy",**
The primary metric is set as accuracy.

**n_cross_validations=3**
One cross validation can cause an overfit hence it was set to 3. An average of the 3 validation metrics will be used.
**End - Update based on review 1/18/2021 Round 2 Set 1**

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**
Below are the metrics for both runs.

HyperDrive Model
id: HD_22dae0c9-2f58-4634-a8f1-3a664aa97c93_10
Accuracy: 0.9080424886191198
Max iterations:200
Regularization Strength:1.5

AutoML Model
id: AutoML_27526f0b-7b45-40f0-a412-281414049f7b_14
Accuracy: 0.9172617023688088
AUC_weighted: 0.9465394508134525
precision_score_weighted: 0.9125414684985289
Algortithm: VotingEnsemble

The difference in accuracy between the two models is not very drastic but AutoML model seems to be performing better than the HyperDrive model. AutoML is a better candidate here as most of teh training validations and and necessary calculations are being done by itself and we don't need to make any adjustments. Hyperdrive parameters will need repetitive adjustments and be re-ran until we get the best running model of various experiments.

**Begin - Update based on review 1/18/2021**
Automated machine learning picks an algorithm and hyperparameters and generates a model ready for deployment. Automl applies bayesian sampling for hyperparameters optimization.

VotingEnsemble combines the predictions from multiple learnings. It works by creating two or more standalone models from the training dataset. The voting classifier wraps the models and averages the predictions of the sub-models when asked to make predictions for new data.
**End - Update based on review 1/18/2021**

**Begin - Update based on review 1/18/2021 Round 2 Set 2**
**AutoML Model performance**
Accuracy is the ratio of predictions that exactly match the true class labels. AutoMl model was able to achieve an accuracy of 0.91
Accuracy: 0.9172617023688088 
AUC_weighted: 0.9465394508134525

As shown in the chart the weighted accuracy of the model is not close to the ideal score. This means the model needs more training and tweaking for it to perform better.
AUC_weighted: 0.9465394508134525
![ROC Curve](ROC_Curve.png?raw=true "ROC Curve")

A more acceptable model will have less number of false predictions. The aount of false predictions are 1278 and 914 which is almost 10% of the training data. The model needs to be in such a way that the number of false positives should be kept at minimum.
![Confusion Matrix](Confusion_Matrix.png?raw=true "Confusion Matrix")

The cumulative gains curve plots the percent of positive samples correctly classified as a function of the percent of samples considered where we consider samples in the order of predicted probability.
![Cumulative Curve](Cumulative_Curve.png?raw=true "Cumulative Curve")

Recall is the ability of a model to detect all positive samples and precision is the ability of a model to avoid labeling negative samples as positive. As displayed below the model does not 
precision_score_weighted: 0.9125414684985289
![Precision recall Curve](precisionrecall.png.png?raw=true "Precision recall Curve")

Balanced accuracy is the arithmetic mean of recall for each class. The balanced_accuracy: 0.7603071146063195 which is not closer to 1
balanced_accuracy: 0.7603071146063195,
Below are the numbers collected from the model:
AUC_macro: 0.9465394508134525,
recall_score_weighted: 0.9172617023688088,
average_precision_score_macro: 0.8193072002579888,
precision_score_weighted: 0.9125414684985289,
weighted_accuracy: 0.9561678575520739,
precision_score_macro: 0.7987069765267091,
precision_score_micro: 0.9172617023688088,
average_precision_score_micro: 0.9811384745525,
balanced_accuracy: 0.7603071146063195,
recall_score_macro: 0.7603071146063195,
AUC_micro: 0.9803563979199489,
recall_score_micro: 0.9172617023688088,
average_precision_score_weighted: 0.9541590738469147,
log_loss: 0.1947471160178461,
norm_macro_recall: 0.5206142292126388,
f1_score_macro: 0.7776140086133401,
AUC_weighted: 0.9465394508134525,
f1_score_micro: 0.9172617023688088,
matthews_correlation: 0.5576695325848257,
f1_score_weighted: 0.9144187410523813,
accuracy: 0.9172617023688088
 **End - Update based on review 1/18/2021 Round 2 Set 2**

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**
Cross-validation is the process of taking many subsets of the full training data and training a model on each subset. Higher accuracy can be achieved by higher number in n_cross_validations. I would also increase the experiment_timeout_minutes to help with increased computation required.


## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
compute_target.delete()
