# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

1. Explain the purpose of the analysis.
    
    * The purpose of analysis is to build a model that can identify creditworthiness of borrower.

2. Explain what financial information the data was on, and what you needed to predict.

    * The data used to predict were loan size, interest rate, borrower's income, borrower's debt to income ratio, number of accounts, number of derogatory marks, and total debt to predict whether loans were healthy or risky.

3.  Provide basic information about the variables you were trying to predict (e.g., `value_counts`).

    * A value of 0 in the “loan_status” column means that the loan is healthy. A value of 1 means that the loan has a high risk of defaulting. There were 75,036 healthy loans and 2,500 loans with high risk of defaulting.

4. Describe the stages of the machine learning process you went through as part of this analysis.
    1. Create dataframe for features and targets:     
        * First we separated the data into features and target. X represented features which analyzes "loan_size", "interest_rate", "borrower_income", "debt_to_income", "num_of_accounts", "derogatory_marks", and "total_debt". y represents the target which is "loan_status". 
        
    2. Split training and testing sets:
        * Then we used train_test_split to separate the data into training features (X_train), testing features (X_test), training targets (y_train), testing targets (y_test).
    
    3. Create Logistic Regression Model

5. Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).
    1. Fit a logistic regression model by using training data (X_train and y_train)
    2. Save predictions on testing data labels by using testing feature data (X_test) and the fitted model
    3. Evaluate model's performance (y_test and y_pred)
    
Given small number of high-risk loans, repeat steps using resampled training data. 

    1. Use the RandomOverSampler module from the imbalanced-learn library to resample the data.
    2. Use the LogisticRegression classifier and the resampled data to fit the model and make predictions. 
    3. Evaluate the re-sampled model’s performance.


## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all six machine learning models.

###  Machine Learning Model 1:
Description of Model 1 Accuracy, Precision, and Recall scores.
    
   * Accuracy: Approximately 95% of the loans in the test data were accurately categorized by the model. However, based on `value_counts`, there were very few loans in the data that were considered high risk relative to healthy loans, and so our model could have had high accuracy by simply predicting all loans to be healthy
      
   * Precision: The precision is very high for the 0 class (1.00) and slighlty lower for the 1 cass (0.85). 0 represents healthy loans and 1 represents loans with risk of default. Prediction for healthy loans were much higher for imbalanced data.
      
   * Recall: By accurately identifying 91% of loans that are at high risk of default the model did a fairly good job


### Machine Learning Model 2:
Description of Model 2 Accuracy, Precision, and Recall scores.
  
  * Accuracy: Approximately 99% of the loans in the resampled test data were accurately categorized by the model. 
      
  * Precision: The precision is very high for the 0 class (1.00) and slighlty lower for the 1 cass (0.84). 0 represents healthy loans and 1 represents loans with risk of default. Prediction for risky loans were slighly lower compared to imbalanced data.
      
  * Recall: By accurately identifying 99% of loans that are at high risk of default the model does an accurate job of predicting
      
     
## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?

* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

* If you do not recommend any of the models, please justify your reasoning.

    * Machine Learning Model 2 which uses resampled data to balance dataset does a more accurate job on Accuracy and Recall however slighly lower for Recall for risky loans.  
       
    * Performance depends on the accuracy of 1 values as we are trying to predict the accuracy of loans that are at risk of default. Oversampling artifically increases number of instances in the minority class (1). This comes at the expense of tending to overestimate the frequency of 1 values to have slightly lower precision.

    * The accuracy score is much higher for the resampled data (0.99 vs 0.95), meaning that the model using resampled data was much better at detecting true positives and true negatives. 

    * The precision for the minority class is slightly higher with the orignal data (0.85) versus the resampled data (0.84) meaning that the original data was better at detecting loans that were actually going to default. 

    * In terms of the recall, however, the minority class metric using resampled data was much better (0.99 vs 0.91). Meaning that the resampled data correctly clasified a higher percentage of the truly defaulting borrowers. 

    * Overall, the model using resampled data was much better at detecting loans at high risk of default. Model 2 which uses oversampled data to balance dataset, is recommended as it provided higher accuracy and recall. 
