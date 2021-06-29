# Description

Detect disclosures of data breaches in SEC Filings

## Steps

1. data preparation
Randomly select samples for mannual labeling
Prepare unlabeled samples for label propagation through semi-supervised learning

2. Exploratory Data Analysis
Examine named entities  
Compare the following natural language features between disclosures of a data breach and other disclosures:  
  Named entities,  
  Most frequent words,  
  Length features,  
  Parts-of-speech tags,  
  Sentiment (polarity and subjectivity).  
Conduct Topic modeling and examine the top 20 words for top 10 topics

3. Model Training (1st time)
Compare among Logistic Regression, Tree-based Models (Random Forest and XGBoost), and SVC.
Choose Logistic Regression as the base estimator for label propagation through semi-supervised learning

4. Self Training
Label propagation through semi-supervised learning

5. Model Training (2nd time)
Compare between Logistic Regression and SVC using the propagated labels.
Fine tuning both models.

6. Model Predict
Apply the best model to the whole sample to detect disclosure of a data breach.
