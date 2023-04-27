# Natural Language Processing - Sentiment Analysis of Amazon Reviews

## Authors
**Federico Cimini** (CIS 5190), **Liang-Yun Cheng** (CIS 5190), **Samuel Thudium** (CIS 5190)

## Abstract

![amazon logo](https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg)


Sentiment analysis is a popular introduction to natural language processing (NLP). This problem can be broken down into three simple steps: (1) clean input text, (2) learn embeddings for the words in present data, and (3) use these embeddings to classify the overall sentiment of new input text. We sought to build and improve upon the existing corpus of sentiment analysis notebooks that are prevalent on sites such as Kaggle. The sheer volume of these notebooks suggests that there are many ways to approach this problem, but typically there is no systematic comparison of the performance that these various methodologies have on a common dataset. 

Thus, in this project we present a side-by-side performance of NLP and classification models on over 500,000 Amazon food reviews. Namely, we compare (1) NLP models with increasing fidelity to language structure, which are coupled with (2) classification models of increasing complexity. Overall, we were able to achieve quality results even using simpler models (TF-IDF + Logistic Regression), but saw an expected improvement using more complex models that take sequence information into account (word2vec + LSTM). This report aims to improve and expand upon the existing space of sentiment analysis projects. 

## Repository Guide

The main results and analysis of this project can be found in `Full_Report.pdf`

An abriged version of the final report up to 6 pages long can be found in `Abriged_Report.pdf`

### Notebooks
To replicate this analysis the following files in the folder `notebooks` must be run in the following order:
1. Data Processing, EDA, and baseline model tuning: `00_Project2Milestone_+_TF_IDF_hyperparam_tuning.ipynb`
2. S3 File Access to cleaned data: `01_s3-dataAccess.ipynb`
3. Helper functions for plotting and model evaluation: `02_helper_module.py`
4. TF-IDF + Logistic Regression Tuned Model: `03_ST_tfidf-logreg.ipynb`
5. TF-IDF + Multinomial Naive Bayes Model: `04_ST_tfidf-MNB.ipynb`
6. Word2Vec + Logistic Regression Tuned Model: `05_Word2Vec+LogReg.ipynb`
7. Word2Vec + LSTM tuned Model: `06_Word2Vec+LSTM.ipynb`
8. BERT Base Cased Model: `07_BERT_base_cased.ipynb`
9. Replica of data preprocessing with additonal EDA included: `08_DataCleaning+Visualization_Milestone3.ipynb`

### Data Files
In the folder titled ``data`` the following files can be found:
1. ``Model_results.csv``: Summary of main models metrics.
