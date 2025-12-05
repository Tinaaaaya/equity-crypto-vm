# Financial Sentiment Analysis with Machine Learning & Deep Learning

This project documents my coding skills in R through a sentiment-analysis task using financial social-media comments.  
The raw data and sentiment coding come from the replication package of:

> Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014).  
> *Good debt or bad debt: Detecting semantic orientations in economic texts.*  
> Journal of the Association for Information Science and Technology, 65(4), 782–796.

For demonstration, only the first 500 observations are used.

The project implements **logistic regression**, **random forest**, and **LSTM (long short-term memory)** models to classify comments into **negative**, **neutral**, and **positive** sentiment categories.  
Model performance is compared using confusion matrices on a held-out testing set.

---

## 1. Package Loading

Libraries for importing data, cleaning data, plotting, and building machine-learning models.

```r
library(readr)        # Data import
library(dplyr)        # Data manipulation
library(ggplot2)      # Plotting
library(stringr)      # String manipulation
library(text2vec)     # Text processing and vectorization
library(tokenizers)   # Tokenization
library(tm)           # Text mining
library(randomForest) # Random forest classifier
library(h2o)          # For LSTM model
library(gridExtra)    # For plot arrangement
