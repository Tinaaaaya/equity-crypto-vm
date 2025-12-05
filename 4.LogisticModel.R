#==================
# Data Splitting for Training and Testing
#==================
set.seed(123)
train_index <- createDataPartition(df$sentiment, p = 0.8, list = FALSE)  # 80% for training
train <- df[train_index, ]
test  <- df[-train_index, ]

#==================
# Text Vectorization using text2vec
#==================
# Tokenization and vectorization for the training data
tokenizer <- itoken(train$Message, preprocessor = tolower, tokenizer = text2vec::word_tokenizer, progressbar = FALSE)
vocab <- create_vocabulary(tokenizer) %>%
  prune_vocabulary(term_count_min = 5)   # Prune terms that appear less than 5 times
vectorizer <- vocab_vectorizer(vocab)

dtm_train <- create_dtm(tokenizer, vectorizer)
tfidf <- TfIdf$new()                    # Apply TF-IDF weighting
dtm_train_tfidf <- tfidf$fit_transform(dtm_train)

#==================
# Train a Baseline Model (Logistic Regression)
#==================
# Train a multinomial logistic regression model as a baseline
model_lr <- train(x = as.matrix(dtm_train_tfidf), y = train$sentiment, method = "multinom")

#==================
# Test the Logistic Regression Model
#==================
# Tokenize the test data
tokenizer_test <- itoken(test$Message, preprocessor = tolower, tokenizer = word_tokenizer, progressbar = FALSE)
dtm_test <- create_dtm(tokenizer_test, vectorizer)
dtm_test_tfidf <- tfidf$transform(dtm_test)  # Apply the same TF-IDF transformation to the test set

# Predict using the trained logistic regression model
pred <- predict(model_lr, as.matrix(dtm_test_tfidf))

# Evaluate the performance using confusion matrix
confusionMatrix(pred, test$sentiment)  # Confusion matrix for evaluation
