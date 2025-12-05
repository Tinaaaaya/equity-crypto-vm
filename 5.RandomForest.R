#==================
# Train a Random Forest Model
#==================
tokens <- tokenize_words(df$Message, lowercase = TRUE)
tokens_flat <- sapply(tokens, paste, collapse = " ")
vocab <- create_vocabulary(itoken(tokens_flat))
vectorizer <- vocab_vectorizer(vocab)
dtm <- create_dtm(itoken(tokens_flat), vectorizer)
dtm <- as.matrix(dtm)

set.seed(123)
train_index <- createDataPartition(df$sentiment, p = 0.8, list = FALSE)

X_train <- dtm[train_index, ]
X_test  <- dtm[-train_index, ]
y_train <- df$sentiment[train_index]
y_test  <- df$sentiment[-train_index]

# Train Random Forest model
rf_model <- randomForest(x = X_train, y = y_train, ntree = 300)

# Predict using the trained Random Forest model
pred_rf <- predict(rf_model, X_test)

# Evaluate Random Forest model using confusion matrix
confusionMatrix(pred_rf, y_test)  # Confusion matrix for evaluation
