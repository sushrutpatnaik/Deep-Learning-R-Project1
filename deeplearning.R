#df <- read.csv(file="C:/DataScience_R/reviews/reviews.csv", header=TRUE, sep=",")
library(keras)
library(tidyverse)
text_reviews <- read_csv("http://utdallas.edu/~sxp175331/project1R/reviews.csv") %>%
  mutate(Liked = ifelse(reviews.rating == 5, 1, 0),
         text = paste(`reviews.text`),
         text = gsub("NA", "", text))

text <- text_reviews$text

max_features <- 1000
tokenizer <- text_tokenizer(num_words = max_features)

tokenizer %>% 
  fit_text_tokenizer(text)

tokenizer$document_count

text_seqs <- texts_to_sequences(tokenizer, text)

text_seqs %>%
  head()


# Set parameters:
maxlen <- 100
batch_size <- 32
embedding_dims <- 50
filters <- 64
kernel_size <- 3
hidden_dims <- 50
epochs <- 5

x <- text_seqs %>%
  pad_sequences(maxlen = maxlen)
dim(x)

y <- text_reviews$Liked
length(y)

require(caTools)
set.seed(25) 
sample = sample.split(y, SplitRatio = .75)

x_train = subset(x, sample == TRUE)
x_test  = subset(x, sample == FALSE)
y_train = subset(y, sample == TRUE)
y_test = subset(y, sample == FALSE)

model <- keras_model_sequential() %>% 
  layer_embedding(max_features, embedding_dims, input_length = maxlen) %>%
  layer_dropout(0.2) %>%
  layer_conv_1d(
    filters, kernel_size, 
    padding = "valid", activation = "relu", strides = 1
  ) %>%
  layer_global_max_pooling_1d() %>%
  layer_dense(hidden_dims) %>%
  layer_dropout(0.2) %>%
  layer_activation("relu") %>%
  layer_dense(1) %>%
  layer_activation("sigmoid") %>% compile(
    loss = "binary_crossentropy",
    optimizer = "adam",
    metrics = "accuracy"
  )

hist <- model %>%
  fit(
    x_train,
    y_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_split = 0.3
  )


results <- model %>% evaluate(x_test, y_test)
