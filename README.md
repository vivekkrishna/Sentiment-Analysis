# Sentiment-Analysis

This project is related to identifying the sentiment of the review based on the labelled imdb data corpus of the same.

## Data preprocessing

As part of Data pre-processing, we have extracted review text and removed certain neutral words and punctuation which may not have any sentiment associated with it and polarized the label to binary 0 or 1. we also used the variation of n-grams of 2, 3 etc. to extract features related to sequence of words.

To build features, we used count vectorizer and scipy sparse representations and had columns equivalent number of words in the entire training text corpus.

## Model

We used Stochastic Gradient Descent classifier for building the model against training features.

## References

http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
https://docs.scipy.org/doc/scipy/reference/sparse.html
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html

