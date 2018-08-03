### Project: Classify Emotions in Twitter messages

### Highlights:
  - This is a **multi-class text classification** problem.
  - The goal of this project is to **classify ISEAR Emotion Dataset into 7 classes**.
  - This model was built with **CNN and Word Embeddings** on **Tensorflow**.

### Results:

  - Table of model accuracies:

    Model                   | Accuracy
    ------------------------|----------
    CNN w/o Word Embeddings | 57.4%
    CNN with Word2Vec       | 62.5%
    CNN with GloVe          | 64.0%

### Confusion Matrix:

[[http://github.com/Emotion-Multiclass-CNN/confusion_matrix.png|alt=octocat]]


### Data: ISEAR Emotion Dataset<br> (http://www.affective-sciences.org/home/research/materials-and-online-research/research-material)
  - Input: **Emotion rich tweets**
  - Output: **Emotions**
  - Examples:

    Emotion | Tweet
    --------|------------------------------------------------------------------------------------------------------
    joy     | On days when I feel close to my partner and other friends. When I feel at peace with myself and also <br>experience a close contact with people whom I regard greatly.
    fear    | Every time I imagine that someone I love or I could contact a serious illness, even death.
    sadness | When I think about the short time that we live and relate it to the periods of my life when I think <br> that I did not use this short time.
    disgust | At a gathering I found myself involuntarily sitting next to two people who expressed opinions that <br> I considered very low and discriminating.
    shame   | When I realized that I was directing the feelings of discontent with myself at my partner and this <br> way was trying to put the blame on him instead of sorting out my own feeliings.
    guilt   | I feel guilty when when I realize that I consider material things more important than caring for <br> my relatives. I feel very self-centered.


### Parameters:
  - batch_size:
  - dropout_keep_prob:
  - embedding_dim:
  - evaluate_every:
  - filter_sizes:
  - hidden_unit:
  - l2_reg_lambda:
  - max_pool_size:
  - non_static:
  - num_epochs:
  - num_filters:

### Requirements:
  ##Google Pre-trained News Word2Vec
  - Source: https://code.google.com/archive/p/word2vec/
  - Download: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit

  ##Stanford Pre-trained GloVe Twitter 27B
  - Source: https://nlp.stanford.edu/projects/glove/
  - Download: http://nlp.stanford.edu/data/glove.twitter.27B.zip

### Train:
  - Command: python train.py
  
### Reference:
 - [Implement a cnn for text classification in tensorflow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
# Multiclass-CNN
