### Project: Classify Emotions in Twitter messages

### Highlights:
  - This is a **multi-class text classification** problem.
  - The goal of this project is to **classify ISEAR Emotion Dataset into 7 classes**.
  - This model was built with **CNN and Word Embeddings** on **Tensorflow**.
  - Accuracy on test set: 62.5%

### Data: [ISEAR dataset] 
### Source: http://www.affective-sciences.org/home/research/materials-and-online-research/research-material
  - Input: **Tweets**
  - Output: **Emotions**
  - Examples:

    Emotion | Tweet
    --------|------------------------------------------------------------------------
    joy     | On days when I feel close to my partner and other friends.<br>When I feel at peace with myself and also experience a<br>close contact with people whom I regard greatly.
    --------|-----------------------------------------------------------
    fear    | Every time I imagine that someone I love or I could contact <br>a serious illness, even death.
    --------|-----------------------------------------------------------
    sadness | When I think about the short time that we live and relate<br>it to the periods of my life when I think that I did not<br>use this short time.

### Parameters:
  - batch_size
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
  - Google Pre-trained Word2Vec
  - Source: https://code.google.com/archive/p/word2vec/
  - Download: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit

### Train:
  - Command: python train.py
  
### Reference:
 - [Implement a cnn for text classification in tensorflow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
# Multiclass-CNN
