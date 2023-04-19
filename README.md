# Magic Color Identifier

## Data
To get the original data used for this project, unzip the zip file in the `data` folder.

## Experiments
Each experiment is standalone and they do not need to be executed in any specific order.

### bert-monocolor
Intended to train a fine-tuned bert model which classifies cards with only one color.  
The training took a long time and we could not get satifsfying results with shorter trainings. Because of this no results are recorded.  
The file is still included in this repository for documentation.

### keras-multicolor
Creates a keras model consisting of a embedding layer, LSTM layer, and a dense output layer.  
The embedding layer uses pre-trained word embeddings from [GloVe](https://nlp.stanford.edu/projects/glove/).  
The word embedding file is stored as a split zip file in the `data` folder and needs to be unzipped before starting the experiment.  
The output of the trained model is a vector of size 5 where each value of the vector gives the likelyhood for the 5 possible card colors (white, blue, black, red, green).

### torch-bert-multicolor
Creates a fine-tune of the pretrained 'bert-base-cased' model from [transformers.BertModel](https://huggingface.co/transformers/v4.8.2/model_doc/bert.html?highlight=berttokenizer#bertmodel).  
The checkpoint with the best loss during training is saved to the `checkpoints` folder.
The folder contains our best result split into multiple zip files. These files have to be unzipped in order to be used.  
The output of the trained model is a vector of size 5 where each value of the vector gives the likelyhood for the 5 possible card colors (white, blue, black, red, green).

### Multinomial Naive Bayes (NB) and Logistic Regression (LR)
Uses libraries nltk and sklearn
Uses vectorizers Bag of Words (BOW) and Term Frequency - Inverse Document Frequency (TF-IDF)
Comparison between NB with BOW, NB with TF-IDF, LR with BOW, LR with TF-IDF
Running nb-lr_bow_tfidf.py creates four confusion matrix plots 
