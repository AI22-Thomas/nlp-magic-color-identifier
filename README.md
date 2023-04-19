# Magic Color Identifier

## Setup

All requirements are listed in `requirements.txt`. Used python version is 3.10.5.
The nklt library sometimes requires extra data, which is downloaded automatically when needed.

## Data
The original data is under `data/orcale-cards-oracle-cards-20230417090329.json`.
Original source is Scryfall Bulk Data: https://scryfall.com/docs/api/bulk-data, which contains all Magic: The Gathering cards
at the given time of download. The downloaded file has a timestamp in its name.

The preprocessed data is under `data/cards.json`.

## Preprocessing

All preprocessing is done in the `preprocessing` folder, in the `preprocessing.py` file.
It transforms the Scryfall bulk data into a formatted json file `cards.json` which is used for training and testing.

Further preprocessing is done at each experiment, specific to the needs of the experiment.

General preprocessing steps are:
- Remove all "non-standard" cards
  - This includes cards that are intentionally miss-desgined like "funny" card sets, Land cards as they do not have a color, Tokens, etc.
- Replacing special symbols of magic with english equivalents
  - Example: `{T}` is replaced with `tap`
  - Important note: color symbols are removed entirely, as the model could use them to identify the color of the card directly ("cheating")
- Tokenizing special leftover symbols
  - Example: `+1/+1` is wrapped to be `<+1/+1>` so that it represents a single token
- Cleanup of the color arrays to better representations
  - Example: `["W"]` is replaced with `[1, 0, 0, 0, 0, 0]`
  - Example: `["W", "U"]` is replaced with `[1, 1, 0, 0, 0, 0]`
  - Colors are: `["W", "U", "B", "R", "G", "C"]`, in order: white, blue, black, red, green, colorless
- Also includes a stopword cleanup function, but isn't used in the general preprocessing steps

## Experiments
Each experiment is standalone and they do not need to be executed in any specific order.
They are grouped into folders, one for each library used.

### Keras

#### bert-monocolor
Intended to train a fine-tuned bert model which classifies cards with only one color.  
The training took a long time and we could not get satifsfying results with shorter trainings. Because of this no results are recorded.  
The file is still included in this repository for documentation.

#### keras-multicolor
Creates a keras model consisting of a embedding layer, LSTM layer, and a dense output layer.  
The embedding layer uses pre-trained word embeddings from [GloVe](https://nlp.stanford.edu/projects/glove/).  
The word embedding file is stored as a split zip file in the `data` folder and needs to be unzipped before starting the experiment.  
The output of the trained model is a vector of size 5 where each value of the vector gives the likelyhood for the 5 possible card colors (white, blue, black, red, green).

#### keras-multicolor-tf-idf
Similar to keras-multicolor, but uses a TF-IDF vectorizer instead of a word embedding layer, and a different 
architecture of the neural network.
Best model is saved under `models/tf_idf_keras_best_model.h5`

### Torch

#### torch-bert-multicolor
Creates a fine-tune of the pretrained 'bert-base-cased' model from [transformers.BertModel](https://huggingface.co/transformers/v4.8.2/model_doc/bert.html?highlight=berttokenizer#bertmodel).  
The checkpoint with the best loss during training is saved to the `models` folder.
The output of the trained model is a vector of size 5 where each value of the vector gives the likelyhood for the 5 possible card colors (white, blue, black, red, green).

The file `torch-bert-multicolor-results.py` can be used to just run the finished model under `models/best-checkpoint.ckpt` and print the results.

### scikit-learn

#### tf-idf-mono-color
First try at a TF-IDF model with logistic regression. Uses stopword preprocessing from `scryfall_preprocessor.py`.
Only includes cards with one single color, and uses the color directly as class label.

#### Multinomial Naive Bayes (NB) and Logistic Regression (LR)
Uses libraries nltk and sklearn. \
Uses vectorizers Bag of Words (BOW) and Term Frequency - Inverse Document Frequency (TF-IDF). \
Comparison between NB with BOW, NB with TF-IDF, LR with BOW, LR with TF-IDF. \
Running nb-lr_bow_tfidf.py creates four confusion matrix plots.
