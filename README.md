
# genderBERT
Deep learning model for gender classification on texts using pretrained BERT models for the paper '' **(TODO)**. Our trained model with the highest accuracy (see section 'Results') can be downloaded here **(TODO)**.

## Used pretrained models

**BERT (Bidirectional Encoder Representations from Transformers)** ([source](https://huggingface.co/bert-base-uncased), [paper](https://arxiv.org/pdf/1810.04805.pdf)): Transformers model pretrained on a large corpus of English data with self-supervised learning. We use the base version, which provides 12 layers with 110M parameters and uncased version, which does not distinguish between lower and upper case letters (to reduce complexity and runtime, see this [overview](https://huggingface.co/transformers/pretrained_models.html) for all pretrained versions).

**alBERT (A Lite BERT)** ([source](https://huggingface.co/albert-base-v1), [paper](https://arxiv.org/pdf/1909.11942.pdf)): Lite version of BERT using two different parameter reduction techniques resulting in 18x fewer parameters and about 1.7x faster training speed

**roBERTa (Robustly optimized BERT approach)** ([source](https://huggingface.co/roberta-base), [paper](https://arxiv.org/pdf/1907.11692.pdf)): Further trained BERT model on additional data, with longer sequences and bigger batches, adding dynamical masking patterns and removing BERT's next sentence prediction objective.

**GPT-2 (Generative Pre-trained Transformer 2)** ([source](https://huggingface.co/gpt2), [paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf))

**VCDNN (Variable-Component Deep Neural Network)** ([paper](https://www.isca-speech.org/archive/archive_papers/interspeech_2014/i14_2719.pdf)): for robust speech recognition

**HAN (Hierarchical Attention Networks)** ([paper](https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf)): for document classification



## Used data sets for training & testing

**Amazon** ([source](http://jmcauley.ucsd.edu/data/amazon/)): contains product reviews from Amazon

**StackExchange** ([source](https://archive.org/details/stackexchange)): contains content posted on StackExchange

**Reddit** ([source](https://files.pushshift.io/reddit/)): contains comments on Reddit


## Code
:arrow_forward: For more detailed information, use the corresponding link to the [docsring](https://www.python.org/dev/peps/pep-0257/) at the end of each descirption.
* **tokenizer.py** - Prepares the data for the main model. The given data set gets tokenized with applied padding and oversized texts get truncated. It stores the resulting tokenized texts and the corresponding attention mask. ([docstring](https://github.com/lukasmoldon/genderBERT/blob/master/tokenizer.py#L23-L51))
* **model.py** - Main model file for training, validating and testing. Uses a given pretrained model and a tokenized data set and does training/validation/testing as specified in the given mode of the config.json file. ([docstring](https://github.com/lukasmoldon/genderBERT/blob/master/model.py#L44-L82))
* **majority_voting.py** - Computes the majority voting for a given prediction of a BERT model and displays [accuracy](https://en.wikipedia.org/wiki/Accuracy_and_precision) and [F1](https://en.wikipedia.org/wiki/F1_score) after applying majority voting. Whenever a user with multiple texts but different predictions on gender has a majority for one gender, all minority predictions get changed to the majority prediction. The function does not change predictions for users with no predicted majority for one gender (50/50 case). ([docstring](https://github.com/lukasmoldon/genderBERT/blob/master/majority_voting.py#L9-L22))
* **customBERT.py** - Additional (failed) approach, where BERT gets extended by 3 adjustable layers (e.g. linear). All attempts resulted in an accuracy below 0.75.
* **run_cluster.py** - Script for training the same model with different learning rates, maximal tokencounts and truncating methods consecutively.
* **config.json** - Collection (type dict) of possible setups for model.py. ID of the setup is used in model.py for extracting options.
* **bert-base-uncased-vocab.txt** - Vocab map for the pretrained BERT model [bert-base-uncased](https://huggingface.co/bert-base-uncased) to convert words to token IDs ([source](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt)) (the line of the file represents the ID starting at 0 for the [PAD] flag, see [BERT Tokenizer](https://huggingface.co/transformers/v1.2.0/_modules/pytorch_transformers/tokenization_bert.html))
* **data_subset.csv** - Sample of Amazon training data (10k raw elements) for testing general code functionality on local machines.

## How to use
* **1.**
* **2.**
* **3.**

## Results on data

### Amazon data
<table>
<thead>
  <tr>
    <th></th>
    <th colspan="3">Without majority voting</th>
    <th colspan="3">With majority voting</th>
  </tr>
</thead>
<tbody>
  <tr>
    <th>Model</th>
    <th>Male F1</th>
    <th>Female F1</th>
    <th>Accuracy</th>
    <th>Male F1</th>
    <th>Female F1</th>
    <th>Accuracy</th>
  </tr>
  <tr>
    <td>VCDNN</td>
    <td>0.751</td>
    <td>0.770</td>
    <td>0.761</td>
    <td>0.823</td>
    <td>0.834</td>
    <td>0.831</td>
  </tr>
  <tr>
    <td>BERT (1M)</td>
    <td>0.747</td>
    <td>0.770</td>
    <td>0.759</td>
    <td>0.824</td>
    <td>0.841</td>
    <td>0.833</td>
  </tr>
  <tr>
    <td>BERT/F (1M)</td>
    <td>0.708</td>
    <td>0.669</td>
    <td>0.689</td>
    <td>0.769</td>
    <td>0.732</td>
    <td>0.752</td>
  </tr>
  <tr>
    <td>alBERT (1M)</td>
    <td>0.750</td>
    <td>0.767</td>
    <td>0.759</td>
    <td>0.823</td>
    <td>0.837</td>
    <td>0.830</td>
  </tr>
  <tr>
    <td>GPT2 (1M)</td>
    <td>0.750</td>
    <td>0.767</td>
    <td>0.759</td>
    <td>0.823</td>
    <td>0.837</td>
    <td>0.830</td>
  </tr>
</tbody>
</table>

### StackOverflow data
<table>
<thead>
  <tr>
    <th></th>
    <th colspan="3">Without majority voting</th>
    <th colspan="3">With majority voting</th>
  </tr>
</thead>
<tbody>
  <tr>
    <th>Model</th>
    <th>Male F1</th>
    <th>Female F1</th>
    <th>Accuracy</th>
    <th>Male F1</th>
    <th>Female F1</th>
    <th>Accuracy</th>
  </tr>
  <tr>
    <td>HAN</td>
    <td>0.640</td>
    <td>0.642</td>
    <td>0.641</td>
    <td>0.735</td>
    <td>0.719</td>
    <td>0.727</td>
  </tr>
  <tr>
    <td>BERT (2e-5)</td>
    <td>0.656</td>
    <td>0.642</td>
    <td>0.649</td>
    <td>0.743</td>
    <td>0.725</td>
    <td>0.734</td>
  </tr>
  <tr>
    <td>roBERTa (2e-5)</td>
    <td>0.658</td>
    <td>0.653</td>
    <td>0.655</td>
    <td>0.724</td>
    <td>0.710</td>
    <td>0.717</td>
  </tr>
</tbody>
</table>

### Reddit data
<table>
<thead>
  <tr>
    <th></th>
    <th colspan="3">Without majority voting</th>
    <th colspan="3">With majority voting</th>
  </tr>
</thead>
<tbody>
  <tr>
    <th>Model</th>
    <th>Male F1</th>
    <th>Female F1</th>
    <th>Accuracy</th>
    <th>Male F1</th>
    <th>Female F1</th>
    <th>Accuracy</th>
  </tr>
  <tr>
    <td>BERT</td>
    <td>0.???</td>
    <td>0.???</td>
    <td>0.???</td>
    <td>0.???</td>
    <td>0.???</td>
    <td>0.???</td>
  </tr>
</tbody>
</table>
