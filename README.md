
# genderBERT

## Models

**BERT (Bidirectional Encoder Representations from Transformers)** ([source](https://huggingface.co/bert-base-uncased), [paper](https://arxiv.org/pdf/1810.04805.pdf)): Transformers model pretrained on a large corpus of English data with self-supervised learning

**alBERT (A Lite BERT)** ([source](https://huggingface.co/albert-base-v1), [paper](https://arxiv.org/pdf/1909.11942.pdf)): Lite version of BERT using two different parameter reduction techniques resulting in 18x fewer parameters and about 1.7x faster training speed

**roBERTa (Robustly optimized BERT approach)** ([source](https://huggingface.co/roberta-base), [paper](https://arxiv.org/pdf/1907.11692.pdf)): Further trained BERT model on additional data, with longer sequences and bigger batches, removing BERT's next sentence prediction objective and dynamical masking patterns.

**GPT-2 (Generative Pre-trained Transformer 2)** ([source](https://huggingface.co/gpt2), [paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf))

**VCDNN (Variable-Component Deep Neural Network)** ([paper](https://www.isca-speech.org/archive/archive_papers/interspeech_2014/i14_2719.pdf)): for robust speech recognition

**HAN (Hierarchical Attention Networks)** ([paper](https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf)): for document classification



## Data

[Amazon review data set](http://jmcauley.ucsd.edu/data/amazon/)

[Stackexchange posts](https://archive.org/details/stackexchange)

[Reddit comments data set](https://files.pushshift.io/reddit/)


## Code
TODO


## Statistics

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
