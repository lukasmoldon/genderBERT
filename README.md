
# genderBERT



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
    <td><a href="https://www.isca-speech.org/archive/archive_papers/interspeech_2014/i14_2719.pdf">VCDNN</a></td>
    <td>0.751</td>
    <td>0.770</td>
    <td>0.761</td>
    <td>0.823</td>
    <td>0.834</td>
    <td>0.831</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/bert-base-uncased">BERT</a> (1M)</td>
    <td>0.747</td>
    <td>0.770</td>
    <td>0.759</td>
    <td>0.824</td>
    <td>0.841</td>
    <td>0.833</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/bert-base-uncased">BERT</a>/F (1M)</td>
    <td>0.708</td>
    <td>0.669</td>
    <td>0.689</td>
    <td>0.769</td>
    <td>0.732</td>
    <td>0.752</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/albert-base-v1">BERT</a> (1M)</td>
    <td>0.750</td>
    <td>0.767</td>
    <td>0.759</td>
    <td>0.823</td>
    <td>0.837</td>
    <td>0.830</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/gpt2">GPT2</a> (1M)</td>
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
    <td><a href="https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf">HAN</a></td>
    <td>0.640</td>
    <td>0.642</td>
    <td>0.641</td>
    <td>0.735</td>
    <td>0.719</td>
    <td>0.727</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/bert-base-uncased">BERT</a> (2e-5)</td>
    <td>0.656</td>
    <td>0.642</td>
    <td>0.649</td>
    <td>0.743</td>
    <td>0.725</td>
    <td>0.734</td>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/roberta-base">roBERTa</a> (2e-5)</td>
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
    <td><a href="https://huggingface.co/bert-base-uncased">BERT</a></td>
    <td>0.???</td>
    <td>0.???</td>
    <td>0.???</td>
    <td>0.???</td>
    <td>0.???</td>
    <td>0.???</td>
  </tr>
</tbody>
</table>
