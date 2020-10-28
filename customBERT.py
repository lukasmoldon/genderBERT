# Source:
# https://stackoverflow.com/questions/64156202/add-dense-layer-on-top-of-huggingface-bert-model
# https://towardsdatascience.com/identifying-hate-speech-with-bert-and-cnn-b7aa2cddd60d


import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel
from transformers import AutoTokenizer


class CustomBERTModel(nn.Module):

    def __init__(self):
          super(CustomBERTModel, self).__init__()
          self.bert = BertModel.from_pretrained("bert-base-uncased")
          # custom layers (3x linear) on top of bert:
          self.linear1 = nn.Linear(768, 256) # TODO: modify dimension of all 3 linear layers
          self.linear2 = nn.Linear(256, 128)
          self.linear3 = nn.Linear(128, 2)

    def forward(self, ids, mask):
          sequence_output, pooled_output = self.bert(ids, attention_mask=mask)

          # shape of sequence_output: (batch_size, sequence_length, 768)
          linear1_output = self.linear1(sequence_output[:,0,:].view(-1,768)) # extract the 1st token's embeddings
          linear2_output = self.linear2(linear1_output)
          linear3_output = self.linear3(linear2_output)

          return linear3_output


def train(input_ids, attention_mask, data_loader, epochs):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = CustomBERTModel()
    model.to(torch.device("cuda:0"))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    for epoch in epochs:
        for batch in data_loader: 

            data = batch[0]
            targets = batch[1] # assuming that data loader returns a tuple of data and its targets
            
            optimizer.zero_grad()   
            encoding = tokenizer.batch_encode_plus(data, return_tensors="pt", padding=True, truncation=True, max_length=64, add_special_tokens=True)
            outputs = model(input_ids, attention_mask=attention_mask)
            outputs = F.log_softmax(outputs, dim=1) 
            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    return model

