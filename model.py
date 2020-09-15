import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import ast
import random
import numpy as np


# ------------ CONFIG ------------
EPOCHS = 2
BATCH_SIZE = 32
# --------------------------------


# Load tokenized data
print("Load data ...")
embeddings = pd.read_csv("embeddings_subset.csv")
attention_masks = pd.read_csv("attentionmasks_subset.csv")

# Create tensors from loaded data
print("Create tensors ...")
train_inputs = []
train_labels = []
train_mask = []
for row in embeddings.itertuples():
    train_inputs.append(ast.literal_eval(row.Tokens))
    train_labels.append(row.Gender)
for row in attention_masks.itertuples():
    train_mask.append(ast.literal_eval(row.att_mask))
train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
train_mask = torch.tensor(train_mask)
print(train_mask.shape)

# Create TensorDataset/Dataloader for faster training
print("Create DataLoader ...")
train_data = TensorDataset(train_inputs, train_labels, train_mask)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

# Create model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False
)
optimizer = AdamW(model.parameters(),
                  lr=2e-5,
                  eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=len(train_dataloader) * EPOCHS)

# Train model
seed_val = 42
random.seed(seed_val)
# np.random_seed(seed_val)
torch.manual_seed(seed_val)
loss_values = []
for epoch_i in range(EPOCHS):
    print("-------- Epoch {} / {} ".format(epoch_i + 1, EPOCHS))
    print("Training ...")

    total_loss = 0
    model.train()
    for step, batch in enumerate(train_dataloader):
        if step % 100 == 0 and step > 0:
            print("Batch {:>5,} of {:>5,}.".format(step, len(train_dataloader)))
        model.zero_grad()
        # batch[0]: ids, [1]:masks, [2]:labels
        outputs = model(batch[0],
                        token_type_ids=None,
                        labels=batch[1],
                        attention_mask=batch[2])
        loss = outputs[0]
        total_loss += loss.items()
        loss.backward()
        # Clip norm of gradients to 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    avg_train_loss = total_loss / len(train_dataloader)
    loss_values.append(avg_train_loss)
    print("Average train loss: {0:.2f}".format(avg_train_loss))
print("Training complete!")