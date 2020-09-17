import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import ast
import random
import numpy as np
from sklearn.model_selection import train_test_split

# TODO: Add CUDA support

# ------------ CONFIG ------------
EPOCHS = 5
BATCH_SIZE = 4
# --------------------------------

# Helper function for accuracy
def flat_accuracy(predicitions, labels):
    pred_flat = np.argmax(predicitions, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# Load tokenized data
print("Load data ...")
embeddings = pd.read_csv("embeddings_subset.csv", nrows=1000)
attention_masks = pd.read_csv("attentionmasks_subset.csv",nrows=1000)
# Create tensors from loaded data and train/test/val-split
print("Create tensors ...")
inputs = []
labels = []
mask = []
for row in embeddings.itertuples():
    inputs.append(ast.literal_eval(row.Tokens))
    labels.append(row.Gender)
for row in attention_masks.itertuples():
    mask.append(ast.literal_eval(row.att_mask))

X_train ,X_other, y_train, y_other = train_test_split(inputs, labels, random_state=42, test_size=0.2)
X_test ,X_val, y_test, y_val = train_test_split(X_other, y_other, random_state=42, test_size=0.5)
train_mask ,other_mask, _, other_labels = train_test_split(mask, labels, random_state=42, test_size=0.2)
test_mask ,validation_mask, _, _ = train_test_split(other_mask, other_labels, test_size=0.5)

X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)
X_val =torch.tensor(X_val)

y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)
y_val = torch.tensor(y_val)

train_mask = torch.tensor(train_mask)
test_mask = torch.tensor(test_mask)
validation_mask = torch.tensor(validation_mask)

# Create TensorDataset/Dataloader for faster training
print("Create DataLoader ...")
train_data = TensorDataset(X_train, y_train, train_mask)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)

validation_data = TensorDataset(X_val, y_val, validation_mask)
validation_sampler = RandomSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=BATCH_SIZE)

# Create model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False
)
# TODO: Find good values
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
    print("-------- Epoch {} / {} --------".format(epoch_i + 1, EPOCHS))
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
        total_loss += loss.item()
        loss.backward()
        # Clip norm of gradients to 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    avg_train_loss = total_loss / len(train_dataloader)
    loss_values.append(avg_train_loss)
    print("Average train loss: {0:.2f}".format(avg_train_loss))

    # ---VALIDATION--- 
    print("Start validation ...")
    model.eval()
    tmp_eval_acc, steps = 0, 0
    for batch in validation_dataloader:
        b_inputs, b_labels, b_masks = batch
        with torch.no_grad():
            outputs = model(b_inputs,
                            token_type_ids=None,
                            attention_mask=b_masks)  

        _, predicted = torch.max(outputs[0].data, 1)
        tmp_eval_acc += (predicted == b_labels).sum().item()
        steps += b_labels.size(0)
    print("Accuracy: {0:.2f}".format(tmp_eval_acc/steps))
print("Training complete!")