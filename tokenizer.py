import pandas as pd
from transformers import BertTokenizer
import logging
import tensorflow as tf


# ------------ CONFIG ------------
path_data = "C:/Users/lukas/Documents/GitHub/genderBERT/data_subset.csv"
path_embeddings = "C:/Users/lukas/Documents/GitHub/genderBERT/embeddings_subset.csv"
path_attentionmasks = "C:/Users/lukas/Documents/GitHub/genderBERT/attentionmasks_subset.csv"
max_tokencount = 510 # threshold in [1,510] (BERT is limited to 512 including CLS and SEP) 
truncating_method = "head" # how to truncate list of tokens in ["head", "tail", "headtail"]
# --------------------------------


# Load the data
data = pd.read_csv(path_data)

# tag statistics
print("Total: {}".format(len(data)))
print("Male: {} ({:.2%})".format(len(data[data["Gender"] == 1]), len(data[data["Gender"] == 1])/len(data)))
print("Female: {} ({:.2%})".format(len(data[data["Gender"] == 0]), len(data[data["Gender"] == 0])/len(data)))

# load the tokenizer and selector for truncating oversized token lists
print("Loading BERT tokenizer ...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
max_tokencount = min(max_tokencount, 510)
selector = {
    "head": (lambda x: x[:max_tokencount+1] + x[-1:]),
    "tail": (lambda x: x[:1] + x[-max_tokencount-1:]),
    "headtail": (lambda x: x[:max_tokencount//4] + x[max_tokencount//4-max_tokencount:]) # adapted from https://arxiv.org/abs/1905.05583
}
print("Done!")

# tokenize and truncate oversized data (BERT is limited to 512 tokens)
cnt_oversized = 0
data_tokenized = pd.DataFrame(columns=["Gender", "Tokens"])
logging.disable(logging.WARNING)

print("Applying tokenizer ...")
for index, row in data.iterrows():
    tokens = tokenizer.encode(row["ReviewText"], add_special_tokens=True)
    if len(tokens) > max_tokencount:
        cnt_oversized += 1
        tokens = selector[truncating_method](tokens)
    data_tokenized = data_tokenized.append({"Gender": row["Gender"], "Tokens": tokens}, ignore_index=True)

print("Done!")
print("{} reviews ({:.2%}) were oversized and truncated".format(cnt_oversized, cnt_oversized/len(data_tokenized)))

# padding and attention mask
attention_masks = pd.DataFrame(columns=["att_mask"])
for index, row in data_tokenized.iterrows():
    row["Tokens"] = row["Tokens"] + [0] * (max_tokencount - len(row["Tokens"]) + 2)
    attention_masks = attention_masks.append({"att_mask": [int(token > 0) for token in row["Tokens"]]}, ignore_index=True)

# store data
data_tokenized.to_csv(path_embeddings, index=False)
attention_masks.to_csv(path_attentionmasks, index=False)