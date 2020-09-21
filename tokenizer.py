import pandas as pd
import logging
from transformers import BertTokenizer

cnt_oversized = 0


# ------------ config example values ------------
# file_data = "data_subset.csv"
# returnDF = True # should the function return a pandas dataframe object?
# max_tokencount = 510 # threshold in [1,510] (BERT is limited to 512 including CLS and SEP) 
# truncating_method = "head" # how to truncate list of tokens in ["head", "tail", "headtail"]
# file_results = "tokens_subset.csv" # None = dont save this dataframe as csv
# -----------------------------------------------


def preparate(file_data, returnDF, max_tokencount=510, truncating_method="head", file_results=None):

    # Load the data
    # TODO: Chunking necessary?
    data = pd.read_csv(file_data)

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

    # tokenize, truncate oversized data (BERT is limited to 512 tokens) and apply padding
    logging.disable(logging.WARNING)
    print("Applying tokenizer ...")
    data["ReviewText"] = data["ReviewText"].map(lambda x: tokenize(x, tokenizer, max_tokencount, selector[truncating_method]))
    print("Done!")
    global cnt_oversized
    print("{} reviews ({:.2%}) were oversized and truncated".format(cnt_oversized, cnt_oversized/len(data)))

    # padding and attention mask
    data["att_mask"] = data["ReviewText"].map(lambda x: am(x, max_tokencount))

    # return resulting data
    if file_results != None:
        data.to_csv(file_results, index=False)
    if returnDF:
        return data


def tokenize(text, tokenizer, max_tokencount, truncating_method):
    tokens = tokenizer.encode(text, add_special_tokens=True)
    if len(tokens) > max_tokencount:
        global cnt_oversized
        cnt_oversized += 1
        tokens = truncating_method(tokens)
    tokens += [0] * (max_tokencount - len(tokens) + 2) # this represents the padding
    return tokens


def am(tokens, max_tokencount):
    return [int(token > 0) for token in tokens]