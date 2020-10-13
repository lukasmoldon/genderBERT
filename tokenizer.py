import logging
import datetime
from datetime import date, timedelta
import pandas as pd
import logging
from transformers import AlbertTokenizer, BertTokenizer
from tokenizers import BertWordPieceTokenizer

cnt_oversized = 0

# ------------ config example values ------------
# file_data = "data_subset.csv"
# returnDF = True # should the function return a pandas dataframe object?
# max_tokencount = 510 # threshold in [1,510] (BERT is limited to 512 including CLS and SEP) 
# truncating_method = "head" # how to truncate list of tokens in ["head", "tail", "headtail"]
# file_results = "tokens_subset.csv" # None = dont save this dataframe as csv
# -----------------------------------------------



def prepare_data(file_data, returnDF, max_tokencount=510, truncating_method="head", file_results=None, num_rows=None, embedding_type="bert"):

    """
    Prepare the data for the BERT model.
    (1) Load and analyze the given data set
    (2) Tokenize the words
    (3) Truncate oversized data
    (4) Add padding and create the attention mask

    Parameters:
        file_data(string): Path to input data.

        returnDF(bool): Returns tokens as pandas dataframe if true.

        max_tokencount(int): Maximum amount of words/tokens in [1,510] (BERT is limited to 512 including CLS and SEP) for a single review, 
        truncating gets applied if text length exceeds this timit.

        truncating_method(string): Specifies how to truncate an oversized list of tokens using a method from ["head", "tail", "headtail"].

        file_results(string): Specifies whether resulting dataframe should be saved as csv (give path as string) or not (None).

        num_rows(int): Specifies whether input data should be limited to n rows or not (None).

        embedding_type(bool): Specifies which tokenizer should be used (bert/albert).

    Returns:
        data(pandas.DataFrame): Returns dataframe of tokens if returnDF is set to true.
    '''
    """

    logging.basicConfig(format='%(asctime)s [%(levelname)s] - %(message)s', datefmt='%d-%m-%y %H:%M:%S', level=logging.INFO)
    log_starttime = datetime.datetime.now()

    # Load the data
    names = ["UserId", "ReviewText", "Gender"] if "test" in file_data else ["Gender", "ReviewText"]
    if num_rows is None:
        data = pd.read_csv(file_data, names=names)
    else:
        data = pd.read_csv(file_data, nrows=num_rows, names=names)

    # tag statistics
    logging.info("Total: {}".format(len(data)))
    logging.info("Male: {} ({:.2%})".format(len(data[data["Gender"] == 1]), len(data[data["Gender"] == 1])/len(data)))
    logging.info("Female: {} ({:.2%})".format(len(data[data["Gender"] == 0]), len(data[data["Gender"] == 0])/len(data)))

    # load the tokenizer and selector for truncating oversized token lists
    logging.info("Loading {} tokenizer ...".format(embedding_type))
    # OLD: tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    if embedding_type == "bert":
        tokenizer = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=True)
    else: 
        tokenizer = AlbertTokenizer.from_pretrained("albert-base-v1", do_lower_case=True)

    max_tokencount = min(max_tokencount, 510)
    selector = {
        "head": (lambda x: x[:max_tokencount+1] + x[-1:]),
        "tail": (lambda x: x[:1] + x[-max_tokencount-1:]),
        "headtail": (lambda x: x[:max_tokencount//4] + x[max_tokencount//4-max_tokencount:]) # adapted from https://arxiv.org/abs/1905.05583
    }
    logging.info("Done!")

    # tokenize, truncate oversized data (BERT is limited to 512 tokens) and apply padding
    logging.info("Applying tokenizer ...")
    logging.disable(logging.WARNING)
    data["ReviewText"] = data["ReviewText"].map(lambda x: tokenize(x, tokenizer, max_tokencount, selector[truncating_method], embedding_type))
    logging.disable(logging.DEBUG)
    logging.info("Done!")
    global cnt_oversized
    logging.info("{} reviews ({:.2%}) were oversized and truncated".format(cnt_oversized, cnt_oversized/len(data)))

    # padding and attention mask
    data["att_mask"] = data["ReviewText"].map(lambda x: am(x))

    log_endtime = datetime.datetime.now()
    log_runtime = (log_endtime - log_starttime)
    logging.info("Total runtime: " + str(log_runtime))

    # return resulting data
    if file_results != None:
        data.to_csv(file_results, index=False)
    if returnDF:
        return data


def tokenize(text, tokenizer, max_tokencount, truncating_method, embedding_type):
    tokens = tokenizer.encode(text, add_special_tokens=True)
    tokens = tokens.ids if embedding_type == "bert" else tokens
    if len(tokens) > max_tokencount:
        global cnt_oversized
        cnt_oversized += 1
        tokens = truncating_method(tokens)
    tokens += [0] * (max_tokencount - len(tokens) + 2) # this represents the padding
    return tokens


def am(tokens): # create attention mask
    return [int(token > 0) for token in tokens]