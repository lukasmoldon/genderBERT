import logging
import datetime
from datetime import date, timedelta
import pandas as pd
import logging
from transformers import AlbertTokenizer, BertTokenizer, OpenAIGPTTokenizer, RobertaTokenizer, AutoTokenizer, DistilBertTokenizer
from tokenizers import BertWordPieceTokenizer
import torch
from sklearn import preprocessing

cnt_oversized = 0

# ------------ config example values ------------
# file_data = "data_subset.csv"
# returnDF = True # should the function return a pandas dataframe object?
# max_tokencount = 510 # threshold in [1,510] (BERT is limited to 512 including CLS and SEP) 
# truncating_method = "head" # how to truncate list of tokens in ["head", "tail", "headtail"]
# file_results = "tokens_subset.csv" # None = dont save this dataframe as csv
# -----------------------------------------------



def prepare_data(file_data, return_data, max_tokencount=510, truncating_method="head", file_results=None, num_rows=None, embedding_type="bert", dataset_type="amazon"):

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

        dataset_type(string): Specifies which dataset to use (amazon/reddit/stackover)

    Returns:
        data(pandas.DataFrame): Returns dataframe of tokens if returnDF is set to true.
    '''
    """

    logging.basicConfig(format='%(asctime)s [%(levelname)s] - %(message)s', datefmt='%d-%m-%y %H:%M:%S', level=logging.INFO)
    log_starttime = datetime.datetime.now()
    # Load the data
    if dataset_type == "amazon":
        names = ["UserId", "ReviewText", "Gender"] if "test" in file_data else ["Gender", "ReviewText"]
    elif dataset_type == "demo":
        names = ["Gender", "ReviewText"]
    else:
        names = ["UserId", "Gender", "ReviewText"] if "test" in file_data else ["Gender", "ReviewText"]
    if num_rows is None:
        data = pd.read_csv(file_data, names=names)
    else:
        data = pd.read_csv(file_data, nrows=num_rows, names=names)
    # Swap columns
    if dataset_type != "amazon":
        data = data.reindex(columns=["UserId", "ReviewText", "Gender"]) if "test" in file_data else data
    # tag statistics
    logging.info("Total: {}".format(len(data)))
    logging.info("Male: {} ({:.2%})".format(len(data[data["Gender"] == 1]), len(data[data["Gender"] == 1])/len(data)))
    logging.info("Female: {} ({:.2%})".format(len(data[data["Gender"] == 0]), len(data[data["Gender"] == 0])/len(data)))

    # load the tokenizer and selector for truncating oversized token lists
    logging.info("Loading {} tokenizer ...".format(embedding_type))
    # OLD: tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    if embedding_type == "bert" or embedding_type == "custom_bert":
        tokenizer = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=True)
    elif embedding_type == "albert": 
        tokenizer = AlbertTokenizer.from_pretrained("albert-base-v1", do_lower_case=True)
    elif embedding_type == "gpt":
        tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt", do_lower_case=True)
    elif embedding_type == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=True)
    elif embedding_type == "distilbert":
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", do_lower_case=True)
    elif embedding_type == "sentiment_bert":
        tokenizer = BertTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment", do_lower_case=True)
    else:
        logging.error("Unknown embedding type!")
        exit()
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
    # TODO: Make this better intregrated into the rest of the code
    # Turn dataframe back to dict for torch.save
    results = {}
    results["input_ids"] = torch.tensor(list(data["ReviewText"]))
    results["attention_mask"] = torch.tensor(list(data["att_mask"]))
    target_tensor = torch.tensor(pd.to_numeric(list(data["Gender"])))
    results["target"] = target_tensor
    if 'UserId' in data.columns:
        id_list = list(data["UserId"])
        le = preprocessing.LabelEncoder()
        user_ids = le.fit_transform(id_list)
        id_tens = torch.tensor(user_ids)
        results["user_id"] = id_tens
    # return resulting data
    if file_results != None:
        torch.save(results, file_results)
    if return_data:
        return results


def tokenize_data(file_data, returnRes, max_tokencount=510, truncating_method="head", file_results=None, num_rows=None, embedding_type="bert", dataset_type="amazon"):

    """
    Cleaner version of prepare_data (BUT WITHOUT TRUNCATION STRATEGY)
    Prepare the data for the BERT model.
    (1) Load and analyze the given data set
    (2) Tokenize the words
    (3) Truncate oversized data
    (4) Add padding and create the attention mask

    Parameters:
        file_data(string): Path to input data.

        returnRes(bool): Returns result if true.

        max_tokencount(int): Maximum amount of words/tokens in [1,510] (BERT is limited to 512 including CLS and SEP) for a single review, 
        truncating gets applied if text length exceeds this timit.

        truncating_method(string): Specifies how to truncate an oversized list of tokens using a method from ["head", "tail", "headtail"]. 
                                   MAKES NO DIFFERENCE HERE!

        file_results(string): Specifies whether result should be saved (give path as string) or not (None).

        num_rows(int): Specifies whether input data should be limited to n rows or not (None).

        embedding_type(bool): Specifies which tokenizer should be used (bert/albert).

        dataset_type(string): Specifies which dataset to use (amazon/reddit/stackover)

    Returns:
        data(pandas.DataFrame): Returns dataframe of tokens if returnDF is set to true.
    '''
    """

    logging.basicConfig(format='%(asctime)s [%(levelname)s] - %(message)s', datefmt='%d-%m-%y %H:%M:%S', level=logging.INFO)
    # Load the data
    if dataset_type == "amazon":
        names = ["UserId", "ReviewText", "Gender"] if "test" in file_data else ["Gender", "ReviewText"]
    else:
        names = ["UserId", "Gender", "ReviewText"] if "test" in file_data else ["Gender", "ReviewText"]
    if num_rows is None:
        data = pd.read_csv(file_data, names=names)
    else:
        data = pd.read_csv(file_data, nrows=num_rows, names=names)
    # Swap columns
    if dataset_type != "amazon":
        data = data.reindex(columns=["UserId", "ReviewText", "Gender"]) if "test" in file_data else data
    # tag statistics
    logging.info("Total: {}".format(len(data)))
    logging.info("Male: {} ({:.2%})".format(len(data[data["Gender"] == 1]), len(data[data["Gender"] == 1])/len(data)))
    logging.info("Female: {} ({:.2%})".format(len(data[data["Gender"] == 0]), len(data[data["Gender"] == 0])/len(data)))

    # load the tokenizer and selector for truncating oversized token lists
    logging.info("Loading {} tokenizer ...".format(embedding_type))
    pretrained_name = { "bert": "bert-base-uncased",
                        "custom_bert": "bert-base-uncased",
                        "albert": "albert-base-v1",
                        "gpt2": "gpt2",
                        "distilbert": "distilbert-base-uncased",
                        "roberta": "roberta-base",
                        "sentiment_bert": "nlptown/bert-base-multilingual-uncased-sentiment"}
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name[embedding_type], do_lower_case=True)
    max_tokencount = min(max_tokencount, 510)
    logging.info("Done!")
    # tokenize, truncate oversized data (BERT is limited to 512 tokens) and apply padding
    logging.info("Applying tokenizer ...")
    logging.disable(logging.WARNING)
    print(data.head())
    encoding = tokenizer.batch_encode_plus(data["ReviewText"], return_tensors="pt", padding=True, truncation=True, max_length=max_tokencount, add_special_tokens=True)
    target_tensor = torch.tensor(pd.to_numeric(list(data["Gender"])))
    encoding["target"] = target_tensor
    if 'UserId' in data.columns:
        id_list = list(data["UserId"])
        le = preprocessing.LabelEncoder()
        user_ids = le.fit_transform(id_list)
        id_tens = torch.tensor(user_ids)
        encoding["user_id"] = id_tens
    # return resulting data
    if file_results != None:
        torch.save(encoding, file_results)
    if returnRes:
        return encoding        


def tokenize(text, tokenizer, max_tokencount, truncating_method, embedding_type):
    tokens = tokenizer.encode(text, add_special_tokens=True)
    tokens = tokens.ids if embedding_type == "bert" or embedding_type == "custom_bert" else tokens
    if len(tokens) > max_tokencount:
        global cnt_oversized
        cnt_oversized += 1
        tokens = truncating_method(tokens)
    tokens += [0] * (max_tokencount - len(tokens) + 2) # this represents the padding
    return tokens


def am(tokens): # create attention mask
    return [int(token > 0) for token in tokens]

