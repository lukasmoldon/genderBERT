import logging
import datetime
from datetime import date, timedelta
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AlbertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import ast
import random
import numpy as np
from sklearn.model_selection import train_test_split
from tokenizer import prepare_data
import json
import sys
from sklearn import preprocessing
from majority_voting import mv



# ------------ CONFIG ------------
EPOCHS = 10
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
MAX_TOKENCOUNT = 128
TRUNCATING_METHOD = "headtail"
TOGGLE_PHASES = [True, True, False]
SAVE_MODEL = None
PRELOAD_MODEL = None
LOAD_EMBEDDINGS = None
ROWS_COUNTS = [100, 10, 10]
MODEL_TYPE = "bert"
# --------------------------------

def train_model(epochs, learning_rate, batch_size, max_tokencount=510, truncating_method="head", toggle_phases=[True, True, False], save_model=True,
                preload_model=None, load_embeddings=None, rows_counts=[None, None, None], model_type="bert",
                return_model=False, return_stats=False):
    """
    Run main script for model training/validation/testing.

    Parameters:
        epochs(int): Number of epochs for training.

        learning_rate(int): Learning rate of the model.

        batch_size(int): Batch size.

        max_tokencount(int): Maximum amount of words/tokens in [1,510] (BERT is limited to 512 including CLS and SEP) for a single review, 
        truncating gets applied if text length exceeds this timit.

        truncating_method(string): Specifies how to truncate an oversized list of tokens using a method from ["head", "tail", "headtail"].

        toggle_phases([bool]): Specifies which phases should be run ([Training, Validation, Testing]).

        save_model(string): Saves model after each epoch to the given path, no saving if None.

        preload_model(string): Preload model with given path. Train new model if None.

        load_embeddings(string): Preload embeddings from given path. Create new embeddings if None.

        rows_counts([int]): Specifies how many data entries are used for training/validation/testing ([Training, Validation, Testing]).

        model_type(string): Specifies type of model (bert/albert).

        return_model(bool): Returns final model if set to true.

        return_stats(bool): Returns statistics of accuracy and loss for each epoch in dict format.

    Returns:
        model(BertForSequenceClassification): Returns trained model if return_model is set to true.
    '''
    """
    logging.basicConfig(format='%(asctime)s [%(levelname)s] - %(message)s', datefmt='%d-%m-%y %H:%M:%S', level=logging.INFO)
    log_starttime = datetime.datetime.now()
    # Load tokenized data
    if load_embeddings is not None:
        logging.info("Load train/validation data ...")
        # Todo: implement
        exit()
    else:
        logging.info("Tokenize train/validation data ...")
        dataframes = [None, None, None]
        paths = ["../datasets/amazon/User_level_train.csv", "../datasets/amazon/User_level_validation.csv", "../datasets/amazon/User_level_test_with_id.csv"]
        for i in range(3):
            if toggle_phases[i]:
                if rows_counts[i] is not None:
                    dataframes[i] = prepare_data(paths[i], True, num_rows=rows_counts[i], max_tokencount=max_tokencount, truncating_method=truncating_method, embedding_type=model_type)
                else:
                    dataframes[i] = prepare_data(paths[i], True, max_tokencount=max_tokencount, embedding_type=model_type)
    #test_df = pd.read_csv("test_tokenized.csv")
    #test_df["ReviewText"] = test_df["ReviewText"].map(ast.literal_eval)
    #test_df["att_mask"] = test_df["att_mask"].map(ast.literal_eval)
    #dataframes[2] = test_df
    train_df, val_df, test_df = dataframes
    # Create tensors from loaded data
    logging.info("Create tensors ...")
    train_dataloader = create_dataloader(train_df, batch_size)
    val_dataloader = create_dataloader(val_df, batch_size)
    test_dataloader = create_dataloader(test_df, 512)

    # Create/load model
    if preload_model is None:
        model_init = {"bert": (BertForSequenceClassification, "bert-base-uncased"), 
                      "albert": (AlbertForSequenceClassification, "albert-base-v1")}
        logging.info("Create new model ...")
        model = model_init[model_type][0].from_pretrained(
            model_init[model_type][1],
            attention_probs_dropout_prob=0.2,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False,
        )
    else:
        model_init = {"bert": (BertForSequenceClassification, "bert-base-uncased"), 
                      "albert": (AlbertForSequenceClassification, "albert-base-v1")}
        logging.info("Load pretrained model ...")
        model = model_init[model_type][0].from_pretrained(
            preload_model,
            attention_probs_dropout_prob=0.2,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False
        )

    # Set usage of GPU or CPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        model.cuda()
        logging.info("Use GPU: {}".format(torch.cuda.get_device_name(0)))
    else:
        device = torch.device("cpu")
        logging.warning("No GPU available, use CPU instead.")
    if toggle_phases[0]:
        optimizer = AdamW(model.parameters(),
                        lr=learning_rate,
                        eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=len(train_dataloader) * epochs)
                                                
    # Train model
    seed_val = 42
    random.seed(seed_val)
    # np.random_seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    loss_values = []
    stats = {}

    # TODO: Save best model
    for epoch_i in range(epochs):
        logging.info("-------- Epoch {} / {} --------".format(epoch_i + 1, epochs))
        if toggle_phases[0]:
            loss, acc = train_epoch(model, train_dataloader, optimizer, device, scheduler)
            stats[str(epoch_i)] = {"loss": loss, "accuracy": acc}
            if save_model is not None:
                logging.info("Save model...")
                model.save_pretrained("{}_epoch_{}".format(save_model, epoch_i+1))
        # ---VALIDATION--- 
        if toggle_phases[1]:
            acc = eval_model(model, val_dataloader, device)
            #stats[str(epoch_i)]["accuracy"] = acc
    # ---TESTING ---
    if toggle_phases[2]:
        acc = test_model(model, test_dataloader, device)
        #stats[str(epoch_i)]["accuracy (Non-MV)"] = acc

    logging.info("Training complete!")
    log_endtime = datetime.datetime.now()
    log_runtime = (log_endtime - log_starttime)
    logging.info("Total runtime: " + str(log_runtime))

    if return_stats:
        if return_model:
            return model, stats
        else:
            return stats
    else:
        if return_model:
            return model


def set_config():
    if len(sys.argv) == 2:
        mode = sys.argv[1]
        with open("config.json", "r") as fp:
            config = json.load(fp)
        if mode not in config:
            logging.error("Invalid config number!")
        else:
            global EPOCHS, LEARNING_RATE, BATCH_SIZE, MAX_TOKENCOUNT, TRUNCATING_METHOD, TOGGLE_PHASES, SAVE_MODEL, PRELOAD_MODEL, LOAD_EMBEDDINGS, ROWS_COUNTS, MODEL_TYPE
            EPOCHS = config[mode]["EPOCHS"]
            LEARNING_RATE = config[mode]["LEARNING_RATE"]
            BATCH_SIZE = config[mode]["BATCH_SIZE"]
            MAX_TOKENCOUNT = config[mode]["MAX_TOKENCOUNT"]
            TRUNCATING_METHOD = config[mode]["TRUNCATING_METHOD"]
            TOGGLE_PHASES = config[mode]["TOGGLE_PHASES"]
            SAVE_MODEL = config[mode]["SAVE_MODEL"]
            PRELOAD_MODEL = config[mode]["PRELOAD_MODEL"]
            LOAD_EMBEDDINGS = config[mode]["LOAD_EMBEDDINGS"]
            ROWS_COUNTS = config[mode]["ROWS_COUNTS"]
            MODEL_TYPE = config[mode]["MODEL_TYPE"]
    elif len(sys.argv) == 1:
        logging.warning("Config number missing!")
    else:
        logging.error("Invalid arguments!")


# Helper function for accuracy
def flat_accuracy(predicitions, labels):
    pred_flat = np.argmax(predicitions, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def create_dataloader(dataframe, batch_size):
    if dataframe is None:
        return None
    logging.info("Create tensors ...")
    inputs = list(dataframe["ReviewText"])
    labels = list(dataframe["Gender"])
    mask = list(dataframe["att_mask"])
    X_tens = torch.tensor(inputs)
    y_tens = torch.tensor(labels)
    mask_tens = torch.tensor(mask)
    if "UserId" in dataframe.columns:
        user_ids = list(dataframe["UserId"])
        le = preprocessing.LabelEncoder()
        user_ids = le.fit_transform(user_ids)
        id_tens = torch.tensor(user_ids)
        data = TensorDataset(X_tens, y_tens, mask_tens, id_tens)
    else:
        data = TensorDataset(X_tens, y_tens, mask_tens)
    # Create TensorDataset/Dataloader for faster training
    logging.info("Create DataLoader ...")
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader


def train_epoch(model, data_loader, optimizer, device, scheduler):
    # TODO: How exactly is the loss function specified?
    logging.info("Training ...")
    total_loss = 0
    model.train()
    tmp_train_acc, steps = 0, 0
    for step, batch in enumerate(data_loader):
        b_input_ids = batch[0].to(device)
        b_labels = batch[1].to(device)
        b_input_mask = batch[2].to(device)
        model.zero_grad()
        # batch[0]: ids, [1]:masks, [2]:labels
        outputs = model(b_input_ids,
                        token_type_ids=None,
                        labels=b_labels,
                        attention_mask=b_input_mask)
        loss = outputs[0]
        total_loss += loss.item()
        loss.backward()
        # Clip norm of gradients to 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        logits = outputs[1].to("cpu")
        label_ids = b_labels.to("cpu")
        _, predicted = torch.max(logits, 1)
        tmp_train_acc += (predicted == label_ids).sum().item()
        steps += b_labels.size(0)
    avg_train_loss = total_loss / len(data_loader)
    accuracy = tmp_train_acc/steps
    logging.info("Accuracy: {0:.2f}".format(accuracy))
    logging.info("Average train loss: {0:.2f}".format(avg_train_loss))
    return avg_train_loss, accuracy


def eval_model(model, data_loader, device):
    # TODO: Add loss
    logging.info("-" * 30)
    logging.info("Start validation ...")
    model.eval()
    tmp_eval_acc, steps = 0, 0
    for batch in data_loader:
        b_input_ids = batch[0].to(device)
        b_labels = batch[1].to(device)
        b_input_mask = batch[2].to(device)
        with torch.no_grad():
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)  
            logits = outputs[0].to("cpu")
            label_ids = b_labels.to("cpu")
            _, predicted = torch.max(logits.data, 1)
            tmp_eval_acc += (predicted == label_ids).sum().item()
            steps += b_labels.size(0)
    accuracy = tmp_eval_acc/steps
    logging.info("Accuracy: {0:.2f}".format(accuracy))
    return accuracy


def test_model(model, data_loader, device):
    # TODO: Add loss
    logging.info("-" * 30)
    logging.info("Start testing ...")
    model.eval()
    tmp_eval_acc, steps = 0, 0
    # Create lists to be converted to dataframe later
    user_id_list, label_list, pred_list = [], [], []
    for batch in data_loader:
        b_input_ids = batch[0].to(device)
        b_labels = batch[1].to(device)
        b_input_mask = batch[2].to(device)
        b_user_ids = batch[3].to(device)
        with torch.no_grad():
            outputs = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)  
            logits = outputs[0].to("cpu")
            label_ids = b_labels.to("cpu")
            _, predicted = torch.max(logits.data, 1)
            tmp_eval_acc += (predicted == label_ids).sum().item()
            steps += b_labels.size(0)
            # Add to dataframe lists
            user_id_list += b_user_ids.tolist()
            label_list += b_labels.tolist()
            pred_list += predicted.tolist()
    df = pd.DataFrame({"userid": user_id_list, "label": label_list, "prediction": pred_list})
    non_mv_accuracy = tmp_eval_acc/steps
    logging.info("Accuracy (Non-MV): {0:.2f}".format(non_mv_accuracy))
    mv(df)
    return non_mv_accuracy   

     
set_config()
train_model(EPOCHS, LEARNING_RATE, BATCH_SIZE, MAX_TOKENCOUNT, TRUNCATING_METHOD, TOGGLE_PHASES, SAVE_MODEL, PRELOAD_MODEL, LOAD_EMBEDDINGS, ROWS_COUNTS, MODEL_TYPE)