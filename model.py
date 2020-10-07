import logging
import datetime
from datetime import date, timedelta
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import ast
import random
import numpy as np
from sklearn.model_selection import train_test_split
from tokenizer import prepare_data


# ------------ CONFIG ------------
EPOCHS = 10
BATCH_SIZE = 16
MODEL_TRAIN = True
SAVE_MODEL = True
PRELOAD_MODEL = None
VAL_ROWS = 100000
LOAD_EMBEDDINGS = None
NUM_ROWS_TRAIN = 1000000
# --------------------------------

# Helper function for accuracy
def flat_accuracy(predicitions, labels):
    pred_flat = np.argmax(predicitions, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def train_model(epochs, batch_size, model_train=True, save_model=True, preload_model=None, val_rows=None, load_embeddings=None, num_rows_train=None,
                return_model=False):
    """
    Train a BERT model.

    Parameters:
        epochs(int): Number of epochs for training.

        batch_size(int): Batch size.

        model_train(bool): Specifies whether model should be trained or not. Does training if true, skips it (i.e. only do validation)
        if false.

        save_model(bool): Saves model after each epoch if true.

        preload_model(string): Preload model with given path. Train new model if None.

        val_rows(int): Specifies how many data entries are used in the validation set. No validation if None.

        load_embeddings(string): Preload embeddings from given path. Create new embeddings if None.

        num_rows_train(int): Specifies how many data entries are used in the train set.

        return_model(bool): Returns final model if set to true.

    Returns:
        model(BertForSequenceClassification): Returns trained model if return_model is set to true.
    '''
    """

    logging.basicConfig(format='%(asctime)s [%(levelname)s] - %(message)s', datefmt='%d-%m-%y %H:%M:%S', level=logging.INFO)
    log_starttime = datetime.datetime.now()

    # Load tokenized data
    if load_embeddings is not None:
        logging.info("Load train/validation data ...")
        # Todo
        exit()
    else:
        logging.info("Tokenize train/validation data ...")
        if num_rows_train is not None:
            embeddings = prepare_data("../datasets/amazon/User_level_train.csv", True, num_rows=num_rows_train, max_tokencount=128, truncating_method="headtail")
        else:
            embeddings = prepare_data("../datasets/amazon/User_level_train.csv", True, max_tokencount=128)
        if val_rows is not None:
            val_data = prepare_data("../datasets/amazon/User_level_validation.csv", True, num_rows=val_rows, max_tokencount=128, truncating_method="headtail")
        else:
            val_data = prepare_data("../datasets/amazon/User_level_validation.csv", True, max_tokencount=128)

    # Create tensors from loaded data
    logging.info("Create tensors ...")
    inputs = list(embeddings["ReviewText"])
    labels = list(embeddings["Gender"])
    mask = list(embeddings["att_mask"])
    val_inputs = list(val_data["ReviewText"])
    val_labels = list(val_data["Gender"])
    val_mask = list(val_data["att_mask"])

    X_train = torch.tensor(inputs)
    X_val = torch.tensor(val_inputs)
    y_train = torch.tensor(labels)
    y_val = torch.tensor(val_labels)
    train_mask = torch.tensor(mask)
    validation_mask = torch.tensor(val_mask)

    # Create TensorDataset/Dataloader for faster training
    logging.info("Create DataLoader ...")
    train_data = TensorDataset(X_train, y_train, train_mask)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(X_val, y_val, validation_mask)
    validation_sampler = RandomSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    # Create/load model
    if preload_model is None:
        logging.info("Create new model ...")
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False
        )
    else:
        logging.info("Load pretrained model ...")
        model = BertForSequenceClassification.from_pretrained(
            preload_model,
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

    # TODO: Find good values
    optimizer = AdamW(model.parameters(),
                    lr=2e-5,
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
    for epoch_i in range(epochs):
        logging.info("-------- Epoch {} / {} --------".format(epoch_i + 1, epochs))
        if model_train:
            logging.info("Training ...")
            total_loss = 0
            model.train()
            for step, batch in enumerate(train_dataloader):
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
            avg_train_loss = total_loss / len(train_dataloader)
            loss_values.append(avg_train_loss)
            logging.info("Average train loss: {0:.2f}".format(avg_train_loss))
            if save_model:
                logging.info("Save model...")
                # TODO: Use Tempfile instead of this mess of a name
                model.save_pretrained("genderBERT_epoch_{}_V4".format(epoch_i))
        # ---VALIDATION--- 
        if val_rows is not None:
            logging.info("Start validation ...")
            model.eval()
            tmp_eval_acc, steps = 0, 0
            for batch in validation_dataloader:
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
            logging.info("Accuracy: {0:.2f}".format(tmp_eval_acc/steps))
    logging.info("Training complete!")
    log_endtime = datetime.datetime.now()
    log_runtime = (log_endtime - log_starttime)
    logging.info("Total runtime: " + str(log_runtime))

    if return_model:
        return model

#train_model(EPOCHS, BATCH_SIZE, MODEL_TRAIN, SAVE_MODEL, PRELOAD_MODEL, VAL_ROWS, LOAD_EMBEDDINGS, NUM_ROWS_TRAIN)