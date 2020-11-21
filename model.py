import logging
import datetime
from datetime import date, timedelta
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AlbertForSequenceClassification, AdamW, BertConfig, AutoModelForSequenceClassification
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
from majority_voting import mv_stats_f1
from customBERT import CustomBERTModel


PATHS = {"amazon": ["../datasets/amazon/User_level_train.csv", "../datasets/amazon/User_level_validation.csv", "../datasets/amazon/User_level_test_with_id.csv"],
         "stackover": ["../datasets/stackover/train_so.csv", "../datasets/stackover/validation_so.csv", "../datasets/stackover/test_so.csv"],
         "reddit": ["../datasets/reddit/train_reddit.csv", "../datasets/reddit/validation_reddit.csv", "../datasets/reddit/test_reddit.csv"],
         "try": ["try.csv", None, None]}


def load_embeddings(dataset_type, model_type, toggle_phases, load_embeddings, rows_counts, max_tokencount, truncating_method, save_embeddings):
    # Load tokenized data
    logging.info("Tokenize train/validation data ...")
    embeddings = [None, None, None]
    embeddings = [None, None, None]
    paths = PATHS[dataset_type]
    phases = ["train", "val", "test"]
    embedding_paths = [None, None, None]
    # Create paths for saving embeddings (if saving is selected)
    if save_embeddings:
        for i in range(3):
            embedding_paths[i] = "{}_{}_{}_{}_{}_{}".format(
                dataset_type,
                model_type,
                rows_counts[i],
                max_tokencount,
                truncating_method,
                phases[i],
            )
    for i in range(3):
        # Toggle Phases: Decide whether to load data for train(0)/val(1)/test(2)
        if toggle_phases[i]:
            # Load embedding if given
            if load_embeddings is not None:
                if load_embeddings[i] is not None:
                    embeddings[i] = torch.load(load_embeddings[i])
            # Create exactly row counts many embedded instances
            elif rows_counts[i] is not None:    
                embeddings[i] = prepare_data(paths[i], True, num_rows=rows_counts[i], max_tokencount=max_tokencount,
                                                 truncating_method=truncating_method, embedding_type=model_type,
                                                 dataset_type=dataset_type, file_results=embedding_paths[i])
            # Embed the whole file
            else:
                embeddings[i] = prepare_data(paths[i], True, max_tokencount=max_tokencount, embedding_type=model_type,
                                                 dataset_type=dataset_type, file_results=embedding_paths[i])
    train_data, val_data, test_data = embeddings
    return train_data, val_data, test_data


def create_model(preload_model, model_type, bert_freeze):
    # Create/load model
    if model_type == "custom_bert":
        model = CustomBERTModel()
    else:
        model_init = {"bert": "bert-base-uncased",
                      "albert": "albert-base-v1",
                      "gpt2": "gpt2",
                      "roberta": "roberta-base",
                      "sentiment_bert": "nlptown/bert-base-multilingual-uncased-sentiment"}
        if preload_model is None:
            preload_model = model_init[model_type]
            logging.info("Create new model...")
        else:
            logging.info("Preload model...")
        num_labels = 5 if model_type == "sentiment_bert" else 2
        # GPT2 has no dropout param
        if model_type == "gpt2":
            model = AutoModelForSequenceClassification.from_pretrained(
                preload_model,
                num_labels=num_labels,
                output_attentions=False,
                output_hidden_states=False,
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                preload_model,
                attention_probs_dropout_prob=0.2,
                num_labels=num_labels,
                output_attentions=False,
                output_hidden_states=False,
            )
        if model_type == "sentiment_bert":
            model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
            model.num_labels = 2
    # Freeze BERT layers of model
    if bert_freeze:
        logging.info("Freeze BERT layers...")
        for name, param in model.named_parameters():
            if 'classifier' not in name and 'linear' not in name:  # classifier layer
                param.requires_grad = False
    return model


def load_to_cuda(model):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        model.cuda()
        logging.info("Use GPU: {}".format(torch.cuda.get_device_name(0)))
    else:
        device = torch.device("cpu")
        logging.warning("No GPU available, use CPU instead.")
    return device


def create_optimizer(model, learning_rate, num_train_steps):
    optimizer = AdamW(model.parameters(),
                      lr=learning_rate,
                      eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_train_steps*0.1,
                                                num_training_steps=num_train_steps)
    return optimizer, scheduler


def flat_accuracy(predicitions, labels):
    # Helper function for accuracy
    pred_flat = np.argmax(predicitions, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def create_dataloader(data, batch_size):
    # Creates DataLoader from dataframe.
    if data is None:
        return None
    inputs = data["input_ids"]
    labels = data["target"]
    mask = data["attention_mask"]
    if "user_id" in data:
        user_ids = data["user_id"]
        data = TensorDataset(inputs, labels, mask, user_ids)
    else:
        data = TensorDataset(inputs, labels, mask)
    # Create TensorDataset/Dataloader for faster training
    logging.info("Create DataLoader ...")
    sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader


def train_epoch(model, data_loader, optimizer, device, scheduler):
    # Trains given model for one epoch
    logging.info("Training ...")
    total_loss = 0
    model.train()
    tmp_train_acc, steps = 0, 0
    for _, batch in enumerate(data_loader):
        b_input_ids = batch[0].to(device)
        b_labels = batch[1].to(device)
        b_input_mask = batch[2].to(device)
        model.zero_grad()
        # batch[0]: ids, [1]:masks, [2]:labels
        outputs = model(b_input_ids,
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


def eval_model(model, data_loader, device, model_type):
    # Evaluate given model
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
                            attention_mask=b_input_mask,
                            labels=None)
            ind = 1 if model_type == "custom_bert" else 0
            logits = outputs[ind].to("cpu")
            label_ids = b_labels.to("cpu")
            _, predicted = torch.max(logits.data, 1)
            tmp_eval_acc += (predicted == label_ids).sum().item()
            steps += b_labels.size(0)
    accuracy = tmp_eval_acc/steps
    logging.info("Accuracy: {0:.2f}".format(accuracy))
    return accuracy


def test_model(model, data_loader, device, model_type):
    # Test given model
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
                            attention_mask=b_input_mask,
                            labels=None)
            ind = 1 if model_type == "custom_bert" else 0
            logits = outputs[ind].to("cpu")
            label_ids = b_labels.to("cpu")
            _, predicted = torch.max(logits.data, 1)
            tmp_eval_acc += (predicted == label_ids).sum().item()
            steps += b_labels.size(0)
            # Add to dataframe lists
            user_id_list += b_user_ids.tolist()
            label_list += b_labels.tolist()
            pred_list += predicted.tolist()
    df = pd.DataFrame(
        {"userid": user_id_list, "label": label_list, "prediction": pred_list})
    non_mv_accuracy = tmp_eval_acc/steps
    logging.info("Accuracy (Non-MV): {:.4}".format(non_mv_accuracy))
    f1_male = mv_stats_f1(df, 1, pred_label="prediction")
    logging.info("F1 male = {:.4}".format(f1_male))
    f1_female = mv_stats_f1(df, 0, pred_label="prediction")
    logging.info("F1 female = {:.4}".format(f1_female))
    mv(df)
    return non_mv_accuracy
