from model import load_embeddings, create_model, load_to_cuda, create_optimizer, create_dataloader, train_epoch, eval_model, test_model
import math
import json
import logging
import torch
import sys
import datetime
import random


def set_config(mode):
    # Sets the constants using the config.json file
    if len([1, 1]) == 2:
        #mode = sys.argv[1]
        with open("config.json", "r") as fp:
            config = json.load(fp)
        if mode not in config:
            logging.error("Invalid config number!")
        else:
            global EPOCHS, LEARNING_RATE, BATCH_SIZE, MAX_TOKENCOUNT, TRUNCATING_METHOD, TOGGLE_PHASES, SAVE_MODEL, PRELOAD_MODEL, LOAD_EMBEDDINGS, ROWS_COUNTS, MODEL_TYPE, DATASET_TYPE, BASE_FREEZE
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
            DATASET_TYPE = config[mode]["DATASET_TYPE"]
            BASE_FREEZE = config[mode]["BASE_FREEZE"]
    elif len(sys.argv) == 1:
        logging.warning("Config number missing!")
    else:
        logging.error("Invalid arguments!")


def main(return_model):
    set_config("7")
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s] - %(message)s', datefmt='%d-%m-%y %H:%M:%S', level=logging.INFO)
    log_starttime = datetime.datetime.now()
    train_data, val_data, test_data = load_embeddings(
        DATASET_TYPE, MODEL_TYPE, TOGGLE_PHASES, LOAD_EMBEDDINGS, ROWS_COUNTS, MAX_TOKENCOUNT, TRUNCATING_METHOD, save_embeddings=False)
    logging.info("Create dataloaders ...")
    train_dataloader = create_dataloader(train_data, BATCH_SIZE)
    val_dataloader = create_dataloader(val_data, BATCH_SIZE)
    test_dataloader = create_dataloader(test_data, 256)
    # Create model
    model = create_model(PRELOAD_MODEL, MODEL_TYPE, BASE_FREEZE)
    # Set usage of GPU or CPU
    device = load_to_cuda(model)
    if TOGGLE_PHASES[0]:
        num_train_steps = math.ceil(len(train_dataloader)/BATCH_SIZE) * EPOCHS
        optimizer, scheduler = create_optimizer(
            model, LEARNING_RATE, num_train_steps)
    # Train model
    seed_val = 42
    random.seed(seed_val)
    # np.random_seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    stats = {}
    for epoch_i in range(EPOCHS):
        logging.info(
            "-------- Epoch {} / {} --------".format(epoch_i + 1, EPOCHS))
        if TOGGLE_PHASES[0]:
            loss, acc = train_epoch(
                model, train_dataloader, optimizer, device, scheduler)
            stats[str(epoch_i)] = {"loss": loss, "accuracy": acc}
            if SAVE_MODEL is not None:
                logging.info("Save model...")
                model.save_pretrained(
                    "{}_epoch_{}".format(SAVE_MODEL, epoch_i+1))
        # ---VALIDATION---
        if TOGGLE_PHASES[1]:
            acc = eval_model(model, val_dataloader, device, MODEL_TYPE)
    # ---TESTING ---
    if TOGGLE_PHASES[2]:
        acc = test_model(model, test_dataloader, device, MODEL_TYPE)
    logging.info("Training complete!")
    log_endtime = datetime.datetime.now()
    log_runtime = (log_endtime - log_starttime)
    logging.info("Total runtime: " + str(log_runtime))
    if return_model:
        return model, stats
    else:
        return stats


if __name__ == "__main__":
    main(False)
