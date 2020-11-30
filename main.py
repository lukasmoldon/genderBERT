from model import load_embeddings, create_model, load_to_cuda, create_optimizer, create_dataloader, train_epoch, eval_model, test_model
import math
import json
import logging
import torch
import sys
import datetime
import random


def set_config():
    # Sets the constants using the config.json file
    if len(sys.argv) == 2:
        mode = sys.argv[1]
        with open("config.json", "r") as fp:
            config = json.load(fp)
        if mode not in config:
            logging.error("Invalid config number!")
        else:
            global EPOCHS, LEARNING_RATE, BATCH_SIZE, MAX_TOKENCOUNT, TRUNCATING_METHOD, TOGGLE_PHASES, SAVE_MODEL, PRELOAD_MODEL, LOAD_EMBEDDINGS, ROWS_COUNTS, MODEL_TYPE, DATASET_TYPE, BASE_FREEZE
            # Number of epochs
            EPOCHS = config[mode]["EPOCHS"]
            # Learning rate
            LEARNING_RATE = config[mode]["LEARNING_RATE"]
            # Batch size
            BATCH_SIZE = config[mode]["BATCH_SIZE"]
            # Max token count (used for embedding)
            MAX_TOKENCOUNT = config[mode]["MAX_TOKENCOUNT"]
            # Methods: Head, Tail, Headtail
            TRUNCATING_METHOD = config[mode]["TRUNCATING_METHOD"]
            # Bool array of the form: [Do_Train_Phase, Do_Val_Phase, Do_Test_Phase]
            TOGGLE_PHASES = config[mode]["TOGGLE_PHASES"]
            # Save to given path, do not save if none
            SAVE_MODEL = config[mode]["SAVE_MODEL"]
            # Load model from given path, do not load if none
            PRELOAD_MODEL = config[mode]["PRELOAD_MODEL"]
            # Load embeddings from given path of a size 3 array [train, val, test]
            # do not load if corresponding entry is none
            LOAD_EMBEDDINGS = config[mode]["LOAD_EMBEDDINGS"]
            # Number of rows to consider, given by a size 3 array (see above)
            ROWS_COUNTS = config[mode]["ROWS_COUNTS"]
            # Types: bert, albert, roberta, distilbert, custombert
            MODEL_TYPE = config[mode]["MODEL_TYPE"]
            # Types: amazon, reddit, stackover
            DATASET_TYPE = config[mode]["DATASET_TYPE"]
            # Freeze base layers if True
            BASE_FREEZE = config[mode]["BASE_FREEZE"]
    elif len(sys.argv) == 1:
        logging.warning("Config number missing!")
    else:
        logging.error("Invalid arguments!")


def main(return_model):

    """
    Main function of the module. Create a model, train and evaluate it based on the config JSON.
    For an explanation of the JSON see the set_config() function.
    Steps of main function:
    (1) Create/Load embeddings for train, validation and test data.
    (2) Create dataloader from embeddings for faster and easier training.
    (3) Create model, optimizer and scheduler. Load to CUDA.
    (4) Train, validate and test model (according to config).
    (5) Save/return model if parameter is set.

    Parameters:
        return_model(bool): Returns model and stats (loss/acc) if set to True, only return latter if set to False.

    Returns:
        model(transformers.AutoModelForSequenceClassification): Trained model. Gets return if return_model is True.

        stats([dict]): Array of train loss/acc of model for each epoch.
    '''
    """
    
    set_config()
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
        # --- VALIDATION ---
        if TOGGLE_PHASES[1]:
            acc = eval_model(model, val_dataloader, device, MODEL_TYPE)
    # --- TESTING ---
    if TOGGLE_PHASES[2]:
        acc = test_model(model, test_dataloader, device, MODEL_TYPE)
    # --- SAVING ---
    if SAVE_MODEL is not None:
        logging.info("Save model...")
        model.save_pretrained(SAVE_MODEL)
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
