import logging
import datetime
from datetime import date, timedelta
import torch
import json
from model import train_model

# ------------ CONFIG ------------
LEARNING_RATES = [2e-5, 3e-5]
MAX_TOKENCOUNTS = [64, 128]
TRUNCATING_METHODS = ["head", "tail", "headtail"]
# --------------------------------

# ------------ CONSTANTS ------------
EPOCHS = 4
BATCH_SIZE = 16
MODEL_TRAIN = True
SAVE_MODEL = True
PRELOAD_MODEL = None
VAL_ROWS = 100000
LOAD_EMBEDDINGS = None
NUM_ROWS_TRAIN = 1000000
RETURN_MODEL = False
# --------------------------------

log_starttime = datetime.datetime.now()
global_stats = {}



for lr in LEARNING_RATES:
    for max_tc in MAX_TOKENCOUNTS:
        for tm in TRUNCATING_METHODS:
            
            logging.info("Starting LR:{}_MAXTC:{}_TM:{} ...".format(lr, max_tc, tm))

            if RETURN_MODEL:
                model, stats = train_model(EPOCHS, lr, BATCH_SIZE, max_tc, tm, MODEL_TRAIN, SAVE_MODEL, PRELOAD_MODEL, VAL_ROWS, LOAD_EMBEDDINGS, NUM_ROWS_TRAIN, RETURN_MODEL, return_stats=True)
                model.save_pretrained("genderBERT_LR{}_MAXTC{}_TM{}".format(lr, max_tc, tm)) # ":" not allowed in a file name
            else:
                stats = train_model(EPOCHS, lr, BATCH_SIZE, max_tc, tm, MODEL_TRAIN, SAVE_MODEL, PRELOAD_MODEL, VAL_ROWS, LOAD_EMBEDDINGS, NUM_ROWS_TRAIN, RETURN_MODEL, return_stats=True)

            global_stats["LR:{}_MAXTC:{}_TM:{}".format(lr, max_tc, tm)]
            with open("../TODO:_ENTER_PATH_HERE/stats.json", "w") as fp:
                json.dump(global_stats, fp)

logging.info("Done!")
log_endtime = datetime.datetime.now()
log_runtime = (log_endtime - log_starttime)
logging.info("Total runtime: " + str(log_runtime))