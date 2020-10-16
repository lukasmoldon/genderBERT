import logging
import datetime
from datetime import date, timedelta
import pandas as pd



def mv(df, debug=False):
    """
    Compute the majority voting for a given prediction of a BERT model and display accuracy and F1 after applying majority voting. 
    (The function does not change predictions for users with no predicted majority for one gender.)

    Parameters:
        df(pd.DataFrame): Pandas DataFrame with ['userid', 'label', 'prediction'] as columns. There can only be one specific label
        for all occurrences of a single userid in the dataframe.

        debug(bool): Specifies whether debug messages should be on (True) or not (False). Default is False.

    Returns:
        Accuracy, F1 (male), F1 (female) as console output(!)
    '''
    """
    if debug:
        logging.basicConfig(format='%(asctime)s [%(levelname)s] - %(message)s', datefmt='%d-%m-%y %H:%M:%S', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(asctime)s [%(levelname)s] - %(message)s', datefmt='%d-%m-%y %H:%M:%S', level=logging.INFO)
    log_starttime = datetime.datetime.now()
    logging.info("Computing majority voting..")
    ids = df["userid"].unique()
    res_verify = mv_verify(df, ids)
    if res_verify == 0:
        logging.debug("No inconsistencies in input data (unique label assignment)")
    else:
        logging.warning("Found {} user(s) ({:.2%}) with inconsistencies in input data (no unique label assignment)".format(res_verify, res_verify/len(ids)))
    logging.debug("Input data verification successful.")
    voting = mv_compute(df, ids)
    logging.info("Computing majority voting stats..")
    logging.info("")
    logging.info("----------------------------------")
    acc = mv_stats_acc(voting)
    logging.info("Accuracy = {:.4}".format(acc))
    f1_male = mv_stats_f1(voting, 1)
    logging.info("F1 male = {:.4}".format(f1_male))
    f1_female = mv_stats_f1(voting, 0)
    logging.info("F1 female = {:.4}".format(f1_female))
    logging.info("----------------------------------")
    logging.info("")
    log_endtime = datetime.datetime.now()
    log_runtime = (log_endtime - log_starttime)
    logging.info("Total runtime: " + str(log_runtime))



def mv_verify(df, ids):
    cnt_inconsistencies = 0
    for userid in ids:
        if len(list(set(df.loc[df["userid"] == userid]["label"]))) > 1:
            logging.debug("No unique label assignment for user with ID {} !".format(userid))
            cnt_inconsistencies += 1
    return cnt_inconsistencies



def mv_compute(df, ids):
    df_mv = df.copy()
    df_mv["majority_voting"] = df["prediction"]
    cnt_unchanged_users = 0
    cnt_unchanged_reviews = 0
    for userid in ids:
        predictions = list(df.loc[df["userid"] == userid]["prediction"])
        if (predictions.count(0) != predictions.count(1)) and (len(predictions) > 1):
            mv = max(predictions, key=predictions.count)
            df_mv.loc[df_mv["userid"]==userid, "majority_voting"] = mv
        else:
            cnt_unchanged_users += 1
            cnt_unchanged_reviews += len(predictions)
    logging.info("Done! Majority voting rejected for {} review(s) ({:.2%}) of {} user(s) ({:.2%}).".format(
        cnt_unchanged_reviews, 
        cnt_unchanged_reviews/df.shape[0], 
        cnt_unchanged_users, 
        cnt_unchanged_users/len(ids))
    )
    logging.debug("Overview (Prediction vs. Majority Voting):")
    if logging.DEBUG >= logging.root.level:
        print(df_mv)
    return df_mv



def mv_stats_acc(df):
    # see https://en.wikipedia.org/wiki/Accuracy_and_precision
    males = list(df.loc[df["label"] == 1]["majority_voting"])
    females = list(df.loc[df["label"] == 0]["majority_voting"])
    cnt_pos = males.count(1) + females.count(0) 
    cnt_neg = males.count(0) + females.count(1)
    return cnt_pos/(cnt_neg+cnt_pos)



def mv_stats_f1(df, key_pos):
    # see https://en.wikipedia.org/wiki/F1_score
    pos = list(df.loc[df["label"] == key_pos]["majority_voting"])
    neg = list(df.loc[df["label"] == (1-key_pos)]["majority_voting"])
    TP = pos.count(key_pos)
    FP = neg.count(key_pos)
    FN = pos.count(1-key_pos)
    return TP/(TP+0.5*(FP+FN))



# Example where MV can improve accuracy for user 1 but skips user 2:
# mv(pd.DataFrame(data={"userid": [1, 1, 2, 1, 2], "label": [0, 0, 1, 0, 1], "prediction": [0, 0, 1, 1, 0]}), True)
# Example where MV gets corrupted input data, as user 1 has different label assignments:
# mv(pd.DataFrame(data={"userid": [1, 1, 2, 1, 2], "label": [1, 0, 1, 0, 1], "prediction": [0, 0, 1, 1, 0]}), True)