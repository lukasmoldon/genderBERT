import logging
import datetime
from datetime import date, timedelta
import pandas as pd



def mv(df):
    logging.basicConfig(format='%(asctime)s [%(levelname)s] - %(message)s', datefmt='%d-%m-%y %H:%M:%S', level=logging.INFO)
    log_starttime = datetime.datetime.now()
    logging.info("Computing majority voting..")
    ids = df["userid"].unique()
    voting = mv_compute(df, ids)
    logging.info("Computing majority voting stats..")
    logging.info("")
    logging.info("----------------------------------")
    acc = mv_stats_acc(voting, ids)
    logging.info("Accuracy = {:.4}".format(acc))
    f1_male = mv_stats_f1(voting, ids, 1)
    logging.info("F1 male = {:.4}".format(f1_male))
    f1_female = mv_stats_f1(voting, ids, 0)
    logging.info("F1 female = {:.4}".format(f1_female))
    logging.info("----------------------------------")
    logging.info("")
    log_endtime = datetime.datetime.now()
    log_runtime = (log_endtime - log_starttime)
    logging.info("Total runtime: " + str(log_runtime))



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
    return df_mv



def mv_stats_acc(df, ids):
    cnt_pos = 0
    cnt_neg = 0
    for userid in ids:
        truth = list(df.loc[df["userid"] == userid]["label"])[0]
        majority_voting = list(df.loc[df["userid"] == userid]["majority_voting"])
        cnt_pos += majority_voting.count(truth)
        cnt_neg += majority_voting.count(1-truth)
    return cnt_pos/(cnt_neg+cnt_pos)



def mv_stats_f1(df, ids, key_pos):
    TP = 0
    FP = 0
    FN = 0
    for userid in ids:
        truth = list(df.loc[df["userid"] == userid]["label"])[0]
        majority_voting = list(df.loc[df["userid"] == userid]["majority_voting"])
        if truth == key_pos:
            TP += majority_voting.count(truth)
            FN += majority_voting.count(1-truth)
        else:
            FP += majority_voting.count(1-truth)
    return TP/(TP+0.5*(FP+FN))



# mv(pd.DataFrame(data={"userid": [1, 1, 2, 1, 2], "label": [0, 0, 1, 0, 1], "prediction": [0, 0, 1, 0, 0]}))