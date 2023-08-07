from util.data import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score


def get_full_err_scores(np_test_result_recons, np_test_result_pred, np_test_ground_recons, np_test_ground_pred, test_labels):

    np_test_result_recons = np.array(np_test_result_recons)
    np_test_ground_recons = np.array(np_test_ground_recons)
    np_test_result_pred = np.array(np_test_result_pred)
    np_test_ground_pred = np.array(np_test_ground_pred)


    recons_all_scores =  None
    preds_all_scores = None

    feature_num = np_test_result_recons.shape[1]

    labels = test_labels

    for i in range(feature_num):

        # for j in range(0,10500,100):
        #     plt.plot(np_test_ground_recons[j,i,:])
        #     plt.plot(np_test_result_recons[j,i,:])
        #     plt.show()

        recons_scores = get_err_scores_recons(np_test_result_recons[:,i,:], np_test_ground_recons[:,i,:])

        pred_scores = get_err_scores_pred(np_test_result_pred[:,i], np_test_ground_pred[:,i])

        if recons_all_scores is None:
            recons_all_scores = recons_scores
            pred_all_scores = pred_scores
        else:
            recons_all_scores = np.vstack((
                recons_all_scores,
                recons_scores
            ))
            pred_all_scores = np.vstack((
                pred_all_scores,
                pred_scores
            ))

    return recons_all_scores, pred_all_scores


def get_err_scores_recons(test_predict, test_gt):

    length=len(test_predict)
    win=len(test_predict[0])

    err_inter = np.zeros(length+win-1)
    count_inter = np.zeros(length+win-1)
    count_arr = np.ones_like(test_predict)

    test_delta = np.abs(np.subtract(
                        np.array(test_predict).astype(np.float64), 
                        np.array(test_gt).astype(np.float64)
                    ))

    # n_err_mid, n_err_iqr = get_err_median_and_iqr(test_delta)

    # test_delta=test_delta.sum(axis=1)##Recons

    for i in range(length):
        err_inter[i:i+win]+=test_delta[i]
        count_inter[i:i+win]+=count_arr[i]

    test_delta = err_inter/count_inter

    n_err_mid, n_err_iqr = get_err_median_and_iqr(test_delta)

    epsilon=1e-2

    err_scores = (test_delta - n_err_mid) / ( np.abs(n_err_iqr) +epsilon)

    smoothed_err_scores = np.zeros(err_scores.shape)
    before_num = 0
    for i in range(before_num, len(err_scores)):
        smoothed_err_scores[i] = np.mean(err_scores[i-before_num:i+1])
   
    return smoothed_err_scores


def get_err_scores_pred(test_predict, test_gt):

    test_delta = np.abs(np.subtract(
                        np.array(test_predict).astype(np.float64), 
                        np.array(test_gt).astype(np.float64)
                    ))

    # plt.plot(test_gt)
    # plt.plot(test_predict)
    # plt.show()

    n_err_mid, n_err_iqr = get_err_median_and_iqr(test_delta)

    epsilon=1e-2

    err_scores = (test_delta - n_err_mid) / ( np.abs(n_err_iqr) +epsilon)

    smoothed_err_scores = np.zeros(err_scores.shape)##Pred
    before_num = 3
    for i in range(before_num, len(err_scores)):
        smoothed_err_scores[i] = np.mean(err_scores[i-before_num:i+1])
   
    return smoothed_err_scores


def get_loss(predict, gt):
    return eval_mseloss(predict, gt)

def get_f1_scores(total_err_scores, gt_labels, topk=1):
    # remove the highest and lowest score at each timestep
    total_features = total_err_scores.shape[0]

    # topk_indices = np.argpartition(total_err_scores, range(total_features-1-topk, total_features-1), axis=0)[-topk-1:-1]
    topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]
    
    topk_indices = np.transpose(topk_indices)

    total_topk_err_scores = []
    topk_err_score_map=[]
    # topk_anomaly_sensors = []

    for i, indexs in enumerate(topk_indices):
       
        sum_score = sum( score for k, score in enumerate(sorted([total_err_scores[index, i] for j, index in enumerate(indexs)])) )

        total_topk_err_scores.append(sum_score)

    final_topk_fmeas = eval_scores(total_topk_err_scores, gt_labels, 400)

    return final_topk_fmeas

def get_val_performance_data(total_err_scores, normal_scores, gt_labels, topk=1):
    total_features = total_err_scores.shape[0]

    topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]

    total_topk_err_scores = []
    topk_err_score_map=[]

    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)

    thresold = np.max(normal_scores)

    pred_labels = np.zeros(len(total_topk_err_scores))
    pred_labels[total_topk_err_scores > thresold] = 1

    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    print(np.where(pred_labels==1)[0])

    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)

    f1 = f1_score(gt_labels, pred_labels)


    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)

    return f1, pre, rec, auc_score, thresold


def get_scores_topk(total_err_scores, topk=1):

    total_features = total_err_scores.shape[0]

    # topk_indices = np.argpartition(total_err_scores, range(total_features-1-topk, total_features-1), axis=0)[-topk-1:-1]
    topk_indices = np.argpartition(total_err_scores, range(total_features-topk-1, total_features), axis=0)[-topk:]                               ##topk indices in each timestamp with high err here top1

    total_topk_err_scores = []
    topk_err_score_map=[]

    total_topk_err_scores = np.sum(np.take_along_axis(total_err_scores, topk_indices, axis=0), axis=0)                                          ##add err scores if topk here err score is err itself

    return total_topk_err_scores


def get_best_performance_data(total_topk_err_scores, gt_labels):

    final_topk_fmeas ,thresolds = eval_scores(total_topk_err_scores, gt_labels, 400, return_thresold=True)                                      ##f1 scores and corresponding thresholds

    th_i = final_topk_fmeas.index(max(final_topk_fmeas))                                                                                        ##max f1 score and correspondng index
    thresold = thresolds[th_i]

    pred_labels = np.zeros(len(total_topk_err_scores))                                                                                          ##all indices above threshold positive
    pred_labels[total_topk_err_scores > thresold] = 1

    for i in range(len(pred_labels)):
        pred_labels[i] = int(pred_labels[i])
        gt_labels[i] = int(gt_labels[i])

    # print(np.where(pred_labels==1)[0])

    prec_bfr = precision_score(gt_labels, pred_labels)
    rec_bfr = recall_score(gt_labels, pred_labels)

    # detection adjustment
    anomaly_state = False
    for i in range(len(gt_labels)):
        if gt_labels[i] == 1 and pred_labels[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt_labels[j] == 0:
                    break
                else:
                    if pred_labels[j] == 0:
                        pred_labels[j] = 1
            for j in range(i, len(gt_labels)):
                if gt_labels[j] == 0:
                    break
                else:
                    if pred_labels[j] == 0:
                        pred_labels[j] = 1
        elif gt_labels[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred_labels[i] = 1

    pred_labels = np.array(pred_labels)
    gt_labels = np.array(gt_labels)

    # print(topk_indices.squeeze()[np.where(pred_labels==1)[0]])

    pre = precision_score(gt_labels, pred_labels)
    rec = recall_score(gt_labels, pred_labels)

    auc_score = roc_auc_score(gt_labels, total_topk_err_scores)

    return f1_score(gt_labels, pred_labels), max(final_topk_fmeas),prec_bfr, rec_bfr, pre, rec, auc_score, thresold

