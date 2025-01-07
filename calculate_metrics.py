import numpy as np
import pandas as pd
from torchvision.ops import nms

def preprocess(data):
    for col in data.columns:
        if ((col == 'num_dmgs') or (col == "fname") or (col == "time") or (col == "version")):
            continue
        else:
            print(col)
            data[col] = data[col].apply(lambda x: eval(x))
    data['num_dmgs'] = data['gt_bbox'].apply(lambda x: len(x))
    return data

def get_tp(row, col_name, iou_thresh, dist_thresh):
    tp = 0
    metrics_per_pred = row[col_name]
    for pred_id, gt_values in metrics_per_pred.items():
        iou = gt_values[0]
        iou_id = gt_values[1]
        dist = gt_values[2]
        dist_id = gt_values[3]

        if ((iou >= iou_thresh) or (dist <= dist_thresh)):
            tp += 1
    return tp

def get_fp(row, col_name, iou_thresh, dist_thresh):
    fp = 0
    metrics_per_pred = row[col_name]
    for pred_id, gt_values in metrics_per_pred.items():
        iou = gt_values[0]
        iou_id = gt_values[1]
        dist = gt_values[2]
        dist_id = gt_values[3]

        if ((iou < iou_thresh) and (dist > dist_thresh)):
            fp += 1
    return fp

def get_fn(row, col_name, iou_thresh, dist_thresh):
    fp = 0
    metrics_per_gt = row[col_name]
    for gt_id, pred_values in metrics_per_gt.items():
        iou = pred_values[0]
        iou_id = pred_values[1]
        dist = pred_values[2]
        dist_id = pred_values[3]

        if ((iou < iou_thresh) and (dist > dist_thresh)):
            fp += 1
    return fp

def get_acc(row):
    tp = row['tp']
    fp = row['fp']
    tn = row['tn']
    fn = row['fn']
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    #accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    return accuracy

def get_spec(row):
    tp = row['tp']
    fp = row['fp']
    tn = row['tn']
    fn = row['fn']
    specificity = tn/ (tn + fp) if (tn + fp) > 0 else 0
    return specificity

def get_prec(row):
    tp = row['tp']
    fp = row['fp']
    tn = row['tn']
    fn = row['fn']
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0
    return precision

def get_rec(row):
    tp = row['tp']
    fp = row['fp']
    tn = row['tn']
    fn = row['fn']
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0
    return recall

def get_acc_qa(row):
    tp = row['tp_qa']
    fp = row['fp_qa']
    tn = row['tn_qa']
    fn = row['fn_qa']
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    #accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    return accuracy

def get_spec_qa(row):
    tp = row['tp_qa']
    fp = row['fp_qa']
    tn = row['tn_qa']
    fn = row['fn_qa']
    specificity = tn/ (tn + fp) if (tn + fp) > 0 else 0
    return specificity

def get_prec_qa(row):
    tp = row['tp_qa']
    fp = row['fp_qa']
    tn = row['tn_qa']
    fn = row['fn_qa']
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0
    return precision

def get_rec_qa(row):
    tp = row['tp_qa']
    fp = row['fp_qa']
    tn = row['tn_qa']
    fn = row['fn_qa']
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0
    return recall
if __name__ == "__main__":
    unique_folder_name = 'r8tr_test_250103_bbthresh_090'
    #unique_folder_name = '50k'
    #load data
    path = f'results/result_pali_{unique_folder_name}_combine.csv'
    data = pd.read_csv(path)
    data = preprocess(data)
    iou_threshs=[0.5]
    dist_thresh=400#[50,100,150,200,250,300,350,400]
    
    #print(data['num_dmgs'].value_counts())
    #data = data[data['num_dmgs'] <= 15]
    for iou_thresh in iou_threshs:
        data['tp'] = data.apply(get_tp, col_name = "metrics_per_pred", iou_thresh=iou_thresh, dist_thresh=dist_thresh, axis=1)
        data['fp'] = data.apply(get_fp, col_name = "metrics_per_pred", iou_thresh=iou_thresh, dist_thresh=dist_thresh, axis=1)
        data['tn'] = 10
        data['fn'] = data.apply(get_fn, col_name = "metrics_per_gt", iou_thresh=iou_thresh, dist_thresh=dist_thresh, axis=1)
        data['accuracy'] = data.apply(get_acc, axis=1)
        data['specificity'] = data.apply(get_spec, axis=1)
        data['precision'] = data.apply(get_prec, axis=1)
        data['recall'] = data.apply(get_rec, axis=1)
    
#        data['tp_qa'] = data.apply(get_tp, col_name = "filtered_per_pred", iou_thresh=iou_thresh, dist_thresh=dist_thresh, axis=1)
#        data['fp_qa'] = data.apply(get_fp, col_name = "filtered_per_pred", iou_thresh=iou_thresh, dist_thresh=dist_thresh, axis=1)
#        data['tn_qa'] = 10
#        data['fn_qa'] = data.apply(get_fn, col_name = "filtered_per_gt", iou_thresh=iou_thresh, dist_thresh=dist_thresh, axis=1)
#        data['accuracy_qa'] = data.apply(get_acc_qa, axis=1)
#        data['specificity_qa'] = data.apply(get_spec_qa, axis=1)
#        data['precision_qa'] = data.apply(get_prec_qa, axis=1)
#        data['recall_qa'] = data.apply(get_rec_qa, axis=1)
        
        print('IOU threshold:', iou_thresh)
        print('Distance threshold:', dist_thresh)
        print('Accuracy', data['accuracy'].mean())
        print('Specificity', data['specificity'].mean())
        print('Precision', data['precision'].mean())
        print('Recall', data['recall'].mean())
        import ipdb;ipdb.set_trace()
        data.to_csv(f'results/{unique_folder_name}/metrics.csv')
        
#        print('Accuracy after QA', data['accuracy_qa'].mean())
#        print('Specificity after QA', data['specificity_qa'].mean())
#        print('Precision after QA', data['precision_qa'].mean())
#        print('Recall after QA', data['recall_qa'].mean())
    import ipdb;ipdb.set_trace()
