import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
    ) 

def preprocess(data):
    for col in data.columns:
        if ((col == 'num_dmgs') or (col == "fname") or (col == "cdn_url") or (col == "session") or (col == "time")):
            continue
        else:
            print(col)
            data[col] = data[col].apply(lambda x: eval(x))
    data['num_dmgs'] = data['gt_bboxes'].apply(lambda x: len(x))

    return data

def get_all_bboxes(row):
    pred_bboxes = row['pred_bboxes']
    method_lst = row['method_lst']
    qa_answers = row['qa_answers']
    qa_bboxes = row['qa_bboxes']
    
    new_bboxes = []
    for i, bbox in enumerate(pred_bboxes):
        new_bboxes.append(pred_bboxes)
    for i, bbox in enumerate(qa_bboxes):
        if qa_answers[i] == "yes":
            new_bboxes.append(bbox)
    return new_bboxes

def get_all_methods(row):
    pred_bboxes = row['pred_bboxes']
    method_lst = row['method_lst']
    qa_answers = row['qa_answers']
    qa_bboxes = row['qa_bboxes']
    
    all_methods = [x for x in method_lst]

    for i, bbox in enumerate(qa_bboxes):
        if qa_answers[i] == "yes":
            all_methods.append('c')
    return all_methods

def filter_conf(row, conf_threshold):
    pred_bboxes = row['pred_bboxes']
    pred_confs = row['pred_confs']
    filtered_preds = []
    for i in range(len(pred_bboxes)):
        if pred_confs[i] > conf_threshold:
            filtered_preds.append(pred_bboxes[i])
    return filtered_preds

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Parameters:
    - box1, box2: Bounding boxes in the format [x_min, y_min, x_max, y_max].

    Returns:
    - IoU value between box1 and box2.
    """
    # Intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # Areas of the bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Union
    union_area = box1_area + box2_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area if union_area != 0 else 0
    return iou

def combo_1(row):
    pred_bboxes = row['pred_bboxes']
    pred_confs = row['pred_confs']
    qa_answers = row['qa_answers']
    qa_bboxes = row['qa_bboxes']
    filtered_preds = []
    for i, pred in enumerate(pred_bboxes):
        if pred_confs[i] > 0.60:
            filtered_preds.append(pred)
        else:
            for j, qa_crop in enumerate(qa_bboxes):
                if calculate_iou(pred, qa_crop) > 0.2:
                    answer = qa_answers[j]
                    if answer == 'yes':
                        filtered_preds.append(pred)
                else:
                    continue
    return filtered_preds

def get_dino_bboxes(row, use_qa):
    pred_bboxes = row['pred_bboxes']
    method_lst = row['method_lst']
    dino_bboxes = []
    if len(method_lst) > 0:
        for i, method in enumerate(method_lst):
            if method == "b":
                dino_bboxes.append(pred_bboxes[i])
    
    if use_qa:
        qa_answers = row['dino_qa_answers']
        qa_bboxes = row['dino_qa_bboxes']
        final_bboxes = []
        for i in range(len(qa_answers)):
            if qa_answers[i] == 'yes':
                final_bboxes.append(dino_bboxes[i])
        return final_bboxes
    return dino_bboxes

if __name__ == "__main__":
    unique_folder_name = 'out_gd150k_0002_pali_dmg-qa'
    pali_folder = 'out_gd150k_0002_pali_dmg-qa-150k'
    #unique_folder_name = 'out_gd_0007_pali_dmg-qa'
    #load data
    path = f'results/{unique_folder_name}/final.csv'
    dino_bbox_data = pd.read_csv(f'/home/ubuntu/dmg-qa/tests/results/{pali_folder}/result_pali_combine_0.csv')
    
    #NOTE: Args:
    dino_filter = True


    data = pd.read_csv(path)
    data = data.drop_duplicates(subset="cdn_url")
    data.index = range(len(data))
    data = preprocess(data)
    
    data = data.merge(dino_bbox_data[['cdn_url','dino_qa_answers','dino_qa_bboxes']], on='cdn_url')
    data['dino_qa_answers'] = data['dino_qa_answers'].apply(lambda x: eval(x))
    data['dino_qa_bboxes'] = data['dino_qa_bboxes'].apply(lambda x: eval(x))
    #data = data[data['num_dmgs']==0]
    
    #Combine (det+cls) and qa bboxes
    data['all_bboxes'] = data.apply(get_all_bboxes, axis=1)
    data['all_method_lst'] = data.apply(get_all_methods, axis=1)

    lst = []
    for val in [0]:#[0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.60, 0.65, 0.70, 0.75]:
        data['filtered_bbox'] = data.apply(filter_conf, conf_threshold=val, axis=1)
        data['dino_bbox'] = data.apply(get_dino_bboxes,use_qa=dino_filter, axis=1)
        data['combo_1_bboxes'] = data.apply(combo_1, axis=1)
        
        data['is_damaged'] = data['num_dmgs'].apply(lambda x: 1 if x > 0 else 0) #GT
        data['pred_is_damaged'] = data['filtered_bbox'].apply(lambda x: 1 if len(x) > 0 else 0) #All preds
        data['dino_is_damaged'] = data['dino_bbox'].apply(lambda x: 1 if len(x) > 0 else 0) # Dino with or without QA
        #data['pred_combo_1'] = data['combo_1_bboxes'].apply(lambda x: 1 if len(x) > 0 else 0) #Pred combo 1
        data['qa_is_damaged'] = data['qa_answers'].apply(lambda x: 1 if 'yes' in x else 0) #Using only qa to filter
        gt_lst = data['is_damaged'].tolist()
        for attempt in ['pred_is_damaged','dino_is_damaged', 'qa_is_damaged']:
            print(attempt)
            pred_lst = data[attempt].tolist()
            from sklearn.metrics import confusion_matrix, accuracy_score
            tn, fp, fn, tp = confusion_matrix(gt_lst, pred_lst).ravel()
            accuracy = accuracy_score(gt_lst, pred_lst)
            precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0

            precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
            recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            print('Confidence threshold:', val)
            print('Acc',accuracy)
            print('Prec 1', precision_1,'Recall 1', recall_1)
            print('Prec 0', precision_0,'Recall 0', recall_0)
            print('----------------------')
            dct = {
                "threshold": val,
                "accuracy": accuracy,
                "precision_1": precision_1,
                "recall_1": recall_1,
                "precision_0": precision_0,
                "recall_0": recall_0
            }
            lst.append(dct)
    
    import ipdb;ipdb.set_trace()
    final = pd.DataFrame(lst)
    final.to_csv(f'results/{unique_folder_name}/metrics.csv', index=False)

