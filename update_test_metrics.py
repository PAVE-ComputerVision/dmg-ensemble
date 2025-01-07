import argparse
import time
import os
import math
import json
import torch
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from fetch_and_crop import adjust_bbox_to_min_size, fetch_data, get_dmg_bboxes, get_dmg_assurance  

cage_lookup_df = pd.read_csv("amazon_cages_2_0.csv")

def get_cage(vin, pc):
    wmi = vin[:3]
    model_year = vin[9]
    vehicle_desc = vin[3:8]
    correct_cage_row = cage_lookup_df[(cage_lookup_df["Manufacturer Code"] == wmi) &
                (cage_lookup_df["Model Year Code"] == model_year) &
                (cage_lookup_df["Model Code"] == vehicle_desc)]
    if correct_cage_row.shape[0] > 1:
        correct_cage_row = correct_cage_row.head(1) 
    try:
        if pc == 4 or pc == 10:
            svg_url = correct_cage_row["Cage: Left_View"].item()
        elif pc == 5 or pc == 11:
            svg_url = correct_cage_row["Cage: Front_View"].item()
        elif pc == 7 or pc == 12:
            svg_url = correct_cage_row["Cage: Right_View"].item()
        elif pc == 8 or pc ==13:
            svg_url = correct_cage_row["Cage: Rear_View"].item() 
    except:
        print('no svg found')
        return ""
    return svg_url

def save(image_id, result, output_dir):
    out = os.path.join(output_dir, f"{image_id}_results.json")
    with open(out, "w") as f:
        json.dump(result, f)

def get_payload_a(vin, cdn, pc, svg_url):
    return {"vin": vin, 
            "bucket": "cdn", 
            "file": cdn, 
            "photocode": pc, 
            "date": '20240420', 
            "svg": svg_url}

def get_payload_b(pc, cdn, svg_url):
    return {"photocode": pc,
             "file": cdn,
             "bucket":  "cdn",
             "svg": svg_url}

def get_crop_bboxes(crop_data):
    dmg_crop_bboxes = []
    conf_lst = []
    method_lst = []
    for i, data in enumerate(crop_data):
        x1 = data['x_min']
        y1 = data['y_min']
        x2 = data['x_max']
        y2 = data['y_max']
        dmg_crop_bboxes.append([x1,y1,x2,y2])
        conf_lst.append(data['confidence'])
        method_lst.append(data['method'])
    return dmg_crop_bboxes, conf_lst, method_lst

def get_payload_c(pc, cdn, bboxes):
    if bboxes == []:
        bboxes = None
    return {"pc": pc,
            "file": cdn,
            "bucket": "cdn"
            #"bboxes": bboxes
            }


def get_gt_dmg(items):
    comp_lst = []
    damage_name_lst = []
    kpt_lst = []
    group_lst = []
    for item in items:
         comp = item["component"]
         damage_name = item["damage_name"]
         kpt = item["coodrs"]
         group = item["damage_group"]
         comp_lst.append(comp)
         damage_name_lst.append(damage_name)
         kpt_lst.append(kpt)
         group_lst.append(group)
    return comp_lst, damage_name_lst, kpt_lst, group_lst

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

def calculate_center_distance(box1, box2):
    """
    Calculate the Euclidean distance between the centers of two bounding boxes.

    Parameters:
    - box1, box2: Bounding boxes in the format [x_min, y_min, x_max, y_max].

    Returns:
    - Distance between the centers of box1 and box2.
    """
    center1_x = (box1[0] + box1[2]) / 2
    center1_y = (box1[1] + box1[3]) / 2
    center2_x = (box2[0] + box2[2]) / 2
    center2_y = (box2[1] + box2[3]) / 2
    return math.sqrt((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2)

from scipy.optimize import linear_sum_assignment
def evaluate_detections_v2(pred_boxes, gt_boxes):
    """
    Perform matching of key predictions to ground truth detections.
    For each ground truth bbox
        Find the best matching prediction according to IoU and according to distance.
        Store that prediction with [iou value, gt_box_id, dist value, gt_dist_id]
    
    The optimal one-to-one matching between preds and gts is realized by solving the
    assignment problem with the Hungarian Algorithm.
    This is done by creating a cost matrix for IoU and distance, and then solving
    two separate assignment problems: one minimizing (1 - IoU), and one minimizing distance.
    
    Returns:
        assignments_iou: List of dicts with keys {gt_id, pred_id, iou}
        assignments_dist: List of dicts with keys {gt_id, pred_id, dist}
    """
    n_gt = len(gt_boxes)
    n_pred = len(pred_boxes)
    
    # Initialize cost matrices
    # Rows = ground truths, Columns = predictions
    cost_matrix_iou = np.zeros((n_gt, n_pred), dtype=float)
    cost_matrix_dist = np.zeros((n_gt, n_pred), dtype=float)
    data_per_predbox = dict()
    for i, gt_box in enumerate(gt_boxes):
        for j, pred_box in enumerate(pred_boxes):
            iou = calculate_iou(pred_box, gt_box)
            dist = calculate_center_distance(pred_box, gt_box)
            # For IoU-based matching:
            # We want to maximize IoU, but Hungarian solves a minimization problem.
            # Convert IoU to cost by using cost = 1 - IoU.
            cost_matrix_iou[i, j] = 1 - iou

            # For distance-based matching:
            # Distance is already a measure we want to minimize, so we can use it directly.
            cost_matrix_dist[i, j] = dist

    # Solve the assignment problem for IoU-based cost
    gt_indices_iou, pred_indices_iou = linear_sum_assignment(cost_matrix_iou)
    # Solve the assignment problem for distance-based cost
    gt_indices_dist, pred_indices_dist = linear_sum_assignment(cost_matrix_dist)

    # Prepare the results for IoU-based matching
    assignments_iou = []
    for gt_id, pred_id in zip(gt_indices_iou, pred_indices_iou):
        # Recall: cost_matrix_iou[gt_id, pred_id] = 1 - IoU
        iou_value = 1 - cost_matrix_iou[gt_id, pred_id]
        assignments_iou.append({
            "gt_id": gt_id,
            "pred_id": pred_id,
            "iou": iou_value
        })
    
    # Prepare the results for distance-based matching
    assignments_dist = []
    for gt_id, pred_id in zip(gt_indices_dist, pred_indices_dist):
        # cost_matrix_dist[gt_id, pred_id] = distance
        dist_value = cost_matrix_dist[gt_id, pred_id]
        assignments_dist.append({
            "gt_id": gt_id,
            "pred_id": pred_id,
            "dist": dist_value
        })

    import ipdb;ipdb.set_trace()    
    return data_per_predbox, data_per_gtbox


def evaluate_detections(pred_boxes, gt_boxes):
    """
    Perform matching of key predictions to ground truth detections.
    For each ground truth bbox
        Find the best matching prediction according to IoU and according to distance.
        Store that prediction with [iou value, gt_box_id, dist value, gt_dist_id]


    """
    data_per_predbox = dict()
    for i, pred_box in enumerate(pred_boxes):
        max_iou = 0
        min_dist = 1920
        gt_box_iou_id = -1
        gt_box_dist_id = -1
        for j, gt_box in enumerate(gt_boxes):
            iou = calculate_iou(pred_box, gt_box)
            distance = calculate_center_distance(pred_box, gt_box)
            #print(i, j, iou, distance)
            if (iou >= max_iou):
                max_iou = iou
                gt_box_iou_id = j
            if (distance <= min_dist):
                min_dist = distance
                gt_box_dist_id = j

        lst1 = [max_iou, gt_box_iou_id, min_dist, gt_box_dist_id]
        data_per_predbox[i] = lst1
    #print(data_per_predbox)
    #print('---------------------')
    data_per_gtbox = dict()
    for i, gt_box in enumerate(gt_boxes):
        max_iou = 0
        min_dist = 1920
        pred_box_iou_id = -1
        pred_box_dist_id = -1
        for j, pred_box in enumerate(pred_boxes):
            iou = calculate_iou(pred_box, gt_box)
            distance = calculate_center_distance(pred_box, gt_box)
            #print(i, j, iou, distance)
            if (iou >= max_iou):
                max_iou = iou
                pred_box_iou_id = j
            if (distance <= min_dist):
                min_dist = distance
                pred_box_dist_id = j
        lst2 = [max_iou, pred_box_iou_id, min_dist, pred_box_dist_id]
        data_per_gtbox[i] = lst2
    #print(data_per_gtbox)
    return data_per_predbox, data_per_gtbox

def get_coco_bbox(kpts, h, w, dmg):
    """
    xmin ymin xmax ymax
    """
    # Convert kpts from XY to XYWH
    kpt_x, kpt_y = kpts[0] * w, kpts[1] * h
    small = 8*2
    med = 16*2
    big = 32*2
    #change 3
    if dmg == 'small':
        x1, y1 = kpt_x-small, kpt_y-small
        x2, y2 = kpt_x+small, kpt_y+small
    elif dmg == 'medium':
        x1, y1 = kpt_x-med, kpt_y-med
        x2, y2 = kpt_x+med, kpt_y+med
    elif dmg == 'large':
        x1, y1 = kpt_x-big, kpt_y-big
        x2, y2 = kpt_x+big, kpt_y+big
    else:
        x1, y1 = kpt_x-small, kpt_y-small
        x2, y2 = kpt_x+small, kpt_y+small

    bbox = [x1,y1,x2,y2]
    bbox = [round(val,1) for val in bbox]

    return bbox

def construct_gt_bbox(dmg_name_lst, kpts, height, width):
    
    kpts = [(kpt["x"], kpt["y"]) for kpt in kpts]

    scaled_gt_bbox_lst = []
    for (cat, kpt) in zip(dmg_name_lst, kpts):
        #Bbox size categories
        if 'MAJOR' in cat:
            size_cat = 'large'
        elif 'MEDIUM' in cat:
            size_cat = 'medium'
        elif 'MINOR' in cat:
            size_cat = 'small'
        else:
            size_cat = 'small'
        bbox = get_coco_bbox(kpt, height, width, size_cat)
        scaled_gt_bbox_lst.append(bbox)
    return scaled_gt_bbox_lst

if __name__ == "__main__":
    output_dir = f"r8tr_test_250103_bbthresh_090"
    df = pd.read_csv('dmg_test_r8tr.csv')
    df = df.loc[:, df.columns != "Unnamed: 0"] #Drop the first col, its the idx 
    df.columns = df.iloc[0]  # Use the first row as column names
    df = df[1:].reset_index(drop=True)  # Drop the first row and reset index
    
    #missing = torch.load("MISSING_R8TR_SESSIONS.pt") 
    #seven = torch.load("THE17_R8TR_SESSIONS.pt")
    #rater = missing | seven
    
    #lst = []
    #for item in rater[session][2]["detected_damages"]:
    #    if item["photo"]["code"] == pc:
    #        lst.append(item)

    rater_pt = torch.load("R8TR_ALL_INFO.pt")

    all_names_lst, all_gts, all_preds = [], [], []
    all_metrics_per_pred, all_metrics_per_gt = [], []
    filtered_per_pred, filtered_per_gt = [], []
    all_damage_names, all_components, all_kpts = [], [], []
    all_confidences = []
    all_qa_answers = []
    all_times = []
    correct = 0
    total = 0
    accumulated_dicts = []
    for idx, row in df.iterrows():
        session = row["Session Key"]
        for dct in tqdm(rater_pt):
            if len(dct["SessID"]) > 0:
                if dct['SessID'].item() == session:
                    for pc in ['04', '05', '07', '08']:
                        
                        photo_lst = json.loads(dct["photo_lst"].item())
                        damage_name_lst = json.loads(dct["damage_name_lst"].item())
                        comp_lst = json.loads(dct["component_lst"].item())
                        kpt_lst = json.loads(dct["kp_lst"].item())
                        
                        idxs = [i for i in range(len(photo_lst)) if int(photo_lst[i]['code']) == int(pc)]
                        kpt_lst = [kpt_lst[i] for i in idxs]
                        damage_name_lst = [damage_name_lst[i] for i in idxs]
                        comp_lst = [comp_lst[i] for i in idxs]

                        #comp_lst, damage_name_lst, kpt_lst, group_lst = get_gt_dmg(lst)

                        #Construct list of gt bboxes
                        gt_bboxes = construct_gt_bbox(damage_name_lst, kpt_lst, 1080, 1920)   
                        
                        pc = str(pc).zfill(2)
                        cdn = dct[f"PhotoCode_{int(pc)}"].item()  
                        
                        print(cdn)
                        
                        #try:
                        #    cdn = dct["original_images"][pc][0]
                        #except KeyError:
                        #    print(f"no image for session {session} and pc {pc}")
                        #    save(session + f"_{pc}", {"pc": pc, "image": None})
                        #    continue

                        vin = row["Vin"]
                        svg_url = get_cage(vin, int(pc))
                        
                        if svg_url == "":
                            print(f"no image for session {session} and pc {pc}")
                            save(session + f"_{pc}", {"pc": pc, "image": None}, output_dir)
                            continue
                        pld_a = get_payload_a(vin, cdn, int(pc), svg_url)
                        pld_b = pld_a
                        start = time.time()
                        #pld_b = get_payload_b(int(pc), cdn, svg_url)
                        image_id = cdn.split('/')[-1].split('.')[0]
                        crop_data, version = get_dmg_bboxes(image_id, pld_a, pld_b, session)
                        #try:
                        dmg_crop_bboxes, confidence_lst, method_lst = get_crop_bboxes(crop_data)                
                        #except:
                        #    dmg_crop_bboxes = []
                        #    confidence_lst = []
                        #    method_lst = []

                        print(dmg_crop_bboxes)
                        print(method_lst)
                        end = time.time()
                        #data_per_predbox, data_per_gtbox = evaluate_detections_v2(dmg_crop_bboxes, gt_bboxes)
                        data_per_predbox, data_per_gtbox = evaluate_detections(dmg_crop_bboxes, gt_bboxes)
                        save(image_id, pld_b, output_dir)
                        df = pd.DataFrame(crop_data)
                        #print(crop_data)
                        df.to_csv(os.path.join(output_dir, f"{image_id}_crop_data.csv"), index=False)
                        print(f"Processed and saved crop data for image_id: {image_id}")
                        
                        if len(all_times) > 0:
                            print(sum(all_times)/len(all_times))
                        all_names_lst.append(cdn)
                        all_gts.append(gt_bboxes)
                        all_damage_names.append(damage_name_lst)
                        all_components.append(comp_lst)
                        all_kpts.append(kpt_lst)
                        all_preds.append(dmg_crop_bboxes)
                        all_metrics_per_pred.append(data_per_predbox)
                        all_confidences.append(confidence_lst)
                        all_metrics_per_gt.append(data_per_gtbox)
                        all_times.append(end-start)
                        
    dct_ = dict()
    dct_["version"] = version
    dct_["fname"] = all_names_lst
    dct_["gt_bbox"] = all_gts
    dct_["damage_names"] = all_damage_names
    dct_["comp_names"] = all_components
    dct_["dmg_kpts"] = all_kpts
    dct_["pred_bbox"] = all_preds
    dct_["metrics_per_pred"] = all_metrics_per_pred
    dct_["confidences"] = all_confidences
    dct_["metrics_per_gt"] = all_metrics_per_gt
    dct_["time"] = all_times
    result = pd.DataFrame(dct_)
    result.to_csv(f"results/result_pali_{output_dir}_combine.csv", index=False)
