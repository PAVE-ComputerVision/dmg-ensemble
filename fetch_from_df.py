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

def evaluate_detections(pred_boxes, gt_boxes):
    data_per_predbox = dict()
    for i, pred_box in enumerate(pred_boxes):
        max_iou = 0
        min_dist = 1920
        gt_box_iou_id = -1
        gt_box_dist_id = -1
        for j, gt_box in enumerate(gt_boxes):
            iou = calculate_iou(pred_box, gt_box)
            distance = calculate_center_distance(pred_box, gt_box)
            print(i, j, iou, distance)
            if (iou >= max_iou):
                max_iou = iou
                gt_box_iou_id = j
            if (distance <= min_dist):
                min_dist = distance
                gt_box_dist_id = j

        lst1 = [max_iou, gt_box_iou_id, min_dist, gt_box_dist_id]
        data_per_predbox[i] = lst1
    print(data_per_predbox)
    print('---------------------')
    data_per_gtbox = dict()
    for i, gt_box in enumerate(gt_boxes):
        max_iou = 0
        min_dist = 1920
        pred_box_iou_id = -1
        pred_box_dist_id = -1
        for j, pred_box in enumerate(pred_boxes):
            iou = calculate_iou(pred_box, gt_box)
            distance = calculate_center_distance(pred_box, gt_box)
            print(i, j, iou, distance)
            if (iou >= max_iou):
                max_iou = iou
                pred_box_iou_id = j
            if (distance <= min_dist):
                min_dist = distance
                pred_box_dist_id = j
        lst2 = [max_iou, pred_box_iou_id, min_dist, pred_box_dist_id]
        data_per_gtbox[i] = lst2
    print(data_per_gtbox)
    return data_per_predbox, data_per_gtbox

def get_coco_bbox(kpts, h, w, dmg):
    
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
    output_dir = f"new_r8tr_test"
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
    correct = 0
    total = 0
    accumulated_dicts = []

    for idx, row in df.iterrows():
        session = row["Session Key"]
        for dct in tqdm(rater_pt):
            if len(dct["SessID"]) > 0:
                if dct['SessID'].item() == session:
                    for pc in ['04', '05', '07', '08']:
                        
                        print("here3")
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
                        
                        print("here2")
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
                        #its this line that was failing, 305
                        try:
                            crop_data = get_dmg_bboxes(image_id, pld_a, pld_b, session)
                        except:
                            crop_data = [] #does this maek sense/?
                        dmg_crop_bboxes, confidence_lst, method_lst = get_crop_bboxes(crop_data)                
                        
                        print("here1")
                        print(dmg_crop_bboxes)
                        print(method_lst)
                        
                        if len(dmg_crop_bboxes) != 0:
                            correct += 1
                        print(correct)
                        end = time.time()
                        time_per_req = (end-start)
                        total += time_per_req 
                        
                        opt = {}
                        opt['cdn_url'] = cdn
                        opt['fname'] = image_id
                        opt['session'] = session
                        opt['method_lst'] = method_lst
                        opt['gt_bboxes'] = gt_bboxes
                        opt['pred_confs'] = confidence_lst
                        opt['all_pred_bboxes'] = dmg_crop_bboxes
                        opt['time'] = time_per_req
                        accumulated_dicts.append(opt)
                        
                        print("here")
                        #if (idx + 1) % interval == 0:
                        result = pd.DataFrame(accumulated_dicts)
                        if not os.path.exists(f"results/{output_dir}"):
                            os.makedirs(f"results/{output_dir}")
                        path = f"results/{output_dir}/result_combine_0.csv"
                        result.to_csv(path, mode='a', header=not pd.io.common.file_exists(path), index=False)
                        accumulated_dicts.clear()
                    #except Exception as e:
                    #    error_message = f"Error processing item {cdn}: {e}"
                    #    print(error_message)
                    #    logging.error(error_message)
                    #    error_log.append(str(e))
                    #    #import ipdb;ipdb.set_trace()
                    #    continue
    print('Total time', total)
    files  = glob(f"results/{output_dir}/*.csv")
    for i, file in enumerate(files):
        x = pd.read_csv(file)
        if i == 0:
            final = x
        else:
            final = pd.concat([final,x])

    final = final.drop_duplicates(subset="cdn_url")
    #final['gt_bboxes'] = final['gt_bboxes'].apply(lambda x: eval(x))
    #final['all_pred_bboxes'] = final['all_pred_bboxes'].apply(lambda x: eval(x))
    #final['dino_bboxes'] = final['dino_bboxes'].apply(lambda x: eval(x))
    #final['num_gts'] = final['gt_bboxes'].apply(lambda x: len(x))
    #final['num_all_preds'] = final['all_pred_bboxes'].apply(lambda x: len(x))
    #final['num_dino_preds'] = final['dino_bboxes'].apply(lambda x: len(x))
    #final['qa_dmg_num'] = final['qa_answers'].apply(lambda x: 1 if 'yes' in x else 0)
    #final['final_dmg_cnt'] = final.apply(lambda x: x['qa_dmg_num']+x['num_all_preds'], axis=1)
    final.to_csv(f'results/{output_dir}/final.csv', index=False)

                        #data_per_predbox, data_per_gtbox = evaluate_detections(dmg_crop_bboxes, gt_bboxes)
                        #pld_c = get_payload_c(int(pc), cdn, dmg_crop_bboxes)
                        #result = get_dmg_assurance(image_id, pld_c)
                        #qa_answers = result['answers']
                        #save(image_id, result, output_dir)
                        #qa_ans_np = np.array(qa_answers)
                        #qa_bboxes = result['bboxes']
                        #for i, bbox in enumerate(qa_bboxes):
                        #    if qa_answers[i] == "yes":
                        #        dmg_crop_bboxes.append(bbox)
                        #        x_min,y_min,x_max,y_max = bbox
                        #        crop_data.append({
                        #            'image_id': image_id,
                        #            'session': session,
                        #            'method': "c",
                        #            'box_index': 0,
                        #            'x_min': x_min,
                        #            'y_min': y_min,
                        #            'x_max': x_max,
                        #            'y_max': y_max,
                        #            'confidence': -1
                        #        })
                        #df = pd.DataFrame(crop_data)
                        #print(crop_data)
                        #df.to_csv(os.path.join(output_dir, f"{image_id}_crop_data.csv"), index=False)
                        #print(f"Processed and saved crop data for image_id: {image_id}")
                        
                        #filtdata_per_predbox, filtdata_per_gtbox = evaluate_detections(dmg_crop_bboxes, gt_bboxes)

                        #all_names_lst.append(cdn)
                        #all_gts.append(gt_bboxes)
                        #all_damage_names.append(damage_name_lst)
                        #all_components.append(comp_lst)
                        #all_kpts.append(kpt_lst)
                        #all_preds.append(dmg_crop_bboxes)
                        #all_metrics_per_pred.append(data_per_predbox)
                        #all_confidences.append(confidence_lst)
                        #all_qa_answers.append(qa_answers)
                        #all_metrics_per_gt.append(data_per_gtbox)
                        #filtered_per_gt.append(filtdata_per_gtbox)
                        #filtered_per_pred.append(filtdata_per_predbox)
    #dct_ = dict()
    #dct_["fname"] = all_names_lst
    #dct_["gt_bbox"] = all_gts
    #dct_["damage_names"] = all_damage_names
    #dct_["comp_names"] = all_components
    #dct_["dmg_kpts"] = all_kpts
    #dct_["pred_bbox"] = all_preds
    #dct_["metrics_per_pred"] = all_metrics_per_pred
    #dct_["confidences"] = all_confidences
    #dct_["qa_answers"] = all_qa_answers
    #dct_["metrics_per_gt"] = all_metrics_per_gt
    #dct_['filtered_per_pred'] = filtered_per_pred
    #dct_['filtered_per_gt'] = filtered_per_gt
    #result = pd.DataFrame(dct_)
    #result.to_csv(f"results/result_pali_{unique_folder_name}_combine.csv")
