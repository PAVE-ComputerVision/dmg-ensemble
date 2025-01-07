import time
import os
import math
import json
import torch
import logging
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import sys
from fetch_and_crop import adjust_bbox_to_min_size, fetch_data, get_dmg_bboxes, get_dmg_assurance  

cage_lookup_df = pd.read_csv("amazon_cages_2_0.csv")

def save(image_id, result, output_dir):
    out = os.path.join(output_dir, f"{image_id}_results.json")
    with open(out, "w") as f:
        json.dump(result, f)

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
    
    #kpts = [(kpt["x"], kpt["y"]) for kpt in kpts]

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

def get_payload_a(vin, cdn, pc, svg_url):
    return {"vin": vin, 
            "bucket": "cdn", 
            "file": cdn, 
            "photocode": pc, 
            "date": '20240420', 
            "svg": svg_url}

def get_payload_b(pc, cdn):
    return {"photocode": pc,
             "file": cdn,
             "bucket":  "cdn",
             "svg": None}

def get_payload_c(pc, cdn, bboxes):
    return {"pc": pc,
            "file": cdn,
            "bucket": "cdn"
            #"bboxes": bboxes
            }

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_id", type=int, default=0)
    parser.add_argument("--chunks", type=int, default=1)
    args = parser.parse_args()
    
    #output_dir = f"out_gd150k_0002_pali_dmg-qa_undamagedsheet_combine"
    output_dir = f"new_undamagedsheet"
    
    df = pd.read_csv('/home/ubuntu/AI_damage_detection_consistency_Nodamages_Compare_from_V10.csv')
    nodmg_csv = pd.read_csv('/home/ubuntu/AI_damage_detection_consistency_Nodamages_Compare.csv')
    df = df.merge(nodmg_csv[["VIN", "PRODUCT"]], left_on='SessID', right_on='PRODUCT')
    df = df.iloc[int(args.chunk_id)::int(args.chunks)]
    total = 0
    correct = 0
    interval = 10
    error_log = []
    accumulated_dicts = []
    no_vin_lst = []
    with tqdm(total=df.shape[0]) as pbar:
        for idx, (_, row) in enumerate(df.iterrows()):
            try:
                session = row["SessionKey"]
                for pc in ['04', '05', '07', '08']:
                    
                    #Load data from csv
                    pc = str(pc).zfill(2)
                    cdn = row[f"PhotoCode_{int(pc)}"]
                    print(cdn)
                    if cdn == None:
                        continue
                    
                    image_id = cdn.split('/')[-1].split('.')[0]
                    damage_name_lst = []
                    comp_lst = []
                    kpt_lst = []
                    photo_lst = []
                        

                    if len(kpt_lst) > 0:
                        gt_bboxes = construct_gt_bbox(damage_name_lst, kpt_lst, 1080, 1920)   
                    else:
                        gt_bboxes = []

                    #Process lists
                    idxs = [i for i in range(len(photo_lst)) if int(photo_lst[i]['code']) == int(pc)]
                    if len(idxs) > 0:
                        kpt_lst = [kpt_lst[i] for i in idxs]
                        damage_name_lst = [damage_name_lst[i] for i in idxs]
                        comp_lst = [comp_lst[i] for i in idxs]
                        
                    vin = row["VIN"]
                    svg_url = get_cage(vin, int(pc))
                    
                    if svg_url == "":
                        print(f"no image for session {session} and pc {pc}")
                        #save(session + f"_{pc}", {"pc": pc, "image": None}, output_dir)
                    #    print(vin)
                    #    no_vin_lst.append(vin)
                        continue
                    
                    pld_a = get_payload_a(vin, cdn, int(pc), svg_url)
                    #pld_b = get_payload_b(int(pc), cdn)
                    pld_b = pld_a
                    
                    start = time.time()
                    crop_data = get_dmg_bboxes(image_id, pld_a, pld_b, session)
                    dmg_crop_bboxes, confidence_lst, method_lst = get_crop_bboxes(crop_data)                
                    print(dmg_crop_bboxes)
                    print(method_lst)
                    dino_bboxes = []
                    
                    for i in range(len(method_lst)):
                        if method_lst[i] == 'b':
                            dino_bboxes.append(dmg_crop_bboxes[i])
                    print(dino_bboxes)
                    if len(dino_bboxes) > 0:
                        print('consulting qa')
                        pld_c = get_payload_c(int(pc), cdn, dino_bboxes)
                        result = get_dmg_assurance(image_id, pld_c)
                        qa_answers = result['answers']
                        qa_bboxes = result['bboxes']
                    else:
                        qa_answers = []
                        qa_bboxes = []
                    print(qa_answers)
                    if len(dino_bboxes) == 0:
                        if len(qa_answers) == 0 or "yes" not in qa_answers:
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
                    opt['dino_bboxes'] = dino_bboxes
                    opt['qa_answers'] = qa_answers
                    opt['qa_bboxes'] = qa_bboxes
                    opt['time'] = time_per_req
                    accumulated_dicts.append(opt)
                    
                    #if (idx + 1) % interval == 0:
                    result = pd.DataFrame(accumulated_dicts)
                    if not os.path.exists(f"results/{output_dir}"):
                        os.makedirs(f"results/{output_dir}")
                    path = f"results/{output_dir}/result_combine_{args.chunk_id}.csv"
                    result.to_csv(path, mode='a', header=not pd.io.common.file_exists(path), index=False)
                    accumulated_dicts.clear()
                pbar.update(1)
            except Exception as e:
                error_message = f"Error processing item {cdn}: {e}"
                print(error_message)
                logging.error(error_message)
                error_log.append(str(e))
            #    #import ipdb;ipdb.set_trace()
                continue
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

