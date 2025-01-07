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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_id", type=int, default=0)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--time", type=float, default=1.0, help="time in hours")
    args = parser.parse_args()
    
    #output_dir = f"out_gd150k_0002_pali_dmg-qa-50k_damaged_combine_largedata1120"
    output_dir = f"250107_timed_hour_90thresh"
    
    data = pd.read_parquet('/home/ubuntu/AMZ_Delta_DF_241003_30k.parquet')
    #df = pd.read_csv('/home/ubuntu/AMZ_DF_V8.csv')
    no_vin_lst = torch.load('no_vin_AMZ_Delta_DF_241003_30k.pt')
    data = data[~data['VIN'].isin(no_vin_lst)]
    
    df = data

    df = df.iloc[int(args.chunk_id)::int(args.chunks)]
    
    
    #set limit for experiment
    start_time = time.time()
    duration_hours = args.time 
    duration_seconds = duration_hours * 3600
    current_time = time.time() - start_time
    with tqdm(total=df.shape[0]) as pbar:
        total = 0
        interval = 10
        error_log = []
        accumulated_dicts = []
        no_vin_lst = []
        while current_time < duration_seconds:
            for idx, (_, row) in enumerate(df.iterrows()):
                print(f"Test time elapsed: {(current_time/60):.2f} minutes. Time remaining: {((duration_seconds - current_time)/60):.2f} minutes")
                try:
                    session = row["SessionKey"]
                    for pc in ['04', '05', '07', '08']:
                        
                        #Load data from csv
                        pc = str(pc).zfill(2)
                        cdn = row[f"PhotoCode_{int(pc)}"]  
                        if cdn == None:
                            continue
                        
                        image_id = cdn.split('/')[-1].split('.')[0]
                        try:
                            photo_lst = json.loads(json.loads(row["photo_lst"]))
                        except:
                            photo_lst = []
                        try:
                            damage_name_lst = json.loads(json.loads(row["dmg_name_lst"]))
                        except:
                            damage_name_lst = []
                        try:
                            comp_lst = json.loads(json.loads(row["component_lst"]))
                        except:
                            comp_lst = []
                        try:
                            kpt_lst = json.loads(json.loads(row["kp_lst"]))
                        except:
                            kpt_lst = []
                            

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
                            gt_bboxes = [gt_bboxes[i] for i in idxs]
                        vin = row["VIN"]
                        svg_url = get_cage(vin, int(pc))
                        if svg_url == "":
                            print(f"no image for session {session} and pc {pc}")
                            #save(session + f"_{pc}", {"pc": pc, "image": None}, output_dir)
                            print(vin)
                            no_vin_lst.append(vin)
                            continue
                        
                        pld_a = get_payload_a(vin, cdn, int(pc), svg_url)
                        pld_b = pld_a
                        
                        start = time.time()
                        crop_data, version = get_dmg_bboxes(image_id, pld_a, pld_b, session)
                        dmg_crop_bboxes, confidence_lst, method_lst = get_crop_bboxes(crop_data)                
                        print(dmg_crop_bboxes)
                        print(method_lst)
                        end = time.time()
                        time_per_req = (end-start)
                        total += time_per_req 
                        
                        opt = {}
                        opt['cdn_url'] = cdn
                        opt['fname'] = image_id
                        opt['session'] = session
                        opt['method_lst'] = method_lst
                        opt['gt_bboxes'] = gt_bboxes
                        opt['all_pred_bboxes'] = dmg_crop_bboxes
                        opt['pred_confs'] = confidence_lst
                        opt['time'] = time_per_req
                        opt["version"] = version
                        accumulated_dicts.append(opt)
                        
                        #if (idx + 1) % interval == 0:
                        result = pd.DataFrame(accumulated_dicts)
                        if not os.path.exists(f"results/{output_dir}"):
                            os.makedirs(f"results/{output_dir}")
                        path = f"results/{output_dir}/result_combine_{args.chunk_id}.csv"
                        result.to_csv(path, mode='a', header=not pd.io.common.file_exists(path), index=False)
                        accumulated_dicts.clear()
                        current_time = time.time() - start_time
                        if current_time >= duration_seconds:
                            break
                        pbar.update(1)
                except Exception as e:
                    error_message = f"Error processing item {cdn}: {e}"
                    print(error_message)
                    logging.error(error_message)
                    error_log.append(str(e))
                #    import ipdb;ipdb.set_trace()
                    continue
                
                if current_time >= duration_seconds:
                    break
    print('Total time', total)
    files  = glob(f"results/{output_dir}/*.csv")
    for i, file in enumerate(files):
        x = pd.read_csv(file)
        if i == 0:
            final = x
        else:
            final = pd.concat([final,x])
    final = final.drop_duplicates(subset="cdn_url")
    final['num_gts'] = final['gt_bboxes'].apply(lambda x: len(x))
    final['num_preds'] = final['all_pred_bboxes'].apply(lambda x: len(x))
    #final['qa_dmg_num'] = final['qa_answers'].apply(lambda x: 1 if 'yes' in x else 0)
    #final['final_dmg_cnt'] = final.apply(lambda x: x['qa_dmg_num']+x['num_preds'], axis=1)
    final.to_csv(f'results/{output_dir}/final.csv', index=False)

