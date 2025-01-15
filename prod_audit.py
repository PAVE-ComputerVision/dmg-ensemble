import io
import json
import torch
import requests
import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image
from pathlib import Path
from collections import defaultdict
from pycocotools import mask as cocomask 
from torchvision.io import write_png
from torchvision.utils import draw_bounding_boxes


def download_from_cdn(url: str) -> bytes:
    res = requests.get(url, stream=True)
    data = b""
    for chunk in res.iter_content(chunk_size=1024):
        if chunk:
            data += chunk
    return data

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

cage_lookup_df = pd.read_csv("/home/ubuntu/amazon_cages_2_0.csv")
def get_cage(vin, pc, fallback_svg="NOT PRESENT"):
    wmi = vin[:3]
    model_year = vin[9]
    vehicle_desc = vin[3:8]
    
    #specific search
    correct_cage_row = cage_lookup_df[(cage_lookup_df["Manufacturer Code"] == wmi) &
                (cage_lookup_df["Model Year Code"] == model_year) &
                (cage_lookup_df["Model Code"] == vehicle_desc)]
    
    #select the correct view
    if pc == 4:
        view = "Cage: Left_View"
    elif pc == 5:
        view = "Cage: Front_View"
    elif pc == 7:
        view = "Cage: Right_View"
    elif pc == 8:
        view = "Cage: Rear_View"

    if correct_cage_row.shape[0] > 1: #there are cages in the row that match the specific search
        print("specific search match", flush=True)
        correct_cage_row = correct_cage_row.head(1)
    else: #empty result
        
        #wider search
        unique_cages = cage_lookup_df[(cage_lookup_df["Manufacturer Code"]  == wmi)][view].unique()
        if unique_cages.shape[0] == 1: #there is only one unique cage. good, lets pull the correct row
            print("wider search match", flush=True)
            svg_url = unique_cages.item()
            return svg_url
        else:
            return fallback_svg
    
    #carry on with grabbing the specific cage
    try:
        svg_url = correct_cage_row[view].item()
    except:
        return fallback_svg
    return svg_url

if __name__ == "__main__":
    compare_df = pd.read_csv('/home/ubuntu/dmg-ensemble/comparison.csv')
    compare_df = compare_df.iloc[200:300]
    #compare_df['session'] = compare_df['cdn_url'].apply(lambda x: x.split('/')[-3])
    #sess_lst = compare_df.session.tolist()[100:]
    #pc_lst = compare_df.pc.tolist()
    cdn_lst = compare_df['cdn_url'].tolist()
    df = pd.read_parquet("/home/ubuntu/241129.parquet")
    
    lst = []
    for i, cdn in enumerate(cdn_lst):
        try:
            pc = int(cdn.split('/')[-1].split('-')[0])
        except:
            import ipdb;ipdb.set_trace()
        match_ = df[df[f"PhotoCode_{pc}"]==cdn]
        
        try:
            vin = match_["VIN"].item()
        except:
            vin = match_["VIN"].iloc[0]

        if vin == None:
            print("NO VIN")
            continue

        #match_ = df[df["VIN"] == vin]
        #for pc in [4,5,7,8]:
        pc_str = f"PhotoCode_{pc}"
        try:
            url = match_[pc_str].item()
        except:
            idx = match_[f"PhotoCode_{pc}"].keys()[0].item()
            url = match_[pc_str][idx]

        svg = get_cage(vin, pc)
        #damage_name_lst = match_['damage_name_lst']
        #kp_lst = match_['kp_lst']
        #photo_lst = match_['photo_lst']
        #component_lst = match_['component_lst']

        if "NOT PRESENT" in svg:
            print("NO SVG")
            continue
        lst.append([url, svg, pc])
    
    print(len(lst))
    time_lst = []
    for x in lst:
        url, cage, pc  = x
        payload = {"vin": vin, 
                "bucket": "cdn", 
                "file": url, 
                "photocode": pc, 
                "date": '20240420', 
                "svg": cage}
        import time
        start = time.time()
        response = requests.post("https://damage-detection.ai-dev.paveapi.com/", json=payload) #call dmg
        end = time.time()
        print('time', end-start)
        response_data = response.json()
        byte_data = download_from_cdn(url)
        img = Image.open(io.BytesIO(byte_data))
        img = torch.from_numpy(np.asarray(img)).permute(2, 0, 1)
        
#        photo_lst = row['photo_lst']
#        try:
#            damage_name_lst = eval(row["dmg_name_lst"])
#            if type(damage_name_lst) == str:
#                damage_name_lst = eval(row["dmg_name_lst"])
#        except:
#            damage_name_lst = eval(row["damage_name_lst"])
#            if type(damage_name_lst) == str:
#                damage_name_lst = eval(row["damage_name_lst"])
#        kp_lst  = row['kp_lst']
#        component_lst = row['component_lst']
#
#        codes = [int(x['code']) for x in photo_lst]
#        freq = Counter(codes)
#        idxs = [i for i in range(len(photo_lst)) if int(photo_lst[i]['code']) == pc]
#        dmg_kpts = [kp_lst[i] for i in idxs]
#        damage_name_lst = [damage_name_lst[i] for i in idxs]
#        component_lst = [component_lst[i] for i in idxs]
#
#        scaled_gt_bbox_lst = []
#        gt_lbl_lst = []
#        gt_lbl_name_lst = []
#        for j, cat in enumerate(damage_name_lst):
#            #Text categories
#            if 'DENT' in cat:
#                lbl_cat = 'dent'
#            elif 'SCRATCH' in cat:
#                lbl_cat = 'scratch'
#            elif 'MISSING' in cat:
#                lbl_cat = 'missing'
#            elif 'SCRAPED' in cat:
#                lbl_cat = 'scraped'
#            elif 'BROKEN' in cat:
#                lbl_cat = 'broken'
#            else:
#                lbl_cat = 'others'
#
#            #Bbox size categories
#            if 'MAJOR' in cat:
#                size_cat = 'large'
#            elif 'MEDIUM' in cat:
#                size_cat = 'medium'
#            elif 'MINOR' in cat:
#                size_cat = 'small' 
#            else:
#                size_cat = 'small'
#
#            category_id = cat_id_dct[lbl_cat]
#            kpts = dmg_kpts[j]
#            bbox = get_coco_bbox(kpts, height, width, size_cat)
#            scaled_gt_bbox_lst.append(bbox)
#            gt_lbl_lst.append(category_id)
#            gt_lbl_name_lst.append(lbl_cat)
#        
#        if len(scaled_gt_bbox_lst) > 0:
#            gt_bboxes = torch.Tensor(scaled_gt_bbox_lst)
#        else:
#            gt_bboxes = torch.Tensor([])

        if response_data['damages'] != None:
            print('dmg')
            bboxes = []
            for dmg in response_data['damages']:
                conf = dmg['damage-confidence']
                kpt = dmg['keypoint']
                bbox = dmg['bounding_box']
                label = dmg['damage-label']
                if bbox == None and conf == None:
                    continue
                scaled_bb = [bbox[0][0]*1920, bbox[0][1]*1080, bbox[1][0]*1920, bbox[1][1]*1080]
                bboxes.append(scaled_bb)
            bboxes = torch.Tensor(bboxes)
            print(bboxes)
            img = draw_bounding_boxes(img, 
                                      bboxes, 
                                      colors="red", 
                                      width=2)

        else:
            print('no dmg')
            bboxes = torch.Tensor([])
            
        #x = f"{url.split('/')[-1].replace('.jpg', '')}"
        #Path.mkdir(f"outputs/{x}", exist_ok=True, parents=True)
        print(url.split('/')[-1])
        write_png(img, f"output_3_disable/{url.split('/')[-1].replace('jpg', 'png')}")
        time_lst.append(end-start)
    torch.save(time_lst, 'time_lst.pt')
