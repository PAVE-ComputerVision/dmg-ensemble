import time
import torch
import argparse
import requests
import pandas as pd
import os
import json

def adjust_bbox_to_min_size(bbox, min_size=224, img_width=1920, img_height=1080):
    """
    Adjust the bounding box to ensure a minimum width and height by padding if necessary,
    while ensuring the adjusted box stays within the image dimensions.

    Parameters:
    - bbox: Bounding box coordinates in the format [x_min, y_min, x_max, y_max].
    - min_size: The minimum width and height for the adjusted bounding box.
    - img_width: Width of the image.
    - img_height: Height of the image.

    Returns:
    - A new bounding box in the format [x_min, y_min, x_max, y_max] with at least min_size dimensions,
      and constrained within the image boundaries.
    """
    x_min, y_min, x_max, y_max = bbox

    # Calculate current width and height of the bounding box
    width = x_max - x_min
    height = y_max - y_min

    # Check if padding is needed
    if width < min_size:
        pad_w = (min_size - width) / 2
        x_min -= pad_w
        x_max += pad_w

    if height < min_size:
        pad_h = (min_size - height) / 2
        y_min -= pad_h
        y_max += pad_h

    # Ensure coordinates are within image boundaries
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_width, x_max)
    y_max = min(img_height, y_max)

    # Round the coordinates to integers for pixel alignment
    adjusted_bbox = [x_min, y_min, x_max, y_max]
    
    return adjusted_bbox

def fetch_data(payload):
    """
    Fetches bounding box data from the API for either damage detection or classification.
    
    Args:
        image_id (str): Unique identifier for the image.
        api_url (str): Base URL of the API.
        method (str): Method identifier ('a' or 'b').

    Returns:
        list of tuples: Bounding boxes as (x_min, y_min, x_max, y_max, confidence) in normalized format.
    """
    #if method == 'a':
    api_url = 'https://damage-detection.ai-dev.paveapi.com/'
    start = time.time() 
    response = requests.post(api_url, json=payload)
    print(time.time() - start)
    response_data = response.json()
    bounding_boxes = []
    ori_bounding_boxes = []
    model_ids = []
    if response_data['damages'] == None:
        bounding_boxes, ori_bounding_boxes, model_ids = [],[],[]
    else:
        for dmg in response_data['damages']:
            conf = dmg['damage-confidence']
            kpt = dmg['keypoint']
            bbox = dmg['bounding_box']
            label = dmg['damage-label']
            if bbox == None and conf == None:
                continue
            
            if label == None:
                model_ids.append('a')
            else:
                model_ids.append('b')
            
            #NOTE: Delete after adjusting a response to scaled bbox
            scaled_bb = [bbox[0][0]*1920, bbox[0][1]*1080, bbox[1][0]*1920, bbox[1][1]*1080]
            adjusted_bb = adjust_bbox_to_min_size(scaled_bb)
            
            scaled_bb.extend([conf])
            adjusted_bb.extend([conf])
            
            ori_bounding_boxes.append(scaled_bb)
            bounding_boxes.append(adjusted_bb)
#    elif method == 'b':
#    api_url = 'https://damage-classification.ai-dev.paveapi.com/'
#    response = requests.post(api_url, json=payload)
#    response_data = response.json()
#    dmg_boxes = response_data['bounding_boxes']
#    if dmg_boxes == []:
#        dmg_boxes = None
#    confidences = response_data['confidences']
#    bounding_boxes = []
#    ori_bounding_boxes = []
#    model_ids = []
#    if dmg_boxes:
#        for i in range(len(dmg_boxes)):
#            bbox = dmg_boxes[i]
#            conf = confidences[i]
#            adjusted_bb = adjust_bbox_to_min_size(bbox)
#            
#            bbox.extend([conf])
#            adjusted_bb.extend([conf])
#            
#            ori_bounding_boxes.append(bbox)
#            bounding_boxes.append(adjusted_bb)
    return bounding_boxes, ori_bounding_boxes, model_ids, response_data["version"]

def get_dmg_bboxes(image_id, pld_a, pld_b, session):
    """
    Fetch bounding box data from damage detection and classification methods, log results, and prepare data for QA.
    
    Args:
        image_id (str): Unique identifier for the image.
        api_url (str): Base URL of the API.
        output_dir (str): Directory to save the bounding box data.
        
    Returns:
        list: Crop data to be sent to get_dmg_assurance for further processing.
    """

    data = {"a": {}, "b": {}}
    #for method in ['a']:#['a', 'b']:
    #    if method == 'a':
    #print('Running dmg det endpoint')   
    bboxes, ori_bboxes, model_ids, version = fetch_data(pld_a)
    print(bboxes)
    #    elif method == 'b':
    #        print('Running dmg cls endpoint')   
    #        bboxes, ori_bboxes = fetch_data(pld_b, method)
            
    #    data[method]["bboxes"] = bboxes
    #    data[method]["ori_bboxes"] = ori_bboxes

    crop_data = []
    for idx, (x_min, y_min, x_max, y_max, confidence) in enumerate(ori_bboxes):
        crop_data.append({
            'image_id': image_id,
            'session': session,
            'method': 'only_dino',#model_ids[idx],
            'box_index': -1,#idx,
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max,
            'confidence': confidence,
        })
    
    return crop_data, version

def get_dmg_assurance(image_id, payload):
    """
    Calls the damage QA endpoint to process all crop data in a single request.
    
    Args:
        image_id (str): Unique identifier for the image.
        api_url (str): Base URL for damage QA API endpoint.
        crop_data (list): List of dictionaries with bounding box information to send to damage QA.

    Returns:
        dict: JSON response from damage QA.
    """

    image_id = payload['file'].split('/')[-1].split('.')[0]
    endpoint_c = "https://damage-qa.ai-dev.paveapi.com/"
    response = requests.post(endpoint_c, json=payload)
    result = response.json()
    #print(f"Received response from damage QA for image_id: {image_id}")
    return result

def main():
    parser = argparse.ArgumentParser(description="Fetch bounding box data, log crop information, and call damage QA.")
    #parser.add_argument('--image_id', type=str, required=True, help="Unique identifier for the image.")
    #parser.add_argument('--api_url', type=str, required=True, help="Base URL for the API endpoints.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save crop data.")

    args = parser.parse_args()
    
    df = pd.read_csv('dmg_test_r8tr.csv')
    
    cage_lookup_df = pd.read_csv("amazon_cages_2_0.csv")
    new_header = df.iloc[0]
    df = df[1:]
    df.columns = new_header
    
    for i, row in df.iterrows():
        
        #Load image
        cdn = row["Image URL"]
        fname = cdn.split('/')[-1]
        photocode = int(fname.split('-')[0])
        vin = row["Vin"]
        wmi = vin[:3]
        model_year = vin[9]
        vehicle_desc = vin[3:8]
        correct_cage_row = cage_lookup_df[(cage_lookup_df["Manufacturer Code"] == wmi) &
                    (cage_lookup_df["Model Year Code"] == model_year) &
                    (cage_lookup_df["Model Code"] == vehicle_desc)]
        if correct_cage_row.shape[0] > 1:
            correct_cage_row = correct_cage_row.head(1) 
        pc = photocode
        try:
            if pc == 4:
                svg_url = correct_cage_row["Cage: Left_View"].item()
            elif pc == 5:
                svg_url = correct_cage_row["Cage: Front_View"].item()
            elif pc == 7:
                svg_url = correct_cage_row["Cage: Right_View"].item()
            elif pc == 8:
                svg_url = correct_cage_row["Cage: Rear_View"].item() 
        except:
            print('no svg found')
            continue
        pld_a = {"vin": vin, 
                "bucket": "cdn", 
                "file": cdn, 
                "photocode": pc, 
                "date": '20240420', 
                "svg": svg_url}
        
        #pld_a_ = {'vin': '1FDDF6P82MKA55215', 'bucket': 'cdn', 'file': 'https://openapi-cdn.paveapi.com/sessions/sessions/2024-04/AMRR-TXRLOWKCQG/capture/4-9bd8c796-475d-4315-a26c-bd1910a4c137-1920x1080.jpg', 'photocode': '4', 'date': '20240420', 'svg': 'https://cages.paveapi.com/6513e8fe4edde7001ec9a9e9/01-final.svg?ts=1697726554172?v=1697726554316'} 
        
        image_id = pld_a['file'].split('/')[-1].split('.')[0]
        
        pld_b = {}
        pld_b["photocode"] = int(pld_a["photocode"])
        pld_b["file"] = pld_a["file"] 
        pld_b["bucket"] = "cdn"
        pld_b["svg"] = pld_a["svg"]
        
        os.makedirs(args.output_dir, exist_ok=True)

        crop_data = get_dmg_bboxes(image_id, pld_a, pld_b, args.output_dir)
        #crop_data = torch.load('test_crop_data.pt')
        dmg_crop_bboxes = []
        for i, data in enumerate(crop_data):
            x1 = data['x_min']
            y1 = data['y_min']
            x2 = data['x_max']
            y2 = data['y_max']
            dmg_crop_bboxes.append([x1,y1,x2,y2])
        pld_c = {}
        pld_c["pc"] = int(pld_a["photocode"])
        pld_c["file"] = pld_a["file"]
        pld_c["bucket"] = "cdn"
        pld_c["bboxes"] = dmg_crop_bboxes
        #print(len(dmg_crop_bboxes))
        method_c_result = get_dmg_assurance(image_id, pld_c)
        #print(method_c_result)
        method_c_output_path = os.path.join(args.output_dir, f"{image_id}_method_c_results.json")
        with open(method_c_output_path, "w") as f:
            json.dump(method_c_result, f)
        print(f"Method c results saved to: {method_c_output_path}")
if __name__ == "__main__":
    main()

