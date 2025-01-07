import os
import torch
import json
import pandas as pd
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import save_image, draw_bounding_boxes
import random

def get_palette(tensor, shade):
    assert tensor.shape[1] == 4, "tensor must be nx4"
    bboxes = tensor.shape[0]

    rgb_lst = []
    if shade == "blue":
        rgb_lst = [(random.randint(0, 77), random.randint(77, 153), random.randint(179, 255)) for _ in range(bboxes)]
    if shade == "red":
        rgb_lst = [(random.randint(179, 255), random.randint(77, 153), random.randint(77, 128)) for _ in range(bboxes)]
    return rgb_lst

def get_bbox_center(bbox_tnr):
    """
    Calculate the center coordinates of a bounding box.

    Parameters:
    - bbox: A tuple (x_min, y_min, x_max, y_max) representing the bounding box coordinates.

    Returns:
    - A tuple (x_center, y_center) representing the center coordinates of the bounding box.
    """
    pred_kpt_lst = []
    for bbox in bbox_tnr:
        x_min, y_min, x_max, y_max = bbox

        # Calculate the center coordinates
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        pred_kpt_lst.append([x_center, y_center])
    pred_kpt_tnr = torch.Tensor(pred_kpt_lst)
    return pred_kpt_tnr

def plot_keypoints_on_image(ori_img, keypoints_tensors, labels_list, colors, save_path):
    """
    Plots multiple sets of keypoints with labels on an image and saves the result.

    Parameters:
    - image_path: Path to the image file.
    - keypoints_tensors: List of tensors, each of shape (N, 2), representing keypoint coordinates.
    - labels_list: List of label lists, where each list corresponds to labels for a keypoint tensor.
    - colors: List of colors for each keypoint set (e.g., 'red' or 'blue').
    - save_path: Path to save the output image with keypoints and labels.
    """
    # Load and plot the image
    image = ori_img #Image.open(image_path)
    width, height = image.size
    plt.figure(figsize=(width / 100, height / 100), dpi=100)
    plt.imshow(image)

    # Plot each set of keypoints with the specified color and labels
    for keypoints_tensor, labels, color in zip(keypoints_tensors, labels_list, colors):
        keypoints = keypoints_tensor.numpy()
        for i, (x, y) in enumerate(keypoints):
            plt.plot(x, y, 'o', color=color)  # Plot keypoint in specified color
            if color != 'green':
                plt.text(x + 5, y, labels[i], color=color, fontsize=12)  # Label in specified color

    plt.axis('off')  # Hide axes

    # Save the resulting image with keypoints and labels
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == '__main__':
    csv_files = glob('new_r8tr_test/*.csv')
    json_files = glob('new_r8tr_test/*.json')
    img_files = glob('test_images/*.jpg')
    for img_pth in img_files:
        ori_img = Image.open(img_pth).convert('RGB')
        transform = transforms.PILToTensor()
        image_tensor = transform(ori_img)
        fname = img_pth.split('/')[-1].split('.')[0]
        #fname = '8-9d3f303f-2b26-4ea6-a58f-fc23092fa2cf-1920x1080'
        try:
            df = pd.read_csv(f'new_r8tr_test/{fname}_crop_data.csv')
        except:
            print('file not found')
            continue
        with open(f'new_r8tr_test/{fname}_results.json', 'r') as file:
            data = json.load(file)

        print(df)
        session = df['session'].iloc[0]

        x1_lst = df['x_min'].tolist()
        y1_lst = df['y_min'].tolist()
        x2_lst = df['x_max'].tolist()
        y2_lst = df['y_max'].tolist()
        methods = df['method'].tolist()
        bboxes_a = []
        bboxes_b = []
        bboxes_c = []
        for x1, y1, x2, y2, method in zip(x1_lst,y1_lst,x2_lst,y2_lst, methods):
            #if method == 'a':
            bboxes_a.append([x1, y1, x2, y2])
            #elif method == 'b':
            #    bboxes_b.append([x1, y1, x2, y2])
            #elif method == 'c':
            #    continue
                #bboxes_c.append([x1, y1, x2, y2])

        bboxes_a_tnr = torch.Tensor(bboxes_a)
        #bboxes_b_tnr = torch.Tensor(bboxes_b)
        #bboxes_c_tnr = torch.Tensor(bboxes_c)
        
        #kpts_a_tnr = get_bbox_center(bboxes_a_tnr)
        #kpts_b_tnr = get_bbox_center(bboxes_b_tnr)
        #kpts_c_tnr = get_bbox_center(bboxes_c_tnr)
        
        #confs_a = df[df['method']=='a']['confidence'].tolist()
        #confs_b = df[df['method']=='b']['confidence'].tolist()
        #confs_a = [round(value, 2) for value in confs_a]        
        #confs_b = [round(value, 2) for value in confs_b]
        #confs_c = []

        #colors = ['red', 'blue', 'green']
        #folder_pth = f'test_vis/{session}/'
        #if not os.path.exists(folder_pth):
        #    os.makedirs(folder_pth)
        #save_path = os.path.join(folder_pth,f'{fname}.jpg')
        #plot_keypoints_on_image(ori_img,
        #                        [kpts_a_tnr, kpts_b_tnr, kpts_c_tnr],
        #                        [confs_a, confs_b, confs_c],
        #                        colors,
        #                        save_path)
        #continue

        #try:
        print(bboxes_a_tnr)
        colours = get_palette(bboxes_a_tnr, "red")
        img_vis = draw_bounding_boxes(image_tensor, bboxes_a_tnr, colors=colours, width=3)
        flag = 0
        #except IndexError:
        #    img_vis = image_tensor
        #    flag = 1
#        try:
#            colours = get_palette(bboxes_b_tnr, "blue")
#            img_vis = draw_bounding_boxes(img_vis, bboxes_b_tnr, colors=colours, width=3)
#        except IndexError:
#            if flag == 1:
#                img_vis = image_tensor
#        try:
#            #colours = get_palette(bboxes_c_tnr, "green")
#            img_vis = draw_bounding_boxes(img_vis, bboxes_c_tnr, colors="green", width=3)
#        except IndexError:
#            img_vis = img_vis
        #    if flag == 1:
        #        img_vis = image_tensor

        img_vis = img_vis/255.

        #Check if a folder exists if not create
        folder_pth = f'test_vis_new/{session}/'
        if not os.path.exists(folder_pth):
            os.makedirs(folder_pth)
        save_image(img_vis, os.path.join(folder_pth,f'{fname}.jpg'))
        print('done')
