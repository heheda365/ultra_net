import sys
import os

sys.path.append(os.path.abspath("../common"))

import math
import time
import numpy as np
from PIL import Image
from matplotlib import pyplot
import cv2
from datetime import datetime
import dac_sdc

import mymodel
import torch

team_name = 'use_pc_gpu'
team = dac_sdc.Team(team_name, batch_size = 1)

raw_height = 360
raw_width = 640

BATCH_SIZE = team.batch_size
height = 160
width = 320

device = 'cuda'

model = mymodel.UltraNetQua()
# # print(model)
model.load_state_dict(torch.load('test_best.pt', map_location='cpu')['model'])

model = model.to(device)

def get_boxes(pred_boxes, pred_conf):
    n = pred_boxes.size(0)
    # pred_boxes = pred_boxes.view(n, -1, 4)
    # pred_conf = pred_conf.view(n, -1, 1)
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor
    p_boxes = FloatTensor(n, 4)
    # print(pred_boxes.shape, pred_conf.shape)

    for i in range(n):
        _, index = pred_conf[i].max(0)
        # print('index ', index)
        p_boxes[i] = pred_boxes[i][index]

    return p_boxes

def net(data, result):
    model.eval()
    data.resize(1, data.shape[0], data.shape[1], data.shape[2])
    data = data.transpose(0, 3, 1, 2)
    data = torch.from_numpy(data)
    data = data.to(device).float() / 255.0
    inf_out, train_out = model(data)

    inf_out = inf_out.view(inf_out.shape[0], 6, -1)
    inf_out_t = torch.zeros_like(inf_out[:, 0, :])
    for i in range(inf_out.shape[1]):
        inf_out_t += inf_out[:, i, :]
    inf_out_t = inf_out_t.view(inf_out_t.shape[0], -1, 6) / 6

    # 宽为 320 高为 160 时计算的框
    pre_box = get_boxes(inf_out_t[..., :4], inf_out_t[..., 4])

    # 转换到 360 * 640
    pre_box = pre_box[..., :4] * torch.Tensor([raw_width/width, raw_height/height, raw_width/width, raw_height/height]).to(device)

    for box in pre_box:
        xmin = box[0] - box[2] / 2
        xmax = box[0] + box[2] / 2
        ymin = box[1] - box[3] / 2
        ymax = box[1] + box[3] / 2
#         temp = [int(xmin + 0.5), int(xmax + 0.5), int(ymin + 0.5), int(ymax + 0.5)]
        temp = [int(xmin), int(xmax), int(ymin), int(ymax)]
        result.append(temp)

    # print(pre_box)





## Inference (Main Part)
interval_time = 0
total_time = 0
total_energy = 0
result = list()
pro_image_cnt = 0
team.reset_batch_count()

while True:
    # get a batch of images
    image_paths = team.get_next_batch()
    if image_paths is None:
        break
    
    # Read all images in this batch from the SD card.
    # This part doesn't count toward your time/energy usage.
    rgb_imgs = []
    # use bgr image
#     bgr_imgs = []
    for image_path in image_paths:
        bgr_img = cv2.imread(str(image_path))    
        # rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        # bgr_imgs.append(bgr_img)
        rgb_imgs.append(bgr_img)

        
    # Start timer and energy recorder
    start = time.time()    
    
    # data = np.array((len(image_paths), height, width, 3), dtype=np.uint8)
    for i in range(len(rgb_imgs)):
        # image = cv2.resize(rgb_imgs[i], (width, height), interpolation=cv2.INTER_NEAREST) 
        image = cv2.resize(rgb_imgs[i], (width, height), interpolation=cv2.INTER_LINEAR) 
        # data[i, :] = image
    net(image, result)
    
#     last_bais.

    # timer stop after batch processing is complete
    pro_image_cnt += len(image_paths)
    end = time.time()
    t = end - start
    
    energy = 0
#     energy = recorder.frame["power1_power"].mean() * t
        
#     print('Batch Processing time: {} seconds.'.format(t))
#     print('Batch Energy: {} J.'.format(energy))
    print('pro_image_cnt', pro_image_cnt)
    
    total_time += t
    total_energy += energy
print('images nums: {} .'.format(pro_image_cnt))
print('total Processing time: {} seconds.'.format(total_time))
print('fps: {} .'.format(pro_image_cnt / total_time))
print('Batch Energy: {} J.'.format(total_energy))

team.save_results_xml(result, total_time, energy)