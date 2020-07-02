import argparse
import json

from torch.utils.data import DataLoader

from models import *
from utils.datasets import *
from utils.utils import *

from mymodel import *

hyp = {'giou': 3.54,  # giou loss gain
       'cls': 37.4,  # cls loss gain
       'cls_pw': 1.0,  # cls BCELoss positive_weight
       'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
       'obj_pw': 1.0,  # obj BCELoss positive_weight
       'iou_t': 0.225,  # iou training threshold
       'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
       'lrf': -4.,  # final LambdaLR learning rate = lr0 * (10 ** lrf)
       'momentum': 0.937,  # SGD momentum
       'weight_decay': 0.000484,  # optimizer weight decay
       'fl_gamma': 0.5,  # focal loss gamma
       'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98,  # image rotation (+/- deg)
       'translate': 0.05,  # image translation (+/- fraction)
       'scale': 0.05,  # image scale (+/- gain)
       'shear': 0.641}  # image shear (+/- deg)

def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """
 
    # Transform from center and width to exact coordinates
    b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    
    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def get_boxes(pred_boxes, pred_conf):
    n = pred_boxes.size(0)
    # pred_boxes = pred_boxes.view(n, -1, 4)
    # pred_conf = pred_conf.view(n, -1, 1)
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor
    p_boxes = FloatTensor(n, 4)
    # print(pred_boxes.shape, pred_conf.shape)

    for i in range(n):
        _, index = pred_conf[i].max(0)
        p_boxes[i] = pred_boxes[i][index]

    return p_boxes

def test(cfg,
         data,
         weights=None,
         batch_size=16,
         img_size=416,
         conf_thres=0.001,
         iou_thres=0.5,  # for nms
         save_json=False,
         single_cls=False,
         model=None,
         dataloader=None):
    # Initialize/load model and set device
    if model is None:
        device = torch_utils.select_device(opt.device, batch_size=batch_size)

        # Remove previous
        for f in glob.glob('test_batch*.jpg'):
            os.remove(f)

        # Initialize model
        # model = Darknet(cfg, img_size).to(device)
        model = TestNetQua().to(device)

        model.hyp = hyp
        model.nc = 1
        model.arc = 'default'

        # Load weights
        attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            load_darknet_weights(model, weights)

        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:  # called by train.py
        device = next(model.parameters()).device  # get model device

    names = ['aa']
    path = '../dac_test1/test_data'
    # path = '../dac_test_pix/test'
    # path = '../data_training'

    # Dataloader
    if dataloader is None:
        dataset = LoadImagesAndLabels(path, img_size, batch_size, rect=False, cache_labels=False)
        batch_size = min(batch_size, len(dataset))
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]),
                                pin_memory=True,
                                collate_fn=dataset.collate_fn)

    seen = 0
    model.eval()
    loss = torch.zeros(3)
    iou_sum = 0
    test_n = 0

    print(('\n' + '%10s' * 4) % ('IOU', 'l', 'Giou-l', 'obj-l'))
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))    
    for batch_i, (imgs, targets, paths, shapes) in pbar:
        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        bn, _, height, width = imgs.shape  # batch size, channels, height, width
        test_n += bn

        # print(imgs.shape)

        # Disable gradients
        with torch.no_grad():
            # Run model
            inf_out, train_out = model(imgs)  # inference and training outputs
            
            
            # Compute loss
            if hasattr(model, 'hyp'):  # if model has loss hyperparameters
                loss += compute_loss(train_out, targets, model)[1][:3].cpu()  # GIoU, obj, cls

            inf_out = inf_out.view(inf_out.shape[0], 6, -1)
            # ft = torch.cuda.FloatTensor if p[0].is_cuda else torch.Tensor
            inf_out_t = torch.zeros_like(inf_out[:, 0, :])
            for i in range(inf_out.shape[1]):
               inf_out_t += inf_out[:, i, :]
            inf_out_t = inf_out_t.view(inf_out_t.shape[0], -1, 6) / 6


            pre_box = get_boxes(inf_out_t[..., :4], inf_out_t[..., 4])
            # pre_box = get_boxes(inf_out[..., :4], inf_out[..., 4])
            tbox = targets[..., 2:6] * torch.Tensor([width, height, width, height]).to(device)

            # print(pre_box)
            # print(targets)
            # print(pre_box.shape, tbox.shape)
            ious = bbox_iou(pre_box, tbox)
            iou_sum += ious.sum()
            loss_o = loss / (batch_i + 1)

            iou = iou_sum / test_n
            s = ('%10.4f')*4 % (iou, loss_o.sum(), loss_o[0], loss_o[1])
            pbar.set_description(s)
    return iou, loss_o.sum(), loss_o[0], loss_o[1]
           

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-tiny-1cls_1.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='data/coco2014.data', help='*.data path')
    parser.add_argument('--weights', type=str, default='test_best.pt', help='weights path')
    parser.add_argument('--batch-size', type=int, default=8, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    opt = parser.parse_args()
    opt.save_json = opt.save_json or any([x in opt.data for x in ['coco.data', 'coco2014.data', 'coco2017.data']])
    print(opt)

    # Test
    test(opt.cfg,
            opt.data,
            opt.weights,
            opt.batch_size,
            opt.img_size,
            opt.conf_thres,
            opt.iou_thres,
            opt.save_json,
            opt.single_cls)
