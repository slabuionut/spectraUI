import os
import math
from typing import Union, List, Tuple
import cv2 as cv
from torchvision.ops import box_iou

from tqdm import tqdm
import warnings

import numpy as np
import torch
import cv2 as cv


def ap_per_class(tp, conf, pred_cls, target_cls):


    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]


    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]


    px = np.linspace(0, 1, 1000)
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()
        n_p = i.sum()  

        if n_p == 0 or n_l == 0:
            continue
        else:

            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)


            recall = tpc / (n_l + 1e-16)  
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  


            precision = tpc / (tpc + fpc)  
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  


            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])


    f1 = 2 * p * r / (p + r + 1e-16)
    i = f1.mean(0).argmax()  

    return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')


def compute_ap(recall, precision):



    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))


    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    x = np.linspace(0, 1, 101)
    ap = np.trapz(np.interp(x, mrec, mpre), x)  

    return ap, mpre, mrec



def draw_bbox_and_label(x: torch.Tensor, label: str, img: np.ndarray) -> np.ndarray:

    x1, y1, x2, y2 = tuple(map(int, x))
    if x is not None:
        img = cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
    t_size = cv.getTextSize(label, cv.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = (x1 + t_size[0] + 3, y1 + t_size[1] + 4)
    img = cv.putText(img, label, (x1, y1 + t_size[1] + 4), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1)

    return img


def letterbox_image(
        image: np.ndarray,
        inp_dim: Tuple[int, int]) -> np.ndarray:

    img_w, img_h = image.shape[1], image.shape[0]  
    net_w, net_h = inp_dim  


    scale_factor = min(net_w / img_w, net_h / img_h)
    new_w = int(round(img_w * scale_factor))
    new_h = int(round(img_h * scale_factor))

    resized_image = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_CUBIC)
    canvas = np.full((net_w, net_h, 3), 128)
    canvas[(net_h - new_h) // 2: (net_h - new_h) // 2 + new_h, (net_w - new_w) // 2: (net_w - new_w) // 2 + new_w,
    :] = resized_image
    return canvas


def prepare_image(
        image: np.ndarray,
        inp_dim: Tuple[int, int]) -> torch.Tensor:

    img = letterbox_image(image, inp_dim)
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img


def bbox_iou(bbox1: torch.Tensor, bbox2: torch.Tensor, device="cpu"):

    b1_x1, b1_y1, b1_x2, b1_y2 = bbox1[:, 0], bbox1[:, 1], bbox1[:, 2], bbox1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = bbox2[:, 0], bbox2[:, 1], bbox2[:, 2], bbox2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1, torch.zeros(inter_rect_x2.shape, device=device)) * \
                 torch.max(inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_y2.shape, device=device))

    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    return inter_area / (b1_area + b2_area - inter_area)


def transform_prediction(
        pred: torch.Tensor,
        inp_dim: int,
        anchors: Union[List[int], Tuple[int, ...], torch.Tensor],
        num_classes: int,
        device: str = "cpu"
) -> torch.Tensor:

    batch_size = pred.shape[0]
    grid_size = pred.shape[2]
    stride = inp_dim // grid_size
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)


    pred = pred.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    pred = pred.transpose(1, 2).contiguous()
    pred = pred.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)


    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]


    pred[:, :, 0] = torch.sigmoid(pred[:, :, 0])
    pred[:, :, 1] = torch.sigmoid(pred[:, :, 1])
    pred[:, :, 4] = torch.sigmoid(pred[:, :, 4])


    grid = torch.arange(grid_size, dtype=torch.float)
    grid = np.arange(grid_size)
    x_o, y_o = np.meshgrid(grid, grid)


    x_offset = torch.FloatTensor(x_o).view(-1, 1).to(device)
    y_offset = torch.FloatTensor(y_o).view(-1, 1).to(device)


    x_y_offset = torch.cat([x_offset, y_offset], dim=1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
    pred[:, :, :2] += x_y_offset


    anchors = torch.FloatTensor(anchors).to(device)
    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    pred[:, :, 2:4] = torch.exp(pred[:, :, 2:4]) * anchors


    pred[:, :, 5:5 + num_classes] = torch.sigmoid(pred[:, :, 5:5 + num_classes])


    pred[:, :, :4] *= stride

    return pred


def get_predictions(
        pred: torch.Tensor,
        num_classes: int,
        objectness_confidence: float = 0.5,
        nms_confidence_level: float = 0.4,
        device: str = "cpu") -> Union[torch.Tensor, int]:
    conf_mask = (pred[:, :, 4] > objectness_confidence).float().unsqueeze(2)
    pred = pred * conf_mask


    bbox_corner = pred.new(pred.shape)
    bbox_corner[:, :, 0] = (pred[:, :, 0] - (pred[:, :, 2] / 2))  
    bbox_corner[:, :, 1] = (pred[:, :, 1] - (pred[:, :, 3] / 2))  
    bbox_corner[:, :, 2] = (pred[:, :, 0] + (pred[:, :, 2] / 2))  
    bbox_corner[:, :, 3] = (pred[:, :, 1] + (pred[:, :, 3] / 2))  
    pred[:, :, :4] = bbox_corner[:, :, :4]


    output = None
    for idx in range(pred.shape[0]):
        img_pred = pred[idx]


        max_conf, max_idx = torch.max(img_pred[:, 5:5 + num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1).to(device)
        max_idx = max_idx.float().unsqueeze(1).to(device)
        img_pred = torch.cat([img_pred[:, :5], max_conf, max_idx], 1)

        non_zero_idx = torch.nonzero(img_pred[:, 4]).to(device)
        img_pred = img_pred[non_zero_idx.squeeze(), :].view(-1, 7).to(device)
        if not img_pred.shape[0]:
            continue


        img_classes = torch.unique(img_pred[:, -1]).to(device)


        for cls in img_classes:
            class_mask = img_pred * (img_pred[:, -1] == cls).float().unsqueeze(1)
            class_mask_idx = torch.nonzero(class_mask[:, -2]).squeeze()
            img_pred_class = img_pred[class_mask_idx].view(-1, 7)


            conf_sort_idx = torch.sort(img_pred_class[:, 4], descending=True)[1]
            img_pred_class = img_pred_class[conf_sort_idx]


            for d_idx in range(img_pred_class.shape[0]):
                try:
                    ious = bbox_iou(img_pred_class[d_idx].unsqueeze(0), img_pred_class[d_idx + 1:], device=device)
                except (IndexError, ValueError):
                    break


                iou_mask = (ious < nms_confidence_level).float().unsqueeze(1)
                img_pred_class[d_idx + 1:] *= iou_mask
                non_zero_idx = torch.nonzero(img_pred_class[:, 4]).squeeze()
                img_pred_class = img_pred_class[non_zero_idx].view(-1, 7)

            batch_idx = img_pred_class.new(img_pred_class.shape[0], 1).fill_(idx)
            if isinstance(output, torch.Tensor):
                out = torch.cat([batch_idx, img_pred_class], 1)
                output = torch.cat([output, out])
            else:
                output = torch.cat([batch_idx, img_pred_class], 1)
    return output

def xywh2xyxy(box_coord : torch.Tensor):
    n = box_coord.clone()
    n[:, 0] = (box_coord[:, 0] - (box_coord[:, 2] / 2))
    n[:, 1] = (box_coord[:, 1] - (box_coord[:, 3] / 2))
    n[:, 2] = (box_coord[:, 0] + (box_coord[:, 2] / 2))
    n[:, 3] = (box_coord[:, 1] + (box_coord[:, 3] / 2))

    return n

def process_batch(detections, labels, iouv):
    detections[:, [1,3]] = torch.clamp(detections[:, [1,3]], 0.0, 416)
    detections[:, [2,4]] = torch.clamp(detections[:, [2,4]], 0.0, 416)
    
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, 1:5])
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 7]))  
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct

@torch.no_grad()
def run(model, val_dataloader, num_class, net_dim=416, nms_thresh=0.6, objectness_thresh=0.001, device="cpu"):
    model.eval()
    nc = int(num_class)
    iouv = torch.linspace(0.5, 0.95, 10).to(device)
    niou = iouv.numel()

    p, r, f1, mp, mr, map50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    stats, ap, ap_class = [], [], []
 
    for batch_i, (im, targets) in tqdm(enumerate(val_dataloader)):
        im = im.to(device)
        targets = targets.to(device)
        nb = im.shape[0]


        out = model(im)

        
        targets[:, 2:] *= torch.Tensor([net_dim, net_dim, net_dim, net_dim]).to(device)  
        out = get_predictions(
                pred=out.to(device), num_classes=nc,
                objectness_confidence=objectness_thresh,
                nms_confidence_level=nms_thresh, device=device
            )

        
        for si in range(nb):
            labels = targets[targets[:, 0] == si, 1:]
            pred = out[out[:, 0]==si, :] if isinstance(out, torch.Tensor) else torch.zeros((0,0), device=device)
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool, device="cpu"), torch.Tensor(device="cpu"), torch.Tensor(device="cpu"), tcls))
                continue


            if nc==1:
                pred[:, 7] = 0
            
            if pred.shape[0] > 300:
                pred = pred[:300, :]  
                
            predn = pred.clone()


            if nl:
                tbox = xywh2xyxy(labels[:, 1:5]).to(device)  
                labelsn = torch.cat((labels[:, 0:1], tbox), 1).to(device)  
                correct = process_batch(predn, labelsn, iouv)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            stats.append((correct.cpu(), pred[:, 5].cpu(), pred[:, 7].cpu(), tcls))  


    stats = [np.concatenate(x, 0) for x in zip(*stats)]  
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        ap50, ap = ap[:, 0], ap.mean(1)
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()

    return mp, mr, map50, map