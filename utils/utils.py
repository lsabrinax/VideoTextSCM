import os
import os.path as osp
import numpy as np
from shapely.geometry import Polygon
import cv2
import pyclipper
import torch

def mkdir_if_missing(d):
    if not osp.exists(d):
        os.makedirs(d)
        
def write2xml(file_root, results, change_id=False):
    xml_file = open(file_root, 'w')
    xml_file.write('<Frames>\n')
    for i, result in enumerate(results):  # result: pre frame; box:[pts id]
        if change_id:
            xml_file.write('  <frame ID="{}">\n'.format(i+1))
        else:
            xml_file.write('  <frame ID="{}">\n'.format(i))
        for box in result:
            x1,y1,x2,y2,x3,y3,x4,y4,oid=box
            xml_file.write('    <object ID="{}" >\n'.format(oid))
            xml_file.write('      <Point x="{}" y="{}" />\n'.format(int(x1),int(y1)))
            xml_file.write('      <Point x="{}" y="{}" />\n'.format(int(x2),int(y2)))
            xml_file.write('      <Point x="{}" y="{}" />\n'.format(int(x3),int(y3)))
            xml_file.write('      <Point x="{}" y="{}" />\n'.format(int(x4),int(y4)))
            xml_file.write('    </object>\n')
        xml_file.write('  </frame>\n')
    xml_file.write('</Frames>\n')

def write2txt(filename, results, change_id=False):
    save_format = '{imgid},{insid},{x0},{y0},{w},{h},1,-1,-1,-1\n'
    with open(filename, 'w') as f:
        for i, result in enumerate(results):
            if change_id:
                imgid = i + 1
            else:
                imgid = i
            for box in result:
                x1,y1,x2,y2,x3,y3,x4,y4,oid=box
                t, l = min(y1,y2,y3,y4), min(x1,x2,x3,x4)
                b, r = max(y1,y2,y3,y4), max(x1,x2,x3,x4)
                x0, y0, w, h = l, t, r-l+1, b-t+1
                line = save_format.format(imgid=imgid, insid=oid, x0=x0, y0=y0, w=w, h=h)
                f.write(line)

def save_det_res(txt_name, video_name, boxes, save_dir, dataset):
    if dataset == 'roadtext':
        save_format = '{x0},{y0},{x2},{y2}\n'
    elif dataset == 'minetto' or dataset == 'icdar':
        save_format = '{x0},{y0},{x1},{y1},{x2},{y2},{x3},{y3}\n'
    gt_file = os.path.join(save_dir, video_name+'_'+txt_name)
    if torch.is_tensor(boxes) and boxes.shape[0]>0:
        boxes = boxes[:,:8].int().numpy()
        with open(gt_file,'w') as f:
            for box in boxes:
                if dataset == 'roadtext':
                    x0, y0 = min(box[::2]), min(box[1::2])
                    x2, y2 = max(box[::2]), max(box[1::2])
                elif dataset == 'minetto' or dataset == 'icdar':
                    x0, y0, x1, y1, x2, y2, x3, y3 = box.tolist()
                f.write(save_format.format(x0=x0,y0=y0,x1=x1,y1=y1,x2=x2,y2=y2,x3=x3,y3=y3))
        f.close()
    else:
        f = open(gt_file,'w')
        f.close()

def polygon_iou(apts, bpts):
    ious = np.empty((apts.shape[0], bpts.shape[0]))
    for i, apt in enumerate(apts):
        apt = apt.reshape(4, 2)
        polya = Polygon(apt).convex_hull
        for j, bpt in enumerate(bpts):
            bpt = bpt.reshape(4, 2)
            polyb = Polygon(bpt).convex_hull
            inter = polya.intersection(polyb).area
            union = polya.area + polyb.area - inter
            ious[i, j] = inter / (union + 1e-6)
    return ious

def iou(reference, proposals):
    """Compute the IoU between a reference box with multiple proposal boxes.
    args:
        reference - Tensor of shape (1, 4).
        proposals - Tensor of shape (num_proposals, 4)
    returns:
        torch.Tensor - Tensor of shape (num_proposals,) containing IoU of reference box with each proposal box.
    """
    # Intersection box
    tl = torch.max(reference[:,:2], proposals[:,:2])
    br = torch.min(reference[:,:2] + reference[:,2:], proposals[:,:2] + proposals[:,2:])
    sz = (br - tl).clamp(0)
    # Area
    intersection = sz.prod(dim=1)
    union = reference[:,2:].prod(dim=1) + proposals[:,2:].prod(dim=1) - intersection

    return intersection / (union + 1e-6)

def validate_polygons(polygons, ignore_tags, h, w):
    '''
    polygons (numpy.array, required): of shape (num_instances, num_points, 2)
    '''
    if polygons.shape[0] == 0:
        return polygons, ignore_tags
    assert polygons.shape[0] == len(ignore_tags)
    polygons[:, :, 0] = np.clip(polygons[:, :, 0], 0, w - 1)
    polygons[:, :, 1] = np.clip(polygons[:, :, 1], 0, h - 1)

    for i in range(polygons.shape[0]):
        area = Polygon(polygons[i]).area
        if abs(area) < 1:
            ignore_tags[i] = True
    return polygons, ignore_tags
    
def make_seg_shrink(polygons, ignore_tags,h,w):
    min_text_size = 8
    shrink_ratio = 0.4
    polygons, ignore_tags = validate_polygons(
            polygons, ignore_tags, h, w)
    gt = np.zeros((1, h, w), dtype=np.float32)
    mask = np.ones((h, w), dtype=np.float32)
    for i in range(polygons.shape[0]):
        polygon = polygons[i]
        height = min(np.linalg.norm(polygon[0] - polygon[3]),
                     np.linalg.norm(polygon[1] - polygon[2]))
        width = min(np.linalg.norm(polygon[0] - polygon[1]),
                    np.linalg.norm(polygon[2] - polygon[3]))
        if ignore_tags[i] or min(height, width) < min_text_size:
            cv2.fillPoly(mask, polygon.astype(
                np.int32)[np.newaxis, :, :], 0)
            ignore_tags[i] = True
        else:
            polygon_shape = Polygon(polygon)
            distance = polygon_shape.area * \
                (1 - np.power(shrink_ratio, 2)) / polygon_shape.length
            subject = [tuple(l) for l in polygons[i]]
            padding = pyclipper.PyclipperOffset()
            padding.AddPath(subject, pyclipper.JT_ROUND,
                            pyclipper.ET_CLOSEDPOLYGON)
            shrinked = padding.Execute(-distance)
            if shrinked == []:
                cv2.fillPoly(mask, polygon.astype(
                    np.int32)[np.newaxis, :, :], 0)
                ignore_tags[i] = True
                continue
            shrinked = np.array(shrinked[0]).reshape(-1, 2)
            cv2.fillPoly(gt[0], [shrinked.astype(np.int32)], 1)
    return np.squeeze(gt)

