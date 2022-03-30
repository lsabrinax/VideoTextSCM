import os
import os.path as osp
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from utils import utils
from tracker.db_text_multitracker import JDETracker

def get(root, name):
    vars = root.findall(name)
    return vars

def draw_gt(xml_dir, online_im, frame_id, opt):
    tree = ET.parse(xml_dir)
    root = tree.getroot()
    frames = get(root, 'frame')
    try:
        frame = frames[frame_id]
    except:
        return online_im
    if opt.dataset=='icdar':
        assert int(frame_id)+1==int(frame.attrib['ID'])
    elif opt.dataset=='minetto':
        assert int(frame_id)==int(frame.attrib['ID'])
    elif opt.dataset=='roadtext_test':
        assert int(frame_id)+1==int(frame.attrib['ID'])

    objects = get(frame, 'object')
    for obj in objects:
        try:
            quality = obj.attrib['Quality']  # ['MODERATE', 'LOW', 'HIGH', 'MODERTE']
            if quality=='LOW':
                continue
        except: quality='HIGH'
        Points = get(obj, 'Point')
        xs = []
        ys = []
        for point in Points:
            xs.append(float(point.attrib['x']))
            ys.append(float(point.attrib['y']))
        cv2.polylines(online_im, [np.array([[int(xs[0]),int(ys[0])],[int(xs[1]),int(ys[1])], \
            [int(xs[2]),int(ys[2])],[int(xs[3]),int(ys[3])]])], True, (255,255, 255),1)
    return online_im

def eval_seq(opt, dataloader, video_name, frame_dir=None, show_image=False, video_writer=None, timer=None):

    tracker = JDETracker(opt, frame_rate=dataloader.frame_rate)
    results = []
    frame_id = 0
    pre_img0 = None
    pre_boxes = None
    
    for img_path, img0 in dataloader:
        boxes=[]

        # run tracking
        online_targets, pre_img0, pre_boxes = tracker.update(img_path, img0, add_vot_track=opt.add_vot_track, \
            pre_img0=pre_img0, pre_boxes=pre_boxes)
        online_ids = []
        for t in online_targets:
            tlwh = t._tlwh
            tid = t.track_id
            if tlwh[2] * tlwh[3] >= opt.min_box_area:
                boxes.append(list(t.pt)+[int(tid)])
                online_ids.append(tid)
        results.append(boxes)

        if opt.eval_det:
            eval_det_dir = osp.join(opt.output_root, 'det_res')
            if not osp.exists(eval_det_dir):
                os.makedirs(eval_det_dir)
            utils.save_det_res(img_path.split('/')[-1].replace('jpg', 'txt'), video_name, pre_boxes, eval_det_dir, opt.dataset)

        if show_image:
            utils.mkdir_if_missing(frame_dir)
            pred_im = img0.copy()
            for i in range(len(boxes)):
                cv2.polylines(pred_im, [np.array(boxes[i][:8]).reshape(-1, 2).astype(np.int32)], True, (127,255,0),2)
                cv2.putText(pred_im,'{}'.format(int(boxes[i][8])),(int(pre_boxes[i][0]), \
                    int(pre_boxes[i][1])),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1,cv2.LINE_AA)
            cv2.imwrite(os.path.join(frame_dir, '{:05d}_pred.jpg'.format(frame_id)), pred_im)
        if opt.show_gt:
            assert opt.gt_dir != '', 'should give gt dir when show_gt is True'
            gt_im = img0.copy()
            gt_name = video_name + '_GT.xml'
            xml_f = os.path.join(opt.input_root, gt_name)
            gt_im = draw_gt(xml_f, gt_im, frame_id,opt)
            cv2.imwrite(os.path.join(frame_dir, '{:05d}_gt.jpg'.format(frame_id)), gt_im)

        frame_id += 1

    # save results
    if opt.sub_res:
        save_dir = os.path.join(opt.output_root, opt.sub_res_root)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        if opt.dataset == 'icdar':
            videonum = video_name.split('_')[1]
            xml_name = os.path.join(save_dir, 'res_video_'+videonum+'.xml')
            utils.write2xml(xml_name, results, change_id=True)
        elif opt.dataset == 'roadtext' or opt.dataset=='minetto':
            txt_name = os.path.join(save_dir, video_name+'.txt')
            utils.write2txt(txt_name, results)