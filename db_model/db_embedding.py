import json
import os
import cv2
import numpy as np
import torch
import math
import logging
from .model_v3 import DB_Embedding_Model
from scm.experiments.siammask_sharp.custom import Custom
from .representers.seg_detector_representer import SegDetectorRepresenter
from utils.utils import make_seg_shrink
from scm.tools.track2mask import track_all_objs2mask

logger = logging.getLogger('root')
class Demo:
    def __init__(self, weight_path, scm_weight_path, scm_config, img_min_size, conf_thresh):
        self.RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793], dtype=np.float32)
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.weight_path = weight_path
        self.scm_weight_path = scm_weight_path
        self.scm_config = scm_config
        self.min_size = img_min_size
        self.init_torch_tensor()
        self.model = self.init_model()
        self.model.eval()
        self.scm = self.init_scm()
        self.scm.eval()
        self.segdetector_representer = SegDetectorRepresenter()
        self.box_thresh = conf_thresh #0.65

    def init_torch_tensor(self):
        # Use gpu or not
        torch.set_default_tensor_type('torch.FloatTensor')
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    def init_model(self):
        logger.info("model init...")
        model = DB_Embedding_Model().to(self.device) 
        logger.info("model init success!")

        logger.info("resume from {}...".format(self.weight_path))
        if not os.path.exists(self.weight_path):
            logger.info("Checkpoint not found: " + self.weight_path)
            return
        states = torch.load(self.weight_path, map_location=self.device)
        model.load_state_dict(states, strict=True)
        return model

    def init_scm(self):
        with open(self.scm_config, 'r') as f:
            cfg = json.load(f)
        self.scm_cfg = cfg
        scm= Custom(anchors=cfg['anchors']).to(self.device)
        logger.info('load pretrained scm model from {}'.format(self.scm_weight_path))
        if not os.path.exists(self.scm_weight_path):
            logger.info("scm Checkpoint not found: " + self.scm_weight_path)
            return
        states = torch.load(self.scm_weight_path, map_location=self.device)
        if 'state_dict' in states:
            states = states['state_dict']

        #remove module
        new_sate = {}
        for key, value in states.items():
            if key.startswith('module.'):
                key = key.split('module.', 1)[-1]
            new_sate[key] = value
        scm.load_state_dict(new_sate, strict=True)
        return scm

    def resize_image(self, img):
        height, width, _ = img.shape
        if height < width:
            new_height = self.min_size  
            new_width = int(math.ceil(new_height / height * width / 32) * 32)
        else:
            new_width = self.min_size
            new_height = int(math.ceil(new_width / width * height / 32) * 32)
        resized_img = cv2.resize(img, (new_width, new_height))
        self.size = (new_width, new_height)
        return resized_img
        
    def load_image(self, img0):
        img = img0.astype(np.float32)
        original_shape = img.shape[:2] #(h,w)
        img = self.resize_image(img)
        resize_img = img.copy()
        img -= self.RGB_MEAN
        img /= 255.
        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) #shape=(1,3,h,w)
        return img, original_shape, resize_img

    def get_output(self, batch_boxes, batch_scores, feat_refine, original_shape, pp=False):
        '''
        batch_boxes: list[np.array]
        batch_boxes[0]: np.array, shape=(n,4,2)
        batch_scores: list[np.array]
        batch_scores[0]: np.array, shape=(n)
        feat_refine: tensor, shape=(n, 256, h/4, w/4)
        original_shape = (h,w)
        ''' 
        assert len(batch_scores) == 1
        assert len(batch_boxes) == 1
        batch_boxes = batch_boxes[0]
        batch_scores = batch_scores[0]

        #Firstly, remove boxes which score below box_thresh
        idx = batch_scores >= self.box_thresh
        boxes_8 = batch_boxes[idx].reshape(-1, 8)
        boxes_8 = torch.from_numpy(boxes_8).float()
        boxes_score = batch_scores[idx]  
        boxes_score = torch.from_numpy(boxes_score)
        if boxes_score.shape[0] < 1:
            boxes_feat = torch.Tensor([]).cpu()
            boxes_4 = torch.Tensor([]).cpu()
            boxes8_score = torch.Tensor([]).cpu()
            return boxes_feat, boxes_4, boxes8_score, True

        #Secondly, get the min area horizontal rectangle
        x0s = torch.min(boxes_8[:,::2],dim=1,keepdim=True)[0]
        y0s = torch.min(boxes_8[:,1::2],dim=1,keepdim=True)[0]
        x2s = torch.max(boxes_8[:,::2],dim=1,keepdim=True)[0]
        y2s = torch.max(boxes_8[:,1::2],dim=1,keepdim=True)[0]
        boxes_4 = torch.cat((x0s,y0s,x2s,y2s),dim=1).float()  
        if pp:
            boxes_area = (x2s-x0s) * (y2s -y0s)
            area_idx = boxes_area<5000
            boxes_4 = boxes_4[area_idx.expand_as(boxes_4)].reshape(-1,4)
            boxes_8 = boxes_8[area_idx.expand_as(boxes_8)].reshape(-1,8)
            boxes_score = boxes_score[area_idx.squeeze(-1)]
        assert boxes_8.shape[0] == boxes_score.shape[0]
        if boxes_score.shape[0] < 1:
            boxes_feat = torch.Tensor([]).cpu()
            boxes_4 = torch.Tensor([]).cpu()
            boxes8_score = torch.Tensor([]).cpu()
            return boxes_feat, boxes_4, boxes8_score, True

        #Thirdly, crop text instances feature
        boxes_4_resize = torch.zeros_like(boxes_4)
        boxes_4_resize[:,::2] = boxes_4[:,::2] / original_shape[1] * self.size[0]
        boxes_4_resize[:,1::2] = boxes_4[:,1::2] / original_shape[0] * self.size[1] 
        zeros = torch.zeros((boxes_4.shape[0], 1)).cpu().float()
        boxes_4_resize = boxes_4_resize.float()
        boxes_4_resize = torch.cat((zeros, boxes_4_resize),dim=1).to(self.device) #shape=(n, 5), 5=[0,x1,y1,x2,y2]
        boxes_feat = self.model.roi_align(feat_refine, boxes_4_resize) #shape=(n,256,5,16)

        #Fourthly, pred sem and vis feature of text instances
        with torch.no_grad():
            obj_num = boxes_feat.shape[0]
            # visual feat
            v_boxes_feat = boxes_feat.view(obj_num, -1) #shape=(n,256*5*16)
            v_boxes_feat = self.model.embed_layers(v_boxes_feat)  #shape=(n,256)
            # Semantic feat
            s_boxes_feat = self.model.seq_conv(boxes_feat)
            s_boxes_feat = s_boxes_feat.squeeze(2).permute(2,0,1).contiguous()
            s_boxes_feat = self.model.rnn(s_boxes_feat).permute(1, 2, 0).contiguous().view(obj_num, -1)
            boxes_feat = torch.cat((v_boxes_feat, s_boxes_feat), -1)
            # boxes_feat = s_boxes_feat

        boxes_feat = boxes_feat.cpu()  #shaep=(n,512)
        boxes8_score = torch.cat((boxes_8, boxes_score.view(-1,1)), dim=1)
        return boxes_feat, boxes_4, boxes8_score, False

    def inference(self, image_path, img0, add_vot_track=False, pre_img0=None, pre_boxes=None, add_mask=True):

        batch = dict()
        batch['filename'] = [image_path]
        pp = '7_4' in image_path
        img, original_shape, resize_img = self.load_image(img0)
        batch['shape'] = [original_shape]
        
        with torch.no_grad():
            img = img.to(self.device)
            batch['image'] = img
            pred, feat_refine = self.model.forward(img)

            if add_vot_track and torch.is_tensor(pre_boxes) and pre_boxes.shape[0] > 0:
                pre_img, pre_original_shape,pre_resize_img = self.load_image(pre_img0)
                pre_boxes[:,:8:2] *= pre_resize_img.shape[1]/pre_original_shape[1]
                pre_boxes[:,1:8:2] *= pre_resize_img.shape[0]/pre_original_shape[0]
                track_mask, polygons = track_all_objs2mask(pre_resize_img, resize_img, pre_boxes, self.device, self.scm, self.scm_cfg)
                shrink_mask = make_seg_shrink(polygons, [0 for i in range(polygons.shape[0])], resize_img.shape[0], resize_img.shape[1])
                track_mask *= shrink_mask
                track_mask = torch.Tensor(track_mask).to(self.device)
                track_mask = track_mask.unsqueeze(0).unsqueeze(0)
                track_mask[pred<0.6]=0 #0.6
                pred += track_mask
                output = self.segdetector_representer.represent(batch, pred, is_output_polygon=False)
                batch_boxes, batch_scores = output
            else:
                output = self.segdetector_representer.represent(batch, pred, is_output_polygon=False)
                batch_boxes, batch_scores = output

            pred_f, boxes_4, boxes8_score, no_objs = self.get_output(batch_boxes, batch_scores, feat_refine, original_shape, pp)
            return pred_f, boxes_4, boxes8_score, no_objs
