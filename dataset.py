import numpy as np
import cv2
import random
import os
from torch.utils.data import Dataset
from utils import utils
import torch

class VideoDataset(Dataset):
    def __init__(self,img_dir, gt_dir, train_list_path, size=(1280, 1280)):
        '''
        size = (resize_w,resize_h)
        '''
        with open(train_list_path, 'r') as fopen:
            train_list = [line.strip() for line in fopen.readlines()]
        self.train_list = train_list
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.size = size
        self.RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])

    def __len__(self):
        return len(self.train_list)

    def load_img(self, img_name):
        image_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
        original_shape = img.shape[:2]  #(h,w)
        img = cv2.resize(img, self.size)
        img -= self.RGB_MEAN
        img /= 255.
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img, original_shape

    def perturb_box(self, box, min_iou=0.5, sigma_factor=0.1):
        """ Perturb the input box by adding gaussian noise to the co-ordinates
        args:
            box - input box  top-left and w、h
            min_iou - minimum IoU overlap between input box and the perturbed box
            sigma_factor - amount of perturbation, relative to the box size. Can be either a single element, or a list of
                            sigma_factors, in which case one of them will be uniformly sampled. Further, each of the
                            sigma_factor element can be either a float, or a tensor
                            of shape (4,) specifying the sigma_factor per co-ordinate
        returns:
            torch.Tensor - the perturbed box
        """
        def rand_uniform(a, b, shape=1):
            """ sample numbers uniformly between a and b.
            args:
                a - lower bound
                b - upper bound
                shape - shape of the output tensor
            returns:
                torch.Tensor - tensor of shape=shape
            """
            return (b - a) * torch.rand(shape) + a
            
        if isinstance(sigma_factor, list):
            # If list, sample one sigma_factor as current sigma factor
            c_sigma_factor = random.choice(sigma_factor)
        else:
            c_sigma_factor = sigma_factor
        if not isinstance(c_sigma_factor, torch.Tensor):
            c_sigma_factor = c_sigma_factor * torch.ones(4)
        perturb_factor = torch.sqrt(box[2]*box[3])*c_sigma_factor

        # multiple tries to ensure that the perturbed box has iou > min_iou with the input box
        for i_ in range(100):
            c_x = box[0] + 0.5*box[2]
            c_y = box[1] + 0.5 * box[3]
            c_x_per = random.gauss(c_x, perturb_factor[0])
            c_y_per = random.gauss(c_y, perturb_factor[1])
            w_per = random.gauss(box[2], perturb_factor[2])
            h_per = random.gauss(box[3], perturb_factor[3])

            if w_per <= 1:
                w_per = box[2]*rand_uniform(0.15, 0.5)
            if h_per <= 1:
                h_per = box[3]*rand_uniform(0.15, 0.5)
            box_per = torch.Tensor([c_x_per - 0.5*w_per, c_y_per - 0.5*h_per, w_per, h_per]).round()
            if box_per[2] <= 1:
                box_per[2] = box[2]*rand_uniform(0.15, 0.5)
            if box_per[3] <= 1:
                box_per[3] = box[3]*rand_uniform(0.15, 0.5)
            box_iou = utils.iou(box.view(1, 4), box_per.view(1, 4))

            # if there is sufficient overlap, return
            if box_iou > min_iou:
                return box_per
            # else reduce the perturb factor
            perturb_factor *= 0.9

        return box

    def get_all_boxes(self, cur_txt, next_txt ,cur_original_shape, next_original_shape):
        cur_txt_path = os.path.join(self.gt_dir, cur_txt)
        next_txt_path = os.path.join(self.gt_dir, next_txt)

        cur_h, cur_w = cur_original_shape
        next_h, next_w = next_original_shape
        resize_w, resize_h = self.size

        cur_boxes = []
        cur_objs = []
        with open(cur_txt_path, 'rb') as fopen:
            lines = [line for line in fopen.readlines()]
        lines = [line.decode("utf-8").strip() for line in lines]
        for line in lines:
            line_split = line.split(',')
            box = list(map(int, map(float, line_split[:8])))
            x1,y1,x2,y2,x3,y3,x4,y4 = box
            region_point = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]]) #shape=(k，2)
            x1,y1,w,h = cv2.boundingRect(region_point)
            x1,y1,w,h = self.perturb_box(torch.Tensor([x1,y1,w,h]), 0.8)
            x1,y1,x3,y3 = int(x1*resize_w/cur_w), int(y1*resize_h/cur_h), int((x1+w)*resize_w/cur_w), int((y1+h)*resize_h/cur_h) 
            x1 = min(resize_w-1, max(0, x1))
            x3 = min(resize_w-1, max(0, x3))
            y1 = min(resize_h-1, max(0, y1))
            y3 = min(resize_h-1, max(0, y3))  
            cur_boxes.append([x1,y1,x3,y3])
            cur_objs.append(int(line_split[8]))
        
        next_boxes = []
        next_objs = []
        with open(next_txt_path, 'rb') as fopen:
            lines = [line for line in fopen.readlines()]
        lines = [line.decode("utf-8").strip() for line in lines]
        for line in lines:
            line_split = line.split(',')
            box = list(map(int, map(float, line_split[:8])))
            x1,y1,x2,y2,x3,y3,x4,y4 = box
            region_point = np.array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]]) #shape=(k，2)
            x1,y1,w,h = cv2.boundingRect(region_point)
            x1,y1,w,h = self.perturb_box(torch.Tensor([x1,y1,w,h]), 0.8)
            x1,y1,x3,y3 = int(x1*resize_w/next_w), int(y1*resize_h/next_h), int((x1+w)*resize_w/next_w), int((y1+h)*resize_h/next_h)
            x1 = min(resize_w-1, max(0, x1))
            x3 = min(resize_w-1, max(0, x3))
            y1 = min(resize_h-1, max(0, y1))
            y3 = min(resize_h-1, max(0, y3))
            next_boxes.append([x1,y1,x3,y3])
            cur_objs.append(int(line_split[8]))
        cur_boxes = torch.Tensor(cur_boxes)
        next_boxes = torch.Tensor(next_boxes)
        return cur_boxes, next_boxes, cur_objs, next_objs

    def get_triple_list(self,cur_objs,next_objs):
        def get_pair(obj_index):
            pairs = []
            for i in range(len(obj_index)):
                for j in range(i+1, len(obj_index)):
                    pairs.append([obj_index[i], obj_index[j]])
            return pairs

        objs = cur_objs+next_objs
        objs = np.array(objs)
        obj_id_unique = np.unique(objs)

        objs_index = []
        for obj_id in obj_id_unique:
            objs_index.append(np.where(objs==obj_id)[0].tolist())

        triple_list = []
        for i in range(len(objs_index)):
            obj_index = objs_index[i]
            if len(obj_index) < 2:
                continue
            pairs = get_pair(obj_index)
            for j in range(len(objs_index)):
                if j == i:
                    continue
                for neg in objs_index[j]:
                    for pair in pairs:
                        triple = pair+[neg]
                        triple_list.append(triple.copy())
        return triple_list

    def __getitem__(self, index):
        img_pair = self.train_list[index]
        imgcur_name, imgnext_name = img_pair.split(',')
        cur_txt, next_txt = imgcur_name+'.txt', imgnext_name+'.txt'
        imgcur, cur_original_shape = self.load_img(imgcur_name)
        imgnext , next_original_shape= self.load_img(imgnext_name)
        cur_boxes, next_boxes, cur_objs, next_objs = self.get_all_boxes(cur_txt, next_txt ,cur_original_shape, next_original_shape)
        triple_list = self.get_triple_list(cur_objs, next_objs)
        triple_list = torch.Tensor(triple_list).long()
        return imgcur, imgnext, cur_boxes, next_boxes, triple_list

class LoadVideo:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)        
        self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS)))
        self.vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vn = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.count = 0
        base_name =  os.path.basename(path).split('.')[0]
        self.imgdir = os.path.join(os.path.dirname(path), base_name)
        utils.mkdir_if_missing(self.imgdir)
        print('Lenth of the video: {:d} frames'.format(self.vn))

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == len(self):
            raise StopIteration

        res, img0 = self.cap.read()  # BGR
        assert img0 is not None, 'Failed to load frame {:d}'.format(self.count)

        img_path = os.path.join(self.imgdir, str(self.count)+'.jpg') #frame_id start from 0
        if not os.path.isfile(img_path):
            cv2.imwrite(img_path, img0)

        return img_path, img0
    
    def __len__(self):
        return self.vn  # frames of video