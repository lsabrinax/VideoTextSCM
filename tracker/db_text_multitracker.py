import torch
import os.path as osp
import os
from db_model.db_embedding import Demo
from .basetrack import *
from .kalman_filter import KalmanFilter

class JDETracker(object):
    def __init__(self, opt, frame_rate=30):
        self.save_root = osp.join(opt.result_root, 'det_out')
        self.save_det_out = opt.save_det_out
        self.model = Demo(opt.weight_path, opt.scm_weight_path, opt.scm_config,opt.img_min_size, opt.conf_thresh)
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.frame_id = -1
        self.buffer_size = frame_rate
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        STrack.new_video()

    def update(self, img_path, img0, add_vot_track=0, pre_img0=None, pre_boxes=None):

        self.frame_id += 1
        print('================frame_id: {}================='.format(self.frame_id))
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        strack_pool = joint_stracks(self.tracked_stracks, self.lost_stracks)

        ''' Step 1: Network forward, get detections & embeddings
            pred_f: shape=(n,512)
            pred_boxes: shape=(n, 4)  [x1,y1,x3,y3]
            pre_boxes: shape=(n,9)  [x1,y1,x2,y2,x3,y3,x4,y4,score]
        '''
        with torch.no_grad():
            pred_pt = img_path.split('/')[-1].replace('.jpg', '_pred.pt')
            pred_pt = osp.join(self.save_root, pred_pt)
            if not os.path.isfile(pred_pt):
                pred_f, pred_boxes, pre_boxes, no_objs = self.model.inference(img_path,img0, pre_img0=pre_img0, pre_boxes=pre_boxes, add_vot_track=add_vot_track)
                if self.save_det_out:
                    if not os.path.exists(self.save_root):
                        os.makedirs(self.save_root)
                    save_dict = {'pred_f': pred_f, 'pred_boxes': pred_boxes, 'pre_boxes':pre_boxes, 'no_objs': no_objs}
                    torch.save(save_dict, pred_pt)
            else:
                save_dict = torch.load(pred_pt)
                pred_f, pred_boxes, pre_boxes, no_objs = save_dict['pred_f'], save_dict['pred_boxes'], save_dict['pre_boxes'], save_dict['no_objs']

        if no_objs == False and len(pred_f) > 0:
            detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, f.numpy(), pt.clone().numpy(), self.buffer_size, self.frame_id, img_path) for
                          (tlbr, s, f, pt) in zip(pred_boxes, pre_boxes[:, 8], pred_f, pre_boxes[:, :8])]
        else:
            detections = []

        ''' Step 2: association, with embedding, iou and shape'''
        for strack in strack_pool:
            strack.predict()
        dists1 = matching.embedding_distance(strack_pool, detections)
        dists2 = matching.poly_distance(strack_pool, detections)
        dists3 = matching.shape_distance(strack_pool, detections)
        # dists = 0.6*dists1 + 0.4*dists3
        dists = 0.6*dists1 + 0.2*dists2 + 0.2*dists3
        # dists = 0.4*dists1 + 0.3*dists2 + 0.3*dists3
        # dists = 0.8*dists1 + 0.1*dists2 + 0.1*dists3 
        # dists = 0.6*dists1 + 0.3*dists2 + 0.1*dists3
        # dists = 0.6*dists1 + 0.1*dists2 + 0.3*dists3
        # dists = 0.5*dists1 + 0.2*dists2 + 0.3*dists3
        # dists = 0.5*dists1 + 0.3*dists2 + 0.2*dists3
        # dists = 0.5*dists1 + 0.25*dists2 + 0.25*dists3 
        # dists = 0.7*dists1 + 0.2*dists2 + 0.1*dists3
        # dists = 0.7*dists1 + 0.1*dists2 + 0.2*dists3
        # dists = 0.7*dists1 + 0.15*dists2 + 0.15*dists3 
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=0.7)

        ''' step3: Add newly detected tracklets to tracked_stracks'''
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        '''step4: mark the losted stracks'''
        for it in u_track:
            track = strack_pool[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        """ Step 5: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 6: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = [t for t in self.lost_stracks if t.state == TrackState.Lost]
        self.lost_stracks.extend(lost_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        
        return output_stracks,img0, pre_boxes