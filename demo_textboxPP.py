import os
import os.path as osp
import argparse
from utils import log, utils
from track_textboxPP import eval_seq
import warnings

def track(opt):
    
    videos = os.listdir(opt.input_root)
    videos = [v for v in videos if v.endswith(opt.suffix)]
    if opt.input_format == 'video':
        from dataset import LoadVideo as DataSet
    else:
        pass

    for video in videos:
        print(video)
        input_video = os.path.join(opt.input_root, video)
        video_name = video.split('.')[0] #Video_1_1_2
        result_root = os.path.join(opt.output_root, video_name)
        utils.mkdir_if_missing(result_root)
        opt.result_root = result_root
        dataloader = DataSet(input_video)
        frame_dir = None if opt.output_format=='txt' else osp.join(result_root, 'frames')
        logger.info('start tracking...')
        eval_seq(opt, dataloader, video_name, frame_dir=frame_dir)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', type=str, default='mp4')
    parser.add_argument('--input-root', type=str, help='path to the input video')
    parser.add_argument('--input-format', type=str, default='video', help='expected input format, can be video, or image')
    parser.add_argument('--output-root', type=str, default='results', help='expected output root path')
    parser.add_argument('--output-format', type=str, default='video', help='expected output format, can be video, or text')
    parser.add_argument('--add-vot-track', action='store_false', help='whether use SCM Module')    
    parser.add_argument('--show-gt', action='store_true')
    parser.add_argument('--gt-dir', type=str, default='')
    parser.add_argument('--min-box-area', type=float, default=200, help='filter out tiny boxes')
    parser.add_argument('--dataset', type=str, default='icdar', help='icdar or minetoo')    
    parser.add_argument('--sub-res', action='store_true')
    parser.add_argument('--sub-res-root', type=str, default='ourmodel', help='sub dir to save submit files')
    parser.add_argument('--conf-thresh', type=float, default=0.65, help='object confidence threshold')
    parser.add_argument('--weight-path', type=str, default='./db_model/db_embedding/weights/experiment9_cat_5*16_STL/db_embedding_weight_epoch100.pth', help='path to the model of DB_Embedding')
    parser.add_argument('--img-min-size', type=int, default=1280, help='the shorter side of input img')
    parser.add_argument('--scm-config', type=str, default='./scm/experiments/siammask_sharp/config.json', help='path to the config of scm')
    parser.add_argument('--scm-weight-path', type=str, default='./scm/experiments/siammask_sharp/snapshot/checkpoint_e19.pth', help='path to the model of scm')
    parser.add_argument('--eval_det', action='store_true')
    parser.add_argument('--save-det-out', action='store_true', help='whether to save det model output')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    opt = parse_args()
    global logger
    logger = log.init_log('root')
    log.add_file_handler('root', osp.join(opt.output_root, 'log.txt'))
    logger.info(opt)
    track(opt)