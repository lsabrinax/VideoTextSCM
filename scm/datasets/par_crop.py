import cv2
import numpy as np
from os.path import join, isdir, isfile
from os import makedirs
from concurrent import futures
import glob
import time
import argparse
from pathlib import Path
import sys

# Print iterations progress 
def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()

def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz) / (bbox[2]-bbox[0])
    b = (out_sz) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop

def pos_s_2_bbox(pos, s):
    return [pos[0]-s/2, pos[1]-s/2, pos[0]+s/2, pos[1]+s/2]

def crop_like_SiamFCx(image, bbox, exemplar_size=127, context_amount=0.5, search_size=255, padding=(0, 0, 0)):
    target_pos = [(bbox[2]+bbox[0])/2., (bbox[3]+bbox[1])/2.]
    target_size = [bbox[2]-bbox[0]+1, bbox[3]-bbox[1]+1]
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    d_search = (search_size - exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), search_size, padding)
    return x

def crop_img(img, set_crop_base_path,exemplar_size=127, context_amount=0.5, search_size=511, enable_mask=True):

    frame_crop_base_path = join(set_crop_base_path, img.split('/')[-2])
    if not isdir(frame_crop_base_path): makedirs(frame_crop_base_path)
    # print(frame_crop_base_path)
    im = cv2.imread(img)
    avg_chans = np.mean(im, axis=(0, 1))
    ann_file = img+'.txt'
    if not isfile(ann_file) or 'Video_46_6_4' not in ann_file:
        return
    imgid = int(img.split('/')[-1].split('.')[0])
    with open(ann_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            items = line.split(',')
            pts = list(map(float, items[:8]))
            x0, x1 = min(pts[::2]), max(pts[::2])
            w = int(x1 - x0)
            y0, y1 = min(pts[1::2]), max(pts[1::2])
            h = int(y1 - y0)
            if w * h <= 0:
                continue
            bbox = [x0, y0, x1, y1]
            track_id = int(items[8])
            x = crop_like_SiamFCx(im, bbox, exemplar_size=exemplar_size, context_amount=context_amount,
                              search_size=search_size, padding=avg_chans)
            cv2.imwrite(join(frame_crop_base_path, '{:04d}.{:07d}.x.jpg'.format(imgid, track_id)), x)
            if enable_mask:
                im_mask = np.zeros(im.shape)
                cv2.fillPoly(im_mask, [np.array(pts).reshape(-1,2).astype(np.int64)], (1,1,1))

                x = (crop_like_SiamFCx(im_mask, bbox, exemplar_size=exemplar_size, context_amount=context_amount,
                                   search_size=search_size) > 0.5).astype(np.uint8) * 255
                cv2.imwrite(join(frame_crop_base_path, '{:04d}.{:07d}.m.png'.format(imgid, track_id)), x)

def main(data_dir, exemplar_size=127, context_amount=0.5, search_size=511, enable_mask=True, num_threads=24):
    data_dir = Path(data_dir)
    crop_path = data_dir / 'crop{:d}'.format(search_size)
    if not isdir(crop_path): makedirs(crop_path)

    imgs = glob.glob(join(data_dir, 'Video_*/*.jpg'))
    n_imgs = len(imgs)
    with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        fs = [executor.submit(crop_img, img,crop_path, exemplar_size, 
            context_amount, search_size, enable_mask) for img in imgs]
        for i, f in enumerate(futures.as_completed(fs)):
                printProgress(i, n_imgs, prefix='icdar', suffix='Done ', barLength=40)
    print('done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ICDAR Parallel Preprocessing for SCM')
    parser.add_argument('--exemplar_size', type=int, default=127, help='size of exemplar')
    parser.add_argument('--context_amount', type=float, default=0.5, help='context amount')
    parser.add_argument('--search_size', type=int, default=511, help='size of cropped search region')
    parser.add_argument('--enable_mask', action='store_true', help='whether crop mask')
    parser.add_argument('--num_threads', type=int, default=16, help='number of threads')
    parser.add_argument('--data_dir', type=str, default='../../datasets/video_train', help='dir for data to preprocess')
    args = parser.parse_args()
    since = time.time()
    main(args.data_dir, args.exemplar_size, args.context_amount, \
        args.search_size, args.enable_mask, args.num_threads)
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
