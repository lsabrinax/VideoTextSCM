from os.path import join, isfile
import json
import glob
from sys import argv
from collections import defaultdict

def gen_json(data_dir):
    imgs = glob.glob(join(data_dir, 'Video_*/*.jpg'))
    n_imgs = len(imgs)
    dataset = defaultdict(dict)
    for i, img in enumerate(imgs):
        ann_file = img+'.txt'
        if not isfile(ann_file):
            continue
        imgid = int(img.split('/')[-1].split('.')[0])
        crop_base_path = join('crop511', img.split('/')[-2])
        
        with open(ann_file, 'r') as f:
            lines = f.readlines()
            assert len(lines) > 0
        for line in lines:
            line = line.strip()
            items = line.split(',')
            pts = list(map(float, items[:8]))
            x0, x1 = min(pts[::2]), max(pts[::2])
            y0, y1 = min(pts[1::2]), max(pts[1::2])
            bbox = list(map(int, [x0, y0, x1, y1]))
            track_id = int(items[8])
            if '{:07d}'.format(track_id) not in dataset[crop_base_path]:
                dataset[crop_base_path]['{:07d}'.format(track_id)]={'{:04d}'.format(imgid): bbox}
            else:
                dataset[crop_base_path]['{:07d}'.format(track_id)].update({'{:04d}'.format(imgid): bbox})
        print('image id: {:04d} / {:04d}'.format(i, n_imgs))
    json.dump(dataset, open(join(data_dir, 'icdar2015.json'), 'w'), indent=4, sort_keys=True)
    print('done!')

if __name__ == '__main__':
    gen_json(argv[1])