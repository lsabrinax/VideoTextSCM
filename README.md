

# Introduction
This is a PyToch implementation of [Video Text Tracking With a Spatio-Temporal Complementary Model](https://arxiv.org/abs/2111.04987). 

Part of the code is inherited from [DB](https://github.com/MhLiao/DB).
Part of the code is inherited from [SiamMask](https://github.com/foolwood/SiamMask).
## ToDo List

- [x] Release code
- [x] Document for Installation
- [x] Document for training and testing 



## Installation

### Requirements:
- Python 3.6
- PyTorch >= 1.2 
- GCC 5.5
- CUDA 9.2


```bash

  conda create --name scm python=3.6
  conda activate scm

  # install PyTorch with cuda-9.2
  conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=9.2 -c pytorch

  # python dependencies
  pip install -r requirement.txt

  # clone repo
  git clone https://github.com/lsabrinax/VideoTextSCM
  cd VideoTextSCM/

```



## Datasets
The root of the dataset directory can be ```VideoTextSCM/datasets/```.
Download ICDAR 2015 Video Dataset in [official website](https://rrc.cvc.uab.es/?ch=3&com=downloads), and unzip them in dataset directory.


## Testing
run the below command to get the tracking results and submit the results to official website to get the performance

```CUDA_VISIBLE_DEVICES=0 python demo_textboxPP.py --input-root path-to-test-dataset --output-root path-to-save-result --sub-res --dataset icdar --weight-path path-to-embedding-model --scm-config path-to-scm-config --scm-weight-path path-to-scm-model```


## Training
### SCM
```
# prepare datasets
cd VideoTextSCM/scm/datasets
python par_crop.py --enable_mask
python gen_json.py '../../datasets/ch3_train'

#download the pre-trained model
cd VideoTextSCM/scm/experiments/siammask_sharp
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_VOT.pth

#train the model
cd VideoTextSCM
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_scm.py --save-dir path-to-save-scm-model --pretrained ./scm/experiments/siammask_sharp/SiamMask_VOT.pth --config ./scm/experiments/siammask_sharp/config_icdar.json --batch 256 --epochs 20 --clip 5
```

### Embedding
```
cd VideoTextSCM
CUDA_VISIBLE_DEVICES=0 python train_embedding.py --exp_name model-name --batch_size 3 --num_workers 8 --lr 0.0005
```



## Citing the related works

Please cite the related works in your publications if it helps your research:

  @article{gao2021video,
    title={Video Text Tracking With a Spatio-Temporal Complementary Model},
    author={Gao, Yuzhe and Li, Xing and Zhang, Jiajian and Zhou, Yu and Jin, Dian and Wang, Jing and Zhu, Shenggao and Bai, Xiang},
    journal={IEEE Transactions on Image Processing},
    volume={30},
    pages={9321--9331},
    year={2021},
    publisher={IEEE}
  }



