import argparse
import torch
import torch.optim as optim
import os
from pathlib import Path
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from db_model.model_v3 import DB_Embedding_Model
from utils.log import init_log, add_file_handler
from utils.meters import AverageMeter
from utils.timer import Timer
from dataset import VideoDataset

def spatical_triplet_loss(roi_feats, cur_boxes, next_boxes, triple_lists):
    assert len(cur_boxes) == len(triple_lists)
    batch_size = len(next_boxes)
    assert batch_size == len(cur_boxes)
    alpha1 = 0.5
    alpha2 = 1.5
    losses = 0
    cur_boxes_num = sum([boxes.shape[0] for boxes in cur_boxes])
    cur_roi_feat = roi_feats[:cur_boxes_num]
    next_roi_feat = roi_feats[cur_boxes_num:]
    cur_num = next_num = 0
    for i, triple_list in enumerate(triple_lists):
        cur_i_num = cur_boxes[i].shape[0]
        next_i_num = next_boxes[i].shape[0]
        roi_feat = torch.cat([cur_roi_feat[cur_num:cur_num+cur_i_num], next_roi_feat[next_num:next_num+next_i_num]])
        cur_num += cur_i_num
        next_num += next_i_num
        num_of_triple = triple_list.shape[0]
        if num_of_triple == 0:
            continue
        if num_of_triple > 2000:
            rand_triple = torch.randint(0,num_of_triple, size=(2000,)).long()
            triple_list = triple_list[rand_triple,:]
        
        triple_feat = roi_feat[triple_list]
        all_boxes = torch.cat([cur_boxes[i], next_boxes[i]])
        triple_box = all_boxes[triple_list]

        pos_dist = torch.sum(torch.pow(triple_feat[:,0,:] - triple_feat[:,1,:],2), 1)
        neg_dist = torch.sum(torch.pow(triple_feat[:,0,:] - triple_feat[:,2,:],2), 1)
        anchor_center = (triple_box[:, 0, 2:] + triple_box[:, 0, :2])/2
        positive_center = (triple_box[:, 1, 2:] + triple_box[:, 1, :2])/2
        negative_center = (triple_box[:, 2, 2:] + triple_box[:, 2, :2])/2
        W_scale = torch.exp(1-torch.sum(torch.abs(triple_box[:, 0, 2:] - triple_box[:, 0, :2]), 1)/2)
        W_pos_dis = 1-torch.exp(-(torch.sqrt(torch.sum(torch.pow(anchor_center - positive_center, 2),1))))
        W_neg_dis = 1-torch.exp(-(torch.sqrt(torch.sum(torch.pow(anchor_center - negative_center, 2),1))))
        loss_pos = pos_dist*(W_scale+W_pos_dis) - alpha1
        loss_neg = alpha2 - neg_dist*W_neg_dis
        loss = torch.sum(torch.max(loss_pos, torch.zeros_like(loss_pos)) + torch.max(loss_neg, torch.zeros_like(loss_neg)), 0)/triple_box.shape[0]
        losses += loss

    return losses / batch_size

def collect_fn(batch):
    cur_imgs, next_imgs, cur_boxes, next_boxes, triple_lists = zip(*batch)
    cur_imgs = torch.stack(cur_imgs)
    next_imgs = torch.stack(next_imgs)
    return cur_imgs, next_imgs, list(cur_boxes), list(next_boxes), list(triple_lists)

def train(opt):

    exp_dir = Path(opt.exp_name)
    log_dir = 'db_model/log' / exp_dir 
    save_dir = 'db_model/weights' / exp_dir

    #init and load weight for DB
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    net = DB_Embedding_Model()
    state = torch.load(opt.weight_path)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    net.db.load_state_dict(state, strict=True)
    net = net.to(device)
    net.db.eval()
    for name,param in net.named_parameters():
        if 'db' in name:
            param.requires_grad = False

    #optimizer
    optimizer = optim.RMSprop(net.parameters(), lr=opt.lr, alpha=0.9, eps=1e-4, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[15,30,45,50,75], gamma=0.3)
    # log
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    logger = init_log('embed')
    add_file_handler('embed',os.path.join(save_dir, "log.txt"))

    #dataset
    train_data = VideoDataset(opt.img_dir, opt.gt_dir, opt.train_list_file)
    train_loader = DataLoader(train_data, batch_size=opt.batch_size, \
        num_workers=opt.num_workers, shuffle=True, collate_fn=collect_fn)

    batch_time = Timer()
    data_time = Timer()
    losses = AverageMeter()

    global_step = 0
    for epoch in range(opt.epoch_num):
        lr = scheduler.get_last_lr()[0]
        logger.info(f'now learning rate is {lr}')
        for i, input_ in enumerate(train_loader):
            global_step += 1
            batch_time.tic()
            data_time.tic()
            cur_imgs, next_imgs, cur_boxes, next_boxes, triple_lists = input_
            imgs = torch.cat([cur_imgs, next_imgs]).to(device)
            cur_boxes = [boxes.to(device) for boxes in cur_boxes]
            next_boxes = [boxes.to(device) for boxes in next_boxes]
            triple_lists = [triple_list.to(device) for triple_list in triple_lists]
            all_boxes = cur_boxes + next_boxes
            data_time.toc()
            
            pred, roi_feats = net(imgs, all_boxes)
            loss_embd = spatical_triplet_loss(roi_feats, cur_boxes, next_boxes, triple_lists)
            loss = loss_embd
            if loss <= 0:
                continue

            writer.add_scalar('Loss/train_cur', loss.item(), global_step)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item())
            batch_time.toc()
            writer.add_scalar('Loss/train_avg', losses.avg, global_step)

            if global_step % 50 == 0:
                logger.info("epoch:[{}/{}] iter:[{}/{}] data_time:{:.3f} batch_time:{:.3f} || loss:{:.4f}/{:.4f}".format(
                    epoch, opt.epoch_num,
                    i+1, len(train_loader),
                    data_time.average_time,
                    batch_time.average_time,
                    losses.val, losses.avg
                ))
        scheduler.step()
        if (epoch+1) % 20 == 0:
            save_path = os.path.join(save_dir, "db_embedding_weight_epoch{}.pth".format(epoch+1))
            torch.save(net.state_dict(), save_path)
            logger.info("save weight at epoch={} in {}".format(epoch, save_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='./datasets/video_train')
    parser.add_argument('--gt_dir', type=str, default='./datasets/video_train')
    parser.add_argument('--train_list_file', type=str, default='./datasets/video_train/db_train_valid_pair_list.txt')
    parser.add_argument('--weight_path', type=str, default="./db_model/weights/totaltext_resnet50")
    parser.add_argument('--exp_name', type=str, default='exp')
    parser.add_argument('--epoch_num', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    opt = parser.parse_args()
    train(opt)


