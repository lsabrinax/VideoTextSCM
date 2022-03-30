import argparse
import json
import math
import torch
import logging
import os
import cv2
import shutil
from torch.utils.collect_env import get_pretty_env_info
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from utils.log import add_file_handler, init_log, print_speed
from utils.parse_config import load_config
from utils.timer import Timer
from utils.meters import AverageMeter
from scm.datasets.scm_dataset import DataSets
from scm.experiments.siammask_sharp.custom import Custom
from scm.utils.load_helper import load_pretrain, restore_from
from scm.utils.lr_helper import build_lr_scheduler

torch.backends.cudnn.benchmark = True

def collect_env_info():
    env_str = get_pretty_env_info()
    env_str += "\n        OpenCV ({})".format(cv2.__version__)
    return env_str

def build_data_loader(cfg):

    logger.info("build train dataset")  # train_dataset
    train_set = DataSets(cfg['train_datasets'], cfg['anchors'], args.save_dir, args.epochs)
    train_set.shuffle()

    logger.info("build val dataset")  # val_dataset
    if not 'val_datasets' in cfg.keys():
        cfg['val_datasets'] = cfg['train_datasets']
    val_set = DataSets(cfg['val_datasets'], cfg['anchors'])
    val_set.shuffle()

    train_loader = DataLoader(train_set, batch_size=args.batch, num_workers=args.workers,
                              pin_memory=True, sampler=None)
    val_loader = DataLoader(val_set, batch_size=args.batch, num_workers=args.workers,
                            pin_memory=True, sampler=None)

    logger.info('build dataset done')
    return train_loader, val_loader

def build_opt_lr(model, cfg, args, epoch):
    trainable_params = model.mask_model.param_groups(cfg['lr']['start_lr'], cfg['lr']['mask_lr_mult']) + \
                       model.refine_model.param_groups(cfg['lr']['start_lr'], cfg['lr']['mask_lr_mult'])

    optimizer = torch.optim.SGD(trainable_params, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    lr_scheduler = build_lr_scheduler(optimizer, cfg['lr'], epochs=args.epochs)

    return optimizer, lr_scheduler


def main():
    global logger, tb_writer
    args = parser.parse_args()
    init_log('global', logging.INFO)

    if args.log != "":
        add_file_handler('global', os.path.join(args.save_dir, args.log_dir, args.log), logging.INFO)

    logger = logging.getLogger('global')
    logger.info("\n" + collect_env_info())
    logger.info(args)
    cfg = load_config(args)
    logger.info("config \n{}".format(json.dumps(cfg, indent=4)))
    tb_writer = SummaryWriter(os.path.join(args.save_dir, args.log_dir))


    # build dataset
    train_loader, val_loader = build_data_loader(cfg)
    model = Custom(anchors=cfg['anchors'])
    logger.info(model)
    if args.pretrained:
        model = load_pretrain(model, args.pretrained)
    model = model.cuda()
    dist_model = torch.nn.DataParallel(model, list(range(torch.cuda.device_count()))).cuda()
    if args.resume and args.start_epoch != 0:
        model.features.unfix((args.start_epoch - 1) / args.epochs)

    optimizer, lr_scheduler = build_opt_lr(model, cfg, args, args.start_epoch)
    # optionally resume from a checkpoint
    if args.resume:
        assert os.path.isfile(args.resume), '{} is not a valid file'.format(args.resume)
        model, optimizer, args.start_epoch, arch = restore_from(model, optimizer, args.resume)
        dist_model = torch.nn.DataParallel(model, list(range(torch.cuda.device_count()))).cuda()

    logger.info(lr_scheduler)
    logger.info('model prepare done')
    logger.info('start training')
    train(train_loader, dist_model, optimizer, lr_scheduler, args.start_epoch, cfg)
    logger.info('end training')

def BNtoFixed(m):
    class_name = m.__class__.__name__
    if class_name.find('BatchNorm') != -1:
        m.eval()

def train(train_loader, model, optimizer, lr_scheduler, epoch, cfg):

    cur_lr = lr_scheduler.get_cur_lr()
    batch_time = Timer()
    mask_loss = AverageMeter()
    iou_mean = AverageMeter()
    iou_at_5 = AverageMeter()
    iou_at_7 = AverageMeter()
    model.train()
    model.module.features.eval()
    model.module.rpn_model.eval()
    model.module.features.apply(BNtoFixed)
    model.module.rpn_model.apply(BNtoFixed)
    model.module.mask_model.train()
    model.module.refine_model.train()
    model = model.cuda()

    num_per_epoch = len(train_loader.dataset) // args.epochs // args.batch
    start_epoch = epoch
    epoch = epoch
    for iter, input in enumerate(train_loader):
        batch_time.tic()
        if epoch != iter // num_per_epoch + start_epoch:
            epoch = iter // num_per_epoch + start_epoch
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            save_checkpoint({
                    'epoch': epoch,
                    'arch': args.arch,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'anchor_cfg': cfg['anchors']
                }, False,
                os.path.join(args.save_dir, 'checkpoint_e%d.pth' % (epoch)),
                os.path.join(args.save_dir, 'best.pth'))
            if epoch == args.epochs:
                return
            lr_scheduler.step(epoch)
            cur_lr = lr_scheduler.get_cur_lr()
            logger.info('epoch:{}'.format(epoch))

        if iter % num_per_epoch == 0 and iter != 0:
            for idx, pg in enumerate(optimizer.param_groups):
                logger.info("epoch {} lr {}".format(epoch, pg['lr']))
                tb_writer.add_scalar('lr/group%d' % (idx+1), pg['lr'], iter)
        x = {
            'cfg': cfg,
            'template': torch.autograd.Variable(input[0]).cuda(),
            'search': torch.autograd.Variable(input[1]).cuda(),
            'label_cls': torch.autograd.Variable(input[2]).cuda(),
            'label_loc': torch.autograd.Variable(input[3]).cuda(),
            'label_loc_weight': torch.autograd.Variable(input[4]).cuda(),
            'label_mask': torch.autograd.Variable(input[6]).cuda(),
            'label_mask_weight': torch.autograd.Variable(input[7]).cuda(),
        }
        outputs = model(x)
        rpn_cls_loss, rpn_loc_loss, rpn_mask_loss = torch.mean(outputs['losses'][0]), \
            torch.mean(outputs['losses'][1]), torch.mean(outputs['losses'][2])
        mask_iou_mean, mask_iou_at_5, mask_iou_at_7 = torch.mean(outputs['accuracy'][0]), \
            torch.mean(outputs['accuracy'][1]), torch.mean(outputs['accuracy'][2])
        cls_weight, reg_weight, mask_weight = cfg['loss']['weight']
        loss = rpn_cls_loss * cls_weight + rpn_loc_loss * reg_weight + rpn_mask_loss * mask_weight
        optimizer.zero_grad()
        loss.backward()
        if cfg['clip']['split']:
            torch.nn.utils.clip_grad_norm_(model.module.features.parameters(), cfg['clip']['feature'])
            torch.nn.utils.clip_grad_norm_(model.module.rpn_model.parameters(), cfg['clip']['rpn'])
            torch.nn.utils.clip_grad_norm_(model.module.mask_model.parameters(), cfg['clip']['mask'])
            torch.nn.utils.clip_grad_norm_(model.module.refine_model.parameters(), cfg['clip']['mask'])
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)  # gradient clip
        optimizer.step()

        batch_time.toc()
        mask_loss.update(rpn_mask_loss.item())
        iou_mean.update(mask_iou_mean.item())
        iou_at_5.update(mask_iou_at_5.item())
        iou_at_7.update(mask_iou_at_7.item())

        tb_writer.add_scalar('loss/mask', rpn_mask_loss.item(), iter)
        tb_writer.add_scalar('mask/mIoU', mask_iou_mean.item(), iter)
        tb_writer.add_scalar('mask/AP@.5', mask_iou_at_5.item(), iter)
        tb_writer.add_scalar('mask/AP@.7', mask_iou_at_7.item(), iter)

        if (iter + 1) % args.print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}] lr: {3:.6f}\tbatch_time:{4:.3f}'
                        '\trpn_mask_loss:{5:.3f}\tmask_iou_mean:{6:.3f}'
                        '\tmask_iou_at_5:{7:.3f}\tmask_iou_at_7:{8:.3f}'.format(
                        epoch+1, (iter + 1) % num_per_epoch, num_per_epoch, cur_lr, batch_time.average_time,
                        mask_loss.avg, iou_mean.avg, iou_at_5.avg, iou_at_7.avg))
            print_speed(iter + 1, batch_time.average_time, args.epochs * num_per_epoch)

def save_checkpoint(state, is_best, filename='checkpoint.pth', best_file='model_best.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_file)

if __name__ == '__main__':
    global args
    parser = argparse.ArgumentParser(description='PyTorch Tracking Training')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 64)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--clip', default=10.0, type=float,
                        help='gradient clip value')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', dest='pretrained', default='',
                        help='use pre-trained model')
    parser.add_argument('--config', dest='config', required=True,
                        help='hyperparameter of SiamMask in json format')
    parser.add_argument('--arch', dest='arch', default='', choices=['Custom',''],
                        help='architecture of pretrained model')
    parser.add_argument('-l', '--log', default="log.txt", type=str,
                        help='log file')
    parser.add_argument('-s', '--save-dir', default='', type=str,
                        help='save dir')
    parser.add_argument('--log-dir', default='board', help='TensorBoard log dir')
    args = parser.parse_args()
    main()
