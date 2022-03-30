class AverageMeter(object):
    """computes and stores the average and current value"""
    def __init__(self, infos=None):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.infos = infos

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count
