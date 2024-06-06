import numpy as np
import torch
import pandas


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return round(self.v, 8)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
        self.items = []
        self.var = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += float(val * n)
        self.count += n
        self.items.append(val)
        self.avg = round(self.sum / self.count, 4)
        squared_diff = [(x - self.avg) ** 2 for x in self.items]
        self.var  = round(sum(squared_diff) / len( self.items),4)
    def average(self):
        return self.avg
    
    def variance(self):
        return self.var

class LAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = []
        self.avg = []
        self.sum = []
        self.count = 0

    def update(self, val):
        self.val = val
        self.count += 1
        if len(self.sum) == 0:
            assert (self.count == 1)
            self.sum = [v for v in val]
            self.avg = [round(v, 4) for v in val]
        else:
            assert (len(self.sum) == len(val))
            for i, v in enumerate(val):
                self.sum[i] += v
                self.avg[i] = round(self.sum[i] / self.count, 8)


class DAverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.values = {}

    def update(self, values):
        assert (isinstance(values, dict))
        for key, val in values.items():
            if isinstance(val, (float, int)):
                if not (key in self.values):
                    self.values[key] = AverageMeter()
                self.values[key].update(val)
            elif isinstance(val, np.ndarray):
                val = float(val)
                if not (key in self.values):
                    self.values[key] = AverageMeter()
                self.values[key].update(val)
            elif isinstance(val, dict):
                if not (key in self.values):
                    self.values[key] = DAverageMeter()
                self.values[key].update(val)
            elif isinstance(val, list):
                if not (key in self.values):
                    self.values[key] = LAverageMeter()
                self.values[key].update(val)

    def average(self):
        average = {}
        for key, val in self.values.items():
            if isinstance(val, type(self)):
                average[key] = val.average()
            else:
                average[key] = val.avg
        return average


def get_aver(cls, da):
    avger = Averager()
    for i in cls:
        if i in da:
            avger.add(da[i])
    return avger.item()


def count_per_cls_acc(pred, true_label):
    # pred = torch.argmax(pred, dim=1)
    acc_dict = {}
    cls_sample_count = {}
    for cls in true_label.unique():
        indices = torch.where(true_label==cls, 1, 0)
        idx = torch.nonzero(indices)
        device = torch.device("cuda:1")

        pred = pred.to(device)
        true_label = true_label.to(device)
        idx = idx.to(device)
        if torch.cuda.is_available():
            per_acc = (pred == true_label)[idx].type(torch.cuda.FloatTensor).mean().item()
        else:
            per_acc = (pred == true_label)[idx].type(torch.FloatTensor).mean().item()
        acc_dict[int(cls.cpu().data.item())] = per_acc
        cls_sample_count[f"number {int(cls.cpu().data.item())}"] = len(idx)
        cls_sample_count[f"percent {int(cls.cpu().data.item())}"] = round(len(idx) / len(indices), 5)
    return acc_dict, cls_sample_count


def acc_utils(da, sessions, base_class, way, session):
    acc_dict = {}
    acc_dict['all_acc'] = 0.0
    acc_dict['base_acc'] = 0.0
    acc_dict['novel_acc'] = 0.0
    acc_dict['former_acc'] = 0.0
    acc_dict['cur_acc'] = 0.0
    for i in range(sessions):
        acc_dict['sess{}_acc'.format(i)] = None
    # if session == 0:
    #     avger = Averager()
    #     for i in range(base_class):
    #         if i in da:
    #             avger.add(da[i])
    #     acc_dict['all_acc'] = avger.item()
    #     acc_dict['former_acc'] = None
    #     acc_dict['cur_acc'] = None
    #     acc_dict['base_acc'] = avger.item()
    #     acc_dict['novel_acc'] = None
    #     acc_dict['sess0_acc'] = avger.item()
    # else:
    for i in range(session + 1):
        if i == 0:
            sess_cls = range(base_class)
            acc_dict['sess{}_acc'.format(i)] =  get_aver(sess_cls, da)
        else:
            sess_cls = range(base_class + way * (i - 1), base_class + way * i)
            acc_dict['sess{}_acc'.format(i)] =  get_aver(sess_cls, da)

    all_cls = range(base_class + way * session)
    former_cls = range(base_class + way * (session - 1))
    cur_cls = range(base_class + way * (session - 1), base_class + way * session)
    base_cls = range(base_class)
    novel_cls = range(base_class, base_class + way * session)

    acc_dict['all_acc'] = get_aver(all_cls, da)
    acc_dict['former_acc'] = get_aver(former_cls, da)
    acc_dict['cur_acc'] = get_aver(cur_cls, da)
    acc_dict['base_acc'] = get_aver(base_cls, da)
    acc_dict['novel_acc'] = get_aver(novel_cls, da)
    return acc_dict

def cal_cpi(final_out_dict, alpha=0.5):
    acc_aver = {}
    msr_over = 1 - ((final_out_dict['base_acc'][0] - final_out_dict['base_acc'][-1]) / final_out_dict['base_acc'][0])
    acc_aver['acc_base_aver'] = sum(final_out_dict['base_acc']) / len(final_out_dict['base_acc'])
    acc_aver['acc_novel_aver'] = sum(final_out_dict['novel_acc'][1:]) / (len(final_out_dict['novel_acc']) - 1)
    acc_aver['acc_both_aver'] = sum(final_out_dict['both_acc']) / len(final_out_dict['both_acc'])
    acc_aver['acc_cur_aver'] = sum(final_out_dict['cur_acc'][1:]) / (len(final_out_dict['cur_acc']) - 1)
    cpi = alpha * msr_over + (1 - alpha) * acc_aver['acc_novel_aver']
    acc_df = pandas.Series(acc_aver)
    return cpi, msr_over, acc_df

def cal_auxIndex(final_out_dict, alpha=0.5):
    aux_index = {}
    acc_aver = {}
    acc_aver['acc_base_aver'] = 0.0
    acc_aver['acc_novel_aver'] = 0.0
    acc_aver['acc_both_aver'] = 0.0
    acc_aver['acc_cur_aver'] = 0.0
    ar_over = ((final_out_dict['base_acc'][0] - final_out_dict['base_acc'][-1]) / final_out_dict['base_acc'][0])
    msr_over = 1 - ar_over
    acc_aver['acc_base_aver'] = sum(final_out_dict['base_acc']) / len(final_out_dict['base_acc'])
    acc_aver['acc_both_aver'] = sum(final_out_dict['both_acc']) / len(final_out_dict['both_acc'])
    if len(final_out_dict['novel_acc']) - 1 > 0:
        acc_aver['acc_novel_aver'] = sum(final_out_dict['novel_acc'][1:]) / (len(final_out_dict['novel_acc']) - 1)
        acc_aver['acc_cur_aver'] = sum(final_out_dict['cur_acc'][1:]) / (len(final_out_dict['cur_acc']) - 1)
    else:
        acc_aver['acc_novel_aver'] = None
        acc_aver['acc_cur_aver'] = None
    cpi = alpha * msr_over + (1 - alpha) * acc_aver['acc_novel_aver']
    acc_df = pandas.Series(acc_aver)
    return cpi, msr_over, acc_df, ar_over
