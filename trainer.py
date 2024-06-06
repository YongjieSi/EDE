import json
import random
import sys
import logging
import copy
import time
import torch
from utils import factory
from utils.data_manager_for_fscil import DataManager
from utils.toolkit import count_parameters
import os
import pandas as pd
import numpy as np
import pandas


def train(args):
    seed_list = copy.deepcopy(args['seed'])
    device = copy.deepcopy(args['device'])
    final_out_dict = []
    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        out_dic= _train(args)
        final_out_dict.append(out_dic)
    final_result(args,final_out_dict)

def _train(args):
    try:
        os.mkdir("/data/syj/EDE/logs/{}".format(args['model_name']))
    except:
        pass
    logfilename = '/data/syj/EDE/logs/{}/{}_seed{}_{}_{}_{}_{}initcls_{}way_{}shot_bepochs{}_iepochs{}_blrate{}_ilrate{}_R{}'.format(args['model_name'], args['prefix'], args['seed'], args['convnet_type'], args['convnet_type'],
                                                args['dataset'], args['init_cls'], args['increment'], args['shot'], args['init_epoch'], args['new_epochs'], args['init_lrate'], args['lrate'],args['ffn_num'])
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=logfilename + '.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    data_manager = DataManager(args['dataset'], args['shuffle'], args['seed'], args['init_cls'], args['increment'])

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    model = factory.get_model(args['model_name'], args)

    cnn_curve, nme_curve = {'top1': [], 'top5': []}, {'top1': [], 'top5': []}
    sess_acc_dict = {}
    for task in range(args['nb_tasks']):
        logging.info('All params: {}'.format(count_parameters(model._network)))
        logging.info('Trainable params: {}'.format(count_parameters(model._network, True)))
        nt_strat_time = time.time()
        model.incremental_train(data_manager)
        nt_end_time = time.time()
        spend_time = nt_end_time - nt_strat_time
        logging.info('Task{} spend {} seconds to train'.format(task, spend_time))
        cnn_accy, nme_accy, acc_dict, cls_sample_count = model.eval_task()
        sess_acc_dict[f'sess {task}'] = acc_dict
        logging.info(f"acc_dict:{acc_dict}")
        model.after_task()

        if nme_accy is not None:
            logging.info('CNN: {}'.format(cnn_accy['grouped']))
            logging.info('NME: {}'.format(nme_accy['grouped']))

            cnn_curve['top1'].append(cnn_accy['top1'])
            cnn_curve['top5'].append(cnn_accy['top5'])

            nme_curve['top1'].append(nme_accy['top1'])
            nme_curve['top5'].append(nme_accy['top5'])

            logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
            logging.info('CNN top5 curve: {}'.format(cnn_curve['top5']))
            logging.info('NME top1 curve: {}'.format(nme_curve['top1']))
            logging.info('NME top5 curve: {}\n'.format(nme_curve['top5']))
        else:
            logging.info('No NME accuracy.')
            logging.info('CNN: {}'.format(cnn_accy['grouped']))

            cnn_curve['top1'].append(cnn_accy['top1'])
            cnn_curve['top5'].append(cnn_accy['top5'])

            logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
            logging.info('CNN top5 curve: {}\n'.format(cnn_curve['top5']))
        
    out_dict = {}
    out_dict['cur_acc'] = []
    out_dict['former_acc'] = []
    out_dict['both_acc'] = []
    for k, v in sess_acc_dict.items():
        out_dict['cur_acc'].append(v['cur_acc'])
        out_dict['former_acc'].append(v['former_acc'])
        out_dict['both_acc'].append(v['all_acc'])
    out_df = pandas.DataFrame(out_dict)
    out_df = out_df.T
    pandas.set_option('display.max_rows', None)
    pandas.set_option('display.max_columns', None) 
    pandas.set_option('display.width', None)
    pandas.set_option('display.max_colwidth', None)
    logging.info(f"final output:{out_dict}")
    logging.info(f"\n****************************************Pretty Output********************************************\
                \n{out_df}\
                \n***********************************************************************************************")
    return out_dict

def final_result(args,dict_list):
    merged_dict = {}
    h=len(dict_list[0]["cur_acc"])
    w = len(dict_list)
    for d in dict_list:
        for k, v in d.items():
            if k not in merged_dict:
                merged_dict[k] = v
            else:
                merged_dict[k].extend(v)
    matrix_dict = {k: torch.tensor(v).view(w, h) for k, v in merged_dict.items()}
    result_dict = {}
    for k, v in matrix_dict.items():
        v = np.array(v, dtype=np.float32)
        column_means = np.round(np.mean(v, axis=0),4)
        column_stddevs = np.round(np.std(v, axis=0),4)
        formatted_values = [f"{mean*100:.2f}±{std*100:.2f}" for mean, std in zip(column_means, column_stddevs)]
        if k not in result_dict:
            result_dict[k] = formatted_values
        else:
            result_dict[k].extend(formatted_values)

    result_dict = pd.DataFrame.from_dict(result_dict)
    result_dict = result_dict.T
    save_dir ='/logs/{}/{}_seed{}_{}_{}_{}_{}initcls_{}way_{}shot_bepochs{}_iepochs{}_blrate{}_ilrate{}.output.xlsx'.format(args['model_name'], args['prefix'], args['seed'], args['convnet_type'], args['convnet_type'],
                                                args['dataset'], args['init_cls'], args['increment'], args['shot'], args['init_epoch'], args['new_epochs'], args['init_lrate'], args['lrate'])
    # df.to_excel(os.path.join("/data/syj/PyCIL/F_logs/{}/",str(args['model_name']) + '_'+ str(args['dataset'])+  '_'+ str(args['convnet_type'])+'init_cls_'+ str(args['init_cls'])+'init_epoch_'+ str(args['init_epoch'])+'init_lrate_'+ str(args['init_lrate'])+'incre_lrate_'+ str(args['lrate'])+'.output.xlsx'), index=False)
    result_dict.to_excel(save_dir,index=False)
    logging.info(f"\n****************************************Pretty Output********************************************\
                        \n{result_dict}\
                        \n***********************************************************************************************")
    output = f"\n****************************************Pretty Output********************************************\
                    \n{result_dict}\
                    \n***********************************************************************************************"
    print(output)

def _set_device(args):
    device_type = args['device']
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(device))

        gpus.append(device)

    args['device'] = gpus


def _set_random(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info('{}: {}'.format(key, value))
