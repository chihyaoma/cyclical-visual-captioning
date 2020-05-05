# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# revised by Chih-Yao Ma @ 20200501
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import random
import os
import pickle
import yaml
import h5py
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

import opts
import misc.utils as utils
from cycle_utils import is_code_development
from model.create_model import build_model
from trainer import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    opt = opts.parse_opt()
    if opt.path_opt is not None:
        with open(opt.path_opt, 'r') as handle:
            options_yaml = yaml.load(handle, Loader=yaml.FullLoader)
        utils.update_values(options_yaml, vars(opt))

    opt.checkpoint_path = opt.checkpoint_path + opt.exp_name
    print('=============')
    print(opt.exp_name)
    print('=============')

    opt.input_json = opt.data_path + opt.input_json
    opt.input_dic = opt.data_path + opt.input_dic
    opt.input_raw_cap = opt.data_path + opt.input_raw_cap
    opt.seg_feature_root = opt.data_path + opt.seg_feature_root
    opt.feature_root = opt.data_path + opt.feature_root
    opt.proposal_h5 = opt.data_path + opt.proposal_h5
    opt.densecap_references = [
        opt.data_path + reference for reference in opt.densecap_references]

    opt.test_mode = (opt.val_split == 'testing')
    if opt.enable_BUTD:
        assert opt.att_input_mode == 'region', 'region attention only under the BUTD mode'

    cudnn.benchmark = True

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.seed)

    if opt.dataset == 'anet':
        from misc.dataloader_anet import DataLoader
    else:
        raise Exception('only support anet!')

    if not os.path.exists(opt.checkpoint_path):
        os.makedirs(opt.checkpoint_path)

    # open the detection json file.
    print('DataLoader loading proposal file: ', opt.proposal_h5)
    h5_proposal_file = h5py.File(opt.proposal_h5, 'r', driver='core')
    num_proposals = h5_proposal_file['dets_num'][:]
    label_proposals = h5_proposal_file['dets_labels'][:]
    h5_proposal_file.close()

    # Data Loader
    dataset = DataLoader(opt, split=opt.train_split, seq_per_img=opt.seq_per_img,
                         num_proposals=num_proposals,
                         label_proposals=label_proposals)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                             shuffle=True,
                                             num_workers=opt.num_workers)

    dataset_val = DataLoader(opt, split=opt.val_split, seq_per_img=opt.seq_per_img,
                             num_proposals=num_proposals,
                             label_proposals=label_proposals)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batch_size,
                                                 shuffle=False, num_workers=opt.num_workers)
    # ======================================================================

    # Build the Model
    opt.vocab_size = dataset.vocab_size
    opt.detect_size = dataset.detect_size
    opt.seq_length = opt.seq_length
    opt.glove_w = torch.from_numpy(dataset.glove_w).float()
    opt.glove_vg_cls = torch.from_numpy(dataset.glove_vg_cls).float()
    opt.glove_clss = torch.from_numpy(dataset.glove_clss).float()

    opt.wtoi = dataset.wtoi
    opt.itow = dataset.itow
    opt.itod = dataset.itod
    opt.ltow = dataset.ltow
    opt.itoc = dataset.itoc
    opt.wtol = dataset.wtol
    opt.wtod = dataset.wtod
    opt.vg_cls = dataset.vg_cls

    if opt.att_model == 'cyclical':
        model = build_model(opt, device)
    else:
        raise ValueError('Unknown captioning model: {}'.format(opt.att_model))

    infos = {}
    histories = {}
    # if opt.start_from is not None:
    if opt.resume:
        if opt.load_best_score == 1:
            model_path = os.path.join(opt.checkpoint_path, 'model-best.pth')
            info_path = os.path.join(
                opt.checkpoint_path, 'infos_' + opt.id + '-best.pkl')
        else:
            model_path = os.path.join(opt.checkpoint_path, 'model.pth')
            info_path = os.path.join(
                opt.checkpoint_path, 'infos_' + opt.id + '.pkl')

        # open old infos and check if models are compatible
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
            saved_model_opt = infos['opt']

        # opt.learning_rate = saved_model_opt.learning_rate
        print('========================================')
        print('Loading the model %s...' % (model_path))
        if opt.inference_only:
            print('Running Inference only ...')
        print('========================================')
        # model.load_state_dict(torch.load(model_path))
        if not is_code_development():
            model.load_state_dict(torch.load(model_path))
        else:
            model.load_state_dict(torch.load(
                model_path, map_location=lambda storage, loc: storage))

        if os.path.isfile(os.path.join(opt.checkpoint_path, 'histories_' + opt.id + '.pkl')):
            with open(os.path.join(opt.checkpoint_path, 'histories_' + opt.id + '.pkl'), 'rb') as f:
                histories = pickle.load(f)

    best_val_score = infos.get('best_val_score', None)
    iteration = infos.get('iter', 0)

    if opt.resume_decoder_exp_name != '' and not opt.resume:
        start_epoch = opt.start_epoch
    else:
        start_epoch = infos.get('epoch', 0)

    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})

    model = nn.DataParallel(model).to(device)

    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if ('ctx2pool_grd' in key) or ('vis_embed' in key):
                print('Finetune param: {}'.format(key))
                params += [{'params': [value], 'lr': opt.learning_rate * 0.1,  # finetune the fc7 layer
                            'weight_decay': opt.weight_decay, 'betas': (opt.optim_alpha, opt.optim_beta)}]
            else:
                params += [{'params': [value], 'lr': opt.learning_rate,
                            'weight_decay': opt.weight_decay, 'betas': (opt.optim_alpha, opt.optim_beta)}]

    print("Use %s as optmization method" % (opt.optim))
    optimizer = None
    if opt.optim == 'sgd':
        optimizer = optim.SGD(params, lr=opt.learning_rate, momentum=0.9)
    elif opt.optim == 'adam':
        optimizer = optim.Adam(params)
    elif opt.optim == 'adamax':
        optimizer = optim.Adamax(params)
    else:
        raise ValueError('Unknown optimizer: {}'.format(opt.optim))

    # set up tensorboard logger
    tb_logger = utils.set_tb_logger(
        opt.tb_log_dir, opt.exp_name, opt.resume) if not opt.inference_only else None

    # set up trainer
    trainer = Trainer(opt, dataset, model, optimizer, dataloader, dataloader_val)

    # set up LR scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 'max', patience=opt.patience, min_lr=opt.min_lr)

    best_score = {
        "Bleu_1": 0.0,
        "Bleu_2": 0.0,
        "Bleu_3": 0.0,
        "Bleu_4": 0.0,
        "METEOR": 0.0,
        "ROUGE_L": 0.0,
        "CIDEr": 0.0,
        "SPICE": 0.0
    }

    for epoch in range(start_epoch, opt.max_epochs):
        if not opt.inference_only:
            trainer.train(epoch, tb_logger=tb_logger)

        if epoch % opt.val_every_epoch == 0:
            with torch.no_grad():
                lang_stats = trainer.eval(epoch, tb_logger=tb_logger)

            if opt.inference_only:
                break

            # update learning rate by monitoring CIDEr score
            scheduler.step(lang_stats['CIDEr'], epoch)

            # Save model if is improving on validation result
            current_score = lang_stats['CIDEr']

            best_flag = False
            if best_val_score is None or current_score > best_val_score:
                best_val_score = current_score
                best_flag = True
            checkpoint_path = os.path.join(opt.checkpoint_path, 'model.pth')
            if opt.mGPUs:
                torch.save(model.module.state_dict(), checkpoint_path)
            else:
                torch.save(model.state_dict(), checkpoint_path)
            print("model saved to {}".format(checkpoint_path))

            # Dump miscalleous informations
            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['best_val_score'] = best_val_score
            infos['opt'] = opt
            infos['vocab'] = dataset.itow

            histories['val_result_history'] = val_result_history
            histories['loss_history'] = loss_history
            histories['lr_history'] = lr_history
            histories['ss_prob_history'] = ss_prob_history
            with open(os.path.join(opt.checkpoint_path, 'infos_' + opt.id + '.pkl'), 'wb') as f:
                pickle.dump(infos, f)
            with open(os.path.join(opt.checkpoint_path, 'histories_' + opt.id + '.pkl'), 'wb') as f:
                pickle.dump(histories, f)

            if best_flag:
                checkpoint_path = os.path.join(
                    opt.checkpoint_path, 'model-best.pth')
                if opt.mGPUs:
                    torch.save(model.module.state_dict(), checkpoint_path)
                else:
                    torch.save(model.state_dict(), checkpoint_path)

                print("model saved to {} with best cider score {:.3f}".format(
                    checkpoint_path, best_val_score))
                with open(os.path.join(opt.checkpoint_path, 'infos_' + opt.id + '-best.pkl'), 'wb') as f:
                    pickle.dump(infos, f)

                # update best scores
                for metric, _ in best_score.items():
                    best_score[metric] = lang_stats[metric]

            print("===================================")
            print("--> Highest scores on {} set at epoch {}".format(opt.val_split, epoch))
            for metric, score in sorted(best_score.items()):
                print('{}: {:.4f}'.format(metric, score))


if __name__ == '__main__':
    main()
