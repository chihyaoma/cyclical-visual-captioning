import os
import sys
import shutil
import warnings
import pickle

from collections import OrderedDict

import torch

from tensorboardX import SummaryWriter


def set_tb_logger(log_dir, exp_name, resume):
    """ Set up tensorboard logger"""
    log_dir = log_dir + '/' + exp_name
    # remove previous log with the same name, if not resume
    if not resume and os.path.exists(log_dir):
        try:
            shutil.rmtree(log_dir)
        except:
            warnings.warn('Experiment existed in TensorBoard, but failed to remove')
    return SummaryWriter(log_dir=log_dir)


def is_code_development():
    return sys.platform == 'darwin'


def resume_decoder_roiextractor(opts, exp_name, decoder, embed, logit, roi_extractor):
    file_extention = 'model-best.pth'
    info_path = os.path.join(opts.checkpoint_dir + exp_name + '/', 'infos_' + opts.id + '-best.pkl')

    resume_file_name = opts.checkpoint_dir + exp_name + '/' + file_extention
    if os.path.isfile(resume_file_name):

        # open old infos and check if models are compatible
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
        opts.start_epoch = infos.get('epoch', 0)

        if not is_code_development():
            checkpoint = torch.load(resume_file_name)
        else:
            checkpoint = torch.load(resume_file_name, map_location=lambda storage, loc: storage)

        decoder_state_dict, embed_state_dict, logit_state_dict = OrderedDict(), OrderedDict(), OrderedDict()
        roi_extractor_state_dict = OrderedDict()

        # get the keys from the models
        decoder_keys = [key for key, _ in decoder.state_dict().items()]
        if opts.resume_embed:
            embed_keys = [key for key, _ in embed.state_dict().items()]
        if opts.resume_logit:
            logit_keys = [key for key, _ in logit.state_dict().items()]
        if opts.resume_roi_extractor:
            roi_extractor_keys = [key for key, _ in roi_extractor.state_dict().items()]

        # load the values into new state_dict
        for key, value in checkpoint.items():
            new_key = key.split('.', 1)[-1]
            if new_key in decoder_keys:
                decoder_state_dict[new_key] = value
            if opts.resume_embed and new_key in embed_keys:
                embed_state_dict[new_key] = value
            if opts.resume_logit and new_key in logit_keys:
                logit_state_dict[new_key] = value
            if opts.resume_roi_extractor and new_key in roi_extractor_keys:
                roi_extractor_state_dict[new_key] = value

        # load the state_dict into models
        assert set(decoder_state_dict.keys()) == set(decoder_keys)
        decoder.load_state_dict(decoder_state_dict)
        if opts.resume_embed:
            assert set(embed_state_dict.keys()) == set(embed_keys)
            print('==================================================')
            print('resuming embed weights ...')
            print('==================================================')
            embed.load_state_dict(embed_state_dict)
        if opts.resume_logit:
            assert set(logit_state_dict.keys()) == set(logit_keys)
            print('==================================================')
            print('resuming logit weights ...')
            print('==================================================')
            logit.load_state_dict(logit_state_dict)

        if opts.resume_roi_extractor:
            assert set(roi_extractor_state_dict.keys()) == set(roi_extractor_keys)
            print('==================================================')
            print('resuming ROI extractor weights ...')
            print('==================================================')
            roi_extractor.load_state_dict(roi_extractor_state_dict)
            print("=> loaded pre-trained decoder and ROI extractor from '{}' (epoch {})"
                  .format(resume_file_name, opts.start_epoch))
        else:
            print("=> loaded pre-trained decoder from '{}' (epoch {})"
                  .format(resume_file_name, opts.start_epoch))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(resume_file_name))

    return decoder, embed, logit, roi_extractor