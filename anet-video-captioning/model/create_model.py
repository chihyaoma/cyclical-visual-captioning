
import torch
import torch.nn as nn

from model.captioner import DecodeAndGroundCaptionerGVDROI
from model.backbone import RegionalFeatureExtractorGVD
from model.decoder_core import TopDownDecoderCore
from cycle_utils import resume_decoder_roiextractor


def build_model(opts, device):
    pretrained_decoder, embed, logit, roi_extractor = None, None, None, None

    if opts.resume_decoder_exp_name != '' and not opts.resume:
        pretrained_decoder = TopDownDecoderCore(opts)

        if opts.embedding_vocab_plus_1:
            embed = nn.Sequential(nn.Embedding(opts.vocab_size + 1, opts.input_encoding_size),
                                  nn.ReLU(),
                                  nn.Dropout(opts.drop_prob_lm))
        else:
            embed = nn.Sequential(nn.Embedding(opts.vocab_size, opts.input_encoding_size),
                                  nn.ReLU(),
                                  nn.Dropout(opts.drop_prob_lm))

        if opts.embedding_vocab_plus_1:
            logit = nn.Linear(opts.rnn_size, opts.vocab_size + 1)
        else:
            logit = nn.Linear(opts.rnn_size, opts.vocab_size)

        roi_extractor = RegionalFeatureExtractorGVD(opts)

        pretrained_decoder, embed, logit, roi_extractor = resume_decoder_roiextractor(
            opts, opts.resume_decoder_exp_name, pretrained_decoder, embed, logit, roi_extractor)

    model = DecodeAndGroundCaptionerGVDROI(opts, pretrained_decoder, embed, logit, roi_extractor).to(device)
    return model