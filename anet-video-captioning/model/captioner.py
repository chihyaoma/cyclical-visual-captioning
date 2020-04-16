
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbone import RegionalFeatureExtractorGVD
from model.decoder_core import TopDownDecoderCore, AttenedDecoderCore
from model.localizer_core import LocalizerNoLSTMCore
import misc.utils as utils

# lemmatizer = WordNetLemmatizer()


class DecodeAndGroundCaptionerGVDROI(nn.Module):
    """
    Localize ROIs from given caption and generate caption from localized ROIs
    """

    def __init__(self, opts, pretrained_decoder=None, embed=None, logit=None, roi_extractor=None):
        super(DecodeAndGroundCaptionerGVDROI, self).__init__()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.opts = opts
        self.vocab_size = opts.vocab_size
        self.ix_to_word = opts.itow  # ix_to_word is the same for train and val

        if roi_extractor is not None:
            self.roi_feat_extractor = roi_extractor
        else:
            self.roi_feat_extractor = RegionalFeatureExtractorGVD(opts)

        self.seq_length = opts.seq_length
        self.seq_per_img = opts.seq_per_img

        self.decoder_num_layers = 2  # Top-Down model has 2 layer of LSTMs
        self.localizer_num_layers = 1  # 1 layer of LSTM for localizer
        self.rnn_size = opts.rnn_size

        self.ss_prob = 0.0  # Schedule sampling probability

        self.iou_threshold = 0.5

        # ==================================================
        if pretrained_decoder is None:
            self.decoder_core = TopDownDecoderCore(opts)
        else:
            self.decoder_core = pretrained_decoder

        if embed is None:
            if opts.embedding_vocab_plus_1:
                self.embed = nn.Sequential(
                    nn.Embedding(opts.vocab_size + 1,
                                 opts.input_encoding_size),
                    # we probably can't do "padding_idx=0" since the <BOS> is also encoded as "0"
                    nn.ReLU(),
                    nn.Dropout(opts.drop_prob_lm)
                )
            else:
                self.embed = nn.Sequential(
                    nn.Embedding(opts.vocab_size, opts.input_encoding_size),
                    # we probably can't do "padding_idx=0" since the <BOS> is also encoded as "0"
                    nn.ReLU(),
                    nn.Dropout(opts.drop_prob_lm)
                )
        else:
            self.embed = embed

        if logit is None:
            if opts.embedding_vocab_plus_1:
                self.logit = nn.Linear(opts.rnn_size, opts.vocab_size + 1)
            else:
                self.logit = nn.Linear(opts.rnn_size, opts.vocab_size)
        else:
            self.logit = logit
        # ==================================================

        # set up the localizer
        self.localizer_num_layers = 1  # 1 layer of LSTM for localizer
        self.localizer_core = LocalizerNoLSTMCore(opts)

        # set up the reconstructor with the shared AttnLSTM and LangLSTM
        self.attended_roi_decoder_core = AttenedDecoderCore(
            opts, self.decoder_core.att_lstm, self.decoder_core.lang_lstm)

        self.critLM = utils.LMCriterion(opts)
        self.unk_idx = int(opts.wtoi['UNK'])

        self.xe_criterion = utils.LanguageCriterion()

        # self.i_to_visually_groundable = opts.i_to_visually_groundable

    def init_hidden(self, batch_size, num_layers):
        weight = next(self.parameters()).data
        return (weight.new(num_layers, batch_size, self.rnn_size).zero_(),
                weight.new(num_layers, batch_size, self.rnn_size).zero_())

    def obj_grounding(self, att_weights, seq, targets):
        """

        :param att_weights: batch_size * seq_per_img, seq_length, num_padded_proposal
        :param seq: batch_size * seq_per_img, seq_length
        :param targets: batch_size * seq_per_img, seq_length, num_padded_proposal
        :return:
        """

        # the matching criterion is overlap>0.5
        # if multiple max exist, pick the first one, i.e., the proposal with the max confident score
        _, max_weights_ind = torch.max(att_weights, 2)
        max_weights_ind = max_weights_ind.view(-1, 1)
        vis_mask = (seq > self.vocab_size)
        vis_mask = vis_mask.view(-1, 1)
        seq = seq.view(-1, 1)
        targets = targets.view(-1, targets.size(2))

        # matches,_ = torch.max(targets, 1, keepdim=True) # upper bound
        matches = torch.gather(targets, 1, max_weights_ind)
        masked_matches = torch.masked_select(matches, vis_mask)
        masked_seq = torch.masked_select(seq, vis_mask)

        # print('Box accuracy for the current batch: {}'.format(torch.mean(masked_matches)))

        hm_lst = torch.stack((masked_seq.float(), masked_matches), dim=1).data

        return hm_lst

    def _grounder(self, xt, att_feats, mask, bias=None, min_value=-1e8):
        """
        Messy code from GVD
        """
        # xt - B, seq_cnt, enc_size
        # att_feats - B, rois_num, enc_size
        # mask - B, rois_num
        #
        # dot - B, seq_cnt, rois_num

        B, S, _ = xt.size()
        _, R, _ = att_feats.size()

        if hasattr(self, 'alpha_net'):
            # Additive attention for grounding
            if self.alpha_net.weight.size(1) == self.att_hid_size:
                dot = xt.unsqueeze(2) + att_feats.unsqueeze(1)
            else:
                dot = torch.cat((xt.unsqueeze(2).expand(B, S, R, self.att_hid_size),
                                 att_feats.unsqueeze(1).expand(B, S, R, self.att_hid_size)), 3)
            dot = F.tanh(dot)
            dot = self.alpha_net(dot).squeeze(-1)
        else:
            # Dot-product attention for grounding
            assert (xt.size(-1) == att_feats.size(-1))
            dot = torch.matmul(xt, att_feats.permute(
                0, 2, 1).contiguous())  # B, seq_cnt, rois_num

        if bias is not None:
            assert (bias.numel() == dot.numel())
            dot += bias

        if mask.dim() == 2:
            expanded_mask = mask.unsqueeze(1).expand_as(dot)
        elif mask.dim() == 3:  # if expanded already
            expanded_mask = mask
        else:
            raise NotImplementedError

        dot.masked_fill_(expanded_mask, min_value)

        return dot

    def forward(self, segs_feat, input_seq, gt_caption, num, proposals, gt_boxes, mask_boxes, region_feats,
                frm_mask,
                sample_idx, pnt_mask, lang_eval=False, teacher_forcing=False):
        """

        :param segs_feat:
        :param proposals: (batch_size, max num of proposals allowed (200), 6), 6-dim: (4 points for the box, proposal index to be used for self.itod to get detection, proposal score)
        :param input_seq: (batch_size, # of captions for this img, max caption length, 4), 4-dim: (category class, binary class, fine-grain class, from self.wtoi)
        :param num: (batch_size, 3), 3-dim: (# of captions for this img, # of proposals, # of GT bboxs)
        :param mask_boxes: (batch_size, # of captions for this img, max num of GT bbox allowed (100), max caption length)
        :param gt_boxes: (batch_size, max num of GT bbox allowed (100), 5), 5-dim: (4 points for the box, bbox index)
        :return:
        """
        if lang_eval == False or teacher_forcing:
            # During training, we don't need language metric evaluation
            return self._forward_3_loops(segs_feat, input_seq, proposals, gt_caption, num, mask_boxes, gt_boxes,
                                         region_feats, frm_mask, sample_idx, pnt_mask)
        else:
            return self._sample(segs_feat, input_seq, proposals, gt_caption, num, mask_boxes, gt_boxes,
                                region_feats, frm_mask, sample_idx, pnt_mask)

    def _forward_3_loops(self, segs_feat, input_seq, proposals, gt_caption, num, mask_boxes, gt_boxes,
                         region_feats, frm_mask, sample_idx, pnt_mask):
        """

        :param segs_feat: 
        :param proposals: (batch_size, max num of proposals allowed (200), 6), 6-dim: (4 points for the box, proposal index to be used for self.itod to get detection, proposal score)
        :param seq: (batch_size, # of captions for this img, max caption length, 4), 4-dim: (category class, binary class, fine-grain class, from self.wtoi)
        :param num: (batch_size, 3), 3-dim: (# of captions for this img, # of proposals, # of GT bboxs)
        :param mask_boxes: (batch_size, # of captions for this img, max num of GT bbox allowed (100), max caption length)
        :param gt_boxes: (batch_size, max num of GT bbox allowed (100), 5), 5-dim: (4 points for the box, bbox index)
        :return:
        """
        batch_size = segs_feat.size(0)
        num_rois = proposals.size(1)
        gt_caption = gt_caption[:, :self.seq_per_img, :].clone(
        ).view(-1, gt_caption.size(2))  # choose the first seq_per_img
        gt_caption = torch.cat(
            (gt_caption.new(gt_caption.size(0), 1).fill_(0), gt_caption), 1)

        # B*self.seq_per_img, self.seq_length+1, 5
        input_seq = input_seq.view(-1, input_seq.size(2), input_seq.size(3))
        input_seq_update = input_seq.data.clone()

        gt_caption_batch_size = gt_caption.size(0)

        decoder_state = self.init_hidden(
            gt_caption_batch_size, self.decoder_num_layers)

        # calculate the overlaps between the rois and gt_bbox.
        # overlaps, overlaps_no_replicate = utils.bbox_overlaps(proposals.data, gt_boxes.data)
        # calculate the overlaps between the rois/rois and rois/gt_bbox.
        # apply both frame mask and proposal mask
        overlaps = utils.bbox_overlaps(
            proposals.data, gt_boxes.data, (frm_mask | pnt_mask[:, 1:].unsqueeze(-1)).data)

        fc_feats, conv_feats, p_conv_feats, pool_feats, \
            p_pool_feats, g_pool_feats, pnt_mask, overlaps_expanded, cls_pred, cls_loss = self.roi_feat_extractor(
                segs_feat, proposals, num, mask_boxes, region_feats, gt_boxes, overlaps, sample_idx)

        # ====================================================================================
        # regular decoder
        # ====================================================================================
        lang_outputs = []
        decoder_roi_attn, decoder_masked_attn, roi_labels = [], [], []
        frm_mask_output = []
        # unfold the GT caption and localize the ROIs word by word
        for word_idx in range(self.seq_length):
            word = gt_caption[:, word_idx].clone()
            embedded_word = self.embed(word)

            roi_label = utils.bbox_target(mask_boxes[:, :, :, word_idx + 1], overlaps, input_seq[:, word_idx + 1],
                                          input_seq_update[:, word_idx + 1], self.vocab_size)  # roi_label for the target seq
            roi_labels.append(roi_label.view(gt_caption_batch_size, -1))

            # use frame mask during training
            box_mask = mask_boxes[:, 0, :, word_idx + 1].contiguous().unsqueeze(1).expand((
                batch_size, num_rois, mask_boxes.size(2)))
            # frm_mask_on_prop = (
            #     torch.sum((1 - (box_mask | frm_mask)), dim=2) <= 0)
            frm_mask_on_prop = (
                torch.sum((~(box_mask | frm_mask)), dim=2) <= 0)

            frm_mask_on_prop = torch.cat((frm_mask_on_prop.new(batch_size, 1).fill_(
                0.), frm_mask_on_prop), dim=1) | pnt_mask.bool()
            frm_mask_output.append(frm_mask_on_prop)

            output, decoder_state, roi_attn, frame_masked_attn, weighted_pool_feat = self.decoder_core(
                embedded_word, fc_feats, conv_feats, p_conv_feats, pool_feats,
                p_pool_feats, pnt_mask[:, 1:], decoder_state, proposal_frame_mask=frm_mask_on_prop[:, 1:], with_sentinel=False)

            output = F.log_softmax(self.logit(output), dim=1)
            decoder_roi_attn.append(roi_attn)
            decoder_masked_attn.append(frame_masked_attn)

            lang_outputs.append(output)

        # (batch_size, seq_length - 1, max # of ROIs)
        att2_weights = torch.stack(decoder_masked_attn, dim=1)

        # decoder output for captioning task
        lang_outputs = torch.cat([_.unsqueeze(1) for _ in lang_outputs], dim=1)
        roi_labels = torch.cat([_.unsqueeze(1) for _ in roi_labels], dim=1)
        frm_mask_output = torch.cat([_.unsqueeze(1)
                                     for _ in frm_mask_output], 1)

        # object grounding
        xt_clamp = torch.clamp(
            input_seq[:, 1:self.seq_length + 1, 0].clone() - self.vocab_size, min=0)
        xt_all = self.roi_feat_extractor.vis_embed(xt_clamp)

        if hasattr(self.roi_feat_extractor, 'vis_classifiers_bias'):
            bias = self.roi_feat_extractor.vis_classifiers_bias[xt_clamp].type(xt_all.type()) \
                .unsqueeze(2).expand(gt_caption_batch_size, self.seq_length, num_rois)
        else:
            bias = 0

        # att2_weights/ground_weights with both proposal mask and frame mask
        ground_weights = self._grounder(
            xt_all, g_pool_feats, frm_mask_output[:, :, 1:], bias + att2_weights)

        # if we are only training for the decoder, directly return language output
        if self.opts.train_decoder_only:
            lang_outputs = lang_outputs.view(-1, lang_outputs.size(2))

            lm_loss, att2_loss, ground_loss = self.critLM(lang_outputs, att2_weights, ground_weights,
                                                          gt_caption[:,
                                                                     1:self.seq_length + 1].clone(),
                                                          roi_labels[:, :self.seq_length, :].clone(
                                                          ),
                                                          input_seq[:, 1:self.seq_length + 1, 0].clone())

            return lm_loss.unsqueeze(0), att2_loss.unsqueeze(0), ground_loss.unsqueeze(0), cls_loss.unsqueeze(0)

        # ====================================================================================
        # greedy select words with highest probability and use them for localization
        # ====================================================================================
        # this will turn the output_seq to be the leaf on the computational graph
        _, output_seq = lang_outputs.max(2)
        localizer_state = self.init_hidden(
            gt_caption_batch_size, self.localizer_num_layers)

        localized_weighted_feats = []
        localized_conv_feats = []
        localized_roi_probs = []
        for word_idx in range(self.seq_length):
            word = output_seq[:, word_idx].clone()
            embedded_word = self.embed(word)

            localized_weighted_feat, localized_conv_feat, localized_roi_prob, localizer_state = self.localizer_core(
                embedded_word, fc_feats, conv_feats, p_conv_feats, pool_feats, p_pool_feats,
                pnt_mask[:, 1:], localizer_state, None,
                proposal_frame_mask=frm_mask_output[:, word_idx, 1:], with_sentinel=False)

            # if self.opts.localizer_only_groundable:
            #     visually_groundable_mask = np.in1d(np.array(word.cpu()), self.i_to_visually_groundable).astype('float')
            #     visually_groundable_mask = torch.from_numpy(visually_groundable_mask).float().to(localized_weighted_feat.device)
            #     localized_weighted_feat = visually_groundable_mask.unsqueeze(1) * localized_weighted_feat
            #     localized_conv_feat = visually_groundable_mask.unsqueeze(1) * localized_conv_feat
            #     localized_roi_prob = visually_groundable_mask.unsqueeze(1) * localized_roi_prob

            localized_weighted_feats.append(localized_weighted_feat)
            localized_conv_feats.append(localized_conv_feat)
            localized_roi_probs.append(localized_roi_prob)

        # localized_roi_probs = torch.stack(localized_roi_probs, dim=1)  # (batch_size, seq_length - 1, max # of ROIs)

        # ====================================================================================
        # decoder use the localized ROIs to generate caption again
        # ====================================================================================
        consistent_outputs = []
        consistent_decoder_state = self.init_hidden(
            gt_caption_batch_size, self.decoder_num_layers)
        for word_idx in range(self.seq_length):
            word = gt_caption[:, word_idx].clone()
            embedded_word = self.embed(word)

            localized_weighted_feat = localized_weighted_feats[word_idx]
            localized_conv_feat = localized_conv_feats[word_idx]

            output, consistent_decoder_state = self.attended_roi_decoder_core(embedded_word, fc_feats,
                                                                              localized_weighted_feat,
                                                                              localized_conv_feat,
                                                                              consistent_decoder_state,
                                                                              with_sentinel=False)

            output = F.log_softmax(self.logit(output), dim=1)
            consistent_outputs.append(output)

        consistent_outputs = torch.cat(
            [_.unsqueeze(1) for _ in consistent_outputs], dim=1)

        lang_outputs = lang_outputs.view(-1, lang_outputs.size(2))
        lm_loss, att2_loss, ground_loss = self.critLM(lang_outputs, att2_weights, ground_weights,
                                                      gt_caption[:,
                                                                 1:self.seq_length + 1].clone(),
                                                      roi_labels[:, :self.seq_length, :].clone(
                                                      ),
                                                      input_seq[:, 1:self.seq_length + 1, 0].clone())

        # compute another loss for the reconstructor
        consistent_outputs = consistent_outputs.view(
            -1, consistent_outputs.size(2))
        lm_recon_loss = self.xe_criterion(
            consistent_outputs, gt_caption[:, 1:self.seq_length + 1].clone())

        return lm_loss.unsqueeze(0), att2_loss.unsqueeze(0), ground_loss.unsqueeze(0), \
            cls_loss.unsqueeze(0), lm_recon_loss.unsqueeze(0)

    def _sample(self, segs_feat, seq, proposals, gt_caption, num, mask_boxes, gt_boxes, region_feats, frm_mask, sample_idx, pnt_mask):
        """

        :param segs_feat:
        :param proposals: (batch_size, max num of proposals allowed (200), 6), 6-dim: (4 points for the box, proposal index to be used for self.itod to get detection, proposal score)
        :param seq: (batch_size, # of captions for this img, max caption length, 4), 4-dim: (category class, binary class, fine-grain class, from self.wtoi)
        :param num: (batch_size, 3), 3-dim: (# of captions for this img, # of proposals, # of GT bboxs)
        :param mask_boxes: (batch_size, # of captions for this img, max num of GT bbox allowed (100), max caption length)
        :return:
        """
        batch_size = segs_feat.size(0)
        decoder_state = self.init_hidden(batch_size, self.decoder_num_layers)

        # calculate the overlaps between the rois/rois and rois/gt_bbox.
        # apply both frame mask and proposal mask
        overlaps = utils.bbox_overlaps(
            proposals.data, gt_boxes.data, (frm_mask | pnt_mask[:, 1:].unsqueeze(-1)).data)

        fc_feats, conv_feats, p_conv_feats, pool_feats, p_pool_feats, g_pool_feats, \
            pnt_mask, overlaps_expanded, cls_pred, cls_loss = self.roi_feat_extractor(
                segs_feat, proposals, num, mask_boxes, region_feats, gt_boxes, overlaps, sample_idx)

        seq_output = []
        seq_output_logprobs = []
        att2_weights = []

        for word_idx in range(self.seq_length + 1):
            if word_idx == 0:  # input <bos>
                # word = fc_feats.data.new(batch_size).long().zero_()
                word = fc_feats.new_zeros(batch_size).long()
            else:
                sampleLogprobs_tmp, word_tmp = torch.topk(
                    logprobs.data, 2, dim=1)
                unk_mask = (word_tmp[:, 0] != self.unk_idx)  # mask on non-unk
                sampleLogprobs = unk_mask.float() * sampleLogprobs_tmp[:, 0] + (
                    1 - unk_mask.float()) * sampleLogprobs_tmp[:, 1]
                word = unk_mask.long() * \
                    word_tmp[:, 0] + (1 - unk_mask.long()) * word_tmp[:, 1]
                word = word.view(-1).long()

            embedded_word = self.embed(word)

            if word_idx >= 1:
                # seq_output[t] the input of t+2 time step
                seq_output.append(word)
                seq_output_logprobs.append(sampleLogprobs.view(-1))

            if word_idx < self.seq_length:
                output, decoder_state, roi_attn, _, weighted_pool_feat = self.decoder_core(
                    embedded_word, fc_feats, conv_feats, p_conv_feats, pool_feats,
                    p_pool_feats, pnt_mask[:, 1:], decoder_state,
                    with_sentinel=False)

                logprobs = F.log_softmax(self.logit(output), dim=1)
                att2_weights.append(roi_attn)

        att2_weights = torch.cat([_.unsqueeze(1) for _ in att2_weights], 1)

        return torch.cat([_.unsqueeze(1) for _ in seq_output], 1), \
            att2_weights, None
