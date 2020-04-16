import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

import misc.utils as utils
from model.modules import proj_masking
from misc.transformer import Transformer


class RegionalFeatureExtractorGVD(nn.Module):
    """
    Localize ROIs from given caption and generate caption from localized ROIs
    """

    def __init__(self, opts):
        super(RegionalFeatureExtractorGVD, self).__init__()
        self.opts = opts
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.test_mode = opts.test_mode
        self.enable_BUTD = opts.enable_BUTD
        self.att_input_mode = opts.att_input_mode
        self.num_sampled_frm = opts.num_sampled_frm

        self.rnn_size = opts.rnn_size
        self.seq_per_img = opts.seq_per_img
        self.finetune_cnn = opts.finetune_cnn
        self.seg_info_size = 50

        self.att_feat_size = opts.att_feat_size
        self.fc_feat_size = opts.fc_feat_size + self.seg_info_size

        # self.pool_feat_size = self.att_feat_size + 300 * 2
        self.detect_size = opts.detect_size  # number of fine-grained detection classes
        self.pool_feat_size = self.att_feat_size + 300 + self.detect_size + 1

        # if opts.transfer_mode in ('none', 'cls'):
        #     self.vis_encoding_size = 2048
        # elif opts.transfer_mode == 'both':
        #     self.vis_encoding_size = 2348
        # elif opts.transfer_mode == 'glove':
        #     self.vis_encoding_size = 300
        # else:
        #     raise NotImplementedError
        self.vis_encoding_size = opts.vis_encoding_size

        self.t_attn_size = opts.t_attn_size
        self.min_value = -1e8

        self.loc_fc = nn.Sequential(nn.Linear(5, 300),
                                    nn.ReLU(),
                                    nn.Dropout(opts.drop_prob_lm))
        self.det_fc = nn.Sequential(nn.Embedding(opts.detect_size + 1, 300),
                                    nn.ReLU(),
                                    nn.Dropout(opts.drop_prob_lm))
        # initialize the glove weight for the labels.
        self.det_fc[0].weight.data.copy_(opts.glove_clss)
        for p in self.det_fc[0].parameters():
            p.requires_grad = False

        self.vis_embed = nn.Sequential(nn.Embedding(opts.detect_size + 1, self.vis_encoding_size),  # det is 1-indexed
                                       nn.ReLU(),
                                       nn.Dropout(opts.drop_prob_lm)
                                       )

        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, opts.rnn_size),
                                      nn.ReLU(),
                                      nn.Dropout(opts.drop_prob_lm))

        self.seg_info_embed = nn.Sequential(nn.Linear(4, self.seg_info_size),
                                            nn.ReLU(),
                                            nn.Dropout(opts.drop_prob_lm, inplace=True))

        self.att_embed = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2048, opts.rnn_size // 2),  # for rgb feature
                nn.ReLU(),
                nn.Dropout(opts.drop_prob_lm, inplace=True)
            ),
            nn.Sequential(
                nn.Linear(1024, opts.rnn_size // 2),  # for motion feature
                nn.ReLU(),
                nn.Dropout(opts.drop_prob_lm, inplace=True)
            )
        ])

        self.att_embed_aux = nn.Sequential(nn.BatchNorm1d(opts.rnn_size),
                                           nn.ReLU())

        self.pool_embed = nn.Sequential(nn.Linear(self.pool_feat_size, opts.rnn_size),
                                        nn.ReLU(),
                                        nn.Dropout(opts.second_drop_prob))

        self.ctx2att_fc = nn.Linear(opts.rnn_size, opts.att_hid_size)
        self.ctx2pool_fc = nn.Linear(opts.rnn_size, opts.att_hid_size)

        if opts.att_model == 'transformer':
            raise NotImplementedError()

        # if opts.obj_interact:
        #     n_layers = 2
        #     n_heads = 6
        #     attn_drop = 0.2
        #     self.obj_interact = Transformer(self.rnn_size, 0, 0,
        #                                     d_hidden=int(self.rnn_size / 2),
        #                                     n_layers=n_layers,
        #                                     n_heads=n_heads,
        #                                     drop_ratio=attn_drop,
        #                                     pe=False)

        if opts.t_attn_mode == 'bilstm':  # frame-wise feature encoding
            n_layers = 2
            attn_drop = 0.2
            self.context_enc = nn.LSTM(self.rnn_size, self.rnn_size // 2, n_layers, dropout=attn_drop,
                                       bidirectional=True, batch_first=True)
        elif opts.t_attn_mode == 'bigru':
            n_layers = 2
            attn_drop = 0.2
            self.context_enc = nn.GRU(self.rnn_size, self.rnn_size // 2, n_layers, dropout=attn_drop,
                                      bidirectional=True, batch_first=True)
        else:
            raise NotImplementedError

        self.ctx2pool_grd = nn.Sequential(
            nn.Linear(opts.att_feat_size, self.vis_encoding_size),  # fc7 layer
            nn.ReLU(),
            nn.Dropout(opts.drop_prob_lm)
        )  # fc7 layer in detectron

        with open('data/detectron_weights/fc7_w.pkl', 'rb') as f:
            fc7_w = torch.from_numpy(pickle.load(f))
        with open('data/detectron_weights/fc7_b.pkl', 'rb') as f:
            fc7_b = torch.from_numpy(pickle.load(f))

        self.ctx2pool_grd[0].weight[:opts.att_feat_size].data.copy_(fc7_w)
        self.ctx2pool_grd[0].bias[:opts.att_feat_size].data.copy_(fc7_b)

        # =========== Visual Words Embedding ===========
        # find nearest neighbour class for transfer
        with open('data/detectron_weights/cls_score_w.pkl', 'rb') as f:
            cls_score_w = torch.from_numpy(pickle.load(f))  # 1601x2048
        with open('data/detectron_weights/cls_score_b.pkl', 'rb') as f:
            cls_score_b = torch.from_numpy(pickle.load(f))  # 1601x2048

        # index 0 is background
        assert (len(opts.itod) + 1 == opts.glove_clss.size(0))
        # index 0 is background
        assert (len(opts.vg_cls) == opts.glove_vg_cls.size(0))

        sim_matrix = torch.matmul(opts.glove_vg_cls / torch.norm(opts.glove_vg_cls, dim=1).unsqueeze(1),
                                  (opts.glove_clss / torch.norm(opts.glove_clss, dim=1).unsqueeze(1)).transpose(1, 0))

        max_sim, matched_cls = torch.max(sim_matrix, dim=0)

        vis_classifiers = opts.glove_clss.new(
            opts.detect_size + 1, cls_score_w.size(1)).fill_(0)
        self.vis_classifiers_bias = nn.Parameter(
            opts.glove_clss.new(opts.detect_size + 1).fill_(0))
        vis_classifiers[0] = cls_score_w[0]  # background
        self.vis_classifiers_bias[0].data.copy_(cls_score_b[0])
        for i in range(1, opts.detect_size + 1):
            vis_classifiers[i] = cls_score_w[matched_cls[i]]
            self.vis_classifiers_bias[i].data.copy_(
                cls_score_b[matched_cls[i]])
            # if max_sim[i].item() < 0.9:
            #     print(
            #         'index: {}, similarity: {:.2}, {}, {}'.format(
            #             i, max_sim[i].item(), opts.itod[i],
            #             opts.vg_cls[matched_cls[i]]
            #         )
            #     )

        self.vis_embed[0].weight.data.copy_(vis_classifiers)
        # ====================================================================================

    def _grounder(self, xt, att_feats, mask, bias=None):
        # xt - B, seq_cnt, enc_size
        # att_feats - B, rois_num, enc_size
        # mask - B, rois_num

        B, S, _ = xt.size()
        _, R, _ = att_feats.size()

        if hasattr(self, 'alpha_net'):
            # print('Additive attention for grounding!')
            if self.alpha_net.weight.size(1) == self.att_hid_size:
                dot = xt.unsqueeze(2) + att_feats.unsqueeze(1)
            else:
                dot = torch.cat((xt.unsqueeze(2).expand(B, S, R, self.att_hid_size),
                                 att_feats.unsqueeze(1).expand(B, S, R, self.att_hid_size)), 3)
            dot = F.tanh(dot)
            dot = self.alpha_net(dot).squeeze(-1)
        else:
            # print('Dot-product attention for grounding!')
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
        dot.masked_fill_(expanded_mask, self.min_value)

        return dot

    def get_conv_pooled_feats(self, segs_feat, proposals, mask_boxes, num, region_feats, gt_boxes,
                              overlaps, sample_idx, eval_obj_ground, replicate_feat=True, ):
        """
        input images, get the final FC features, Conv features, ROI pooled features, and create a mask for pointer network
        """
        batch_size = segs_feat.size(0)
        seq_batch_size = self.seq_per_img * batch_size
        num_rois = proposals.size(1)

        # some of the ROI proposals are empty. The number of proposals is recorded in num[:, 1]
        # we construct the mask for masking later on. num_rois + 1 because the fake sentinel ROI we will be generating
        pnt_mask = mask_boxes.new_ones(batch_size, num_rois + 1).bool()
        for i in range(batch_size):
            pnt_mask[i, :num.data[i, 1].long() + 1] = 0

        conv_feats = segs_feat
        # sample_idx_mask = conv_feats.new(
        #     batch_size, conv_feats.size(1), 1).fill_(1).byte()
        sample_idx_mask = conv_feats.new_ones(
            batch_size, conv_feats.size(1), 1).bool()

        for i in range(batch_size):
            sample_idx_mask[i, sample_idx[i, 0]:sample_idx[i, 1]] = 0
        fc_feats = torch.mean(segs_feat, dim=1)
        fc_feats = torch.cat((F.layer_norm(fc_feats, [self.fc_feat_size - self.seg_info_size]),
                              F.layer_norm(self.seg_info_embed(num[:, 3:7].float()), [self.seg_info_size])), dim=-1)

        pool_feats = region_feats
        pool_feats = proj_masking(pool_feats.contiguous(
        ), self.ctx2pool_grd, (pnt_mask[:, 1:] == 0).float())
        g_pool_feats = pool_feats

        # cls_accu = 0
        # ========= visual words embedding =========
        vis_word = torch.LongTensor(
            range(0, self.detect_size + 1)).to(self.device)
        # vis_word = torch.LongTensor(range(0, self.detect_size + 1)).to(segs_feat.get_device())
        # encodings = encodings.cuda(x.get_device())

        vis_word_embed = self.vis_embed(vis_word)
        assert (vis_word_embed.size(0) == self.detect_size + 1)

        p_vis_word_embed = vis_word_embed.view(1, self.detect_size + 1, self.vis_encoding_size) \
            .expand(batch_size, self.detect_size + 1, self.vis_encoding_size).contiguous()
        if hasattr(self, 'vis_classifiers_bias'):
            bias = self.vis_classifiers_bias.type(p_vis_word_embed.type()) \
                .view(1, -1, 1).expand(p_vis_word_embed.size(0),
                                       p_vis_word_embed.size(1), g_pool_feats.size(1))
        else:
            bias = None

        # region-class similarity matrix
        # sim_target = utils.sim_mat_target(overlaps_no_replicate, gt_boxes[:, :, 4].data)  # B, num_box, num_rois
        # sim_mask = (sim_target > 0)

        sim_mat_static = self._grounder(
            p_vis_word_embed, g_pool_feats, pnt_mask[:, 1:], bias)

        # sim_mat_static_update = sim_mat_static.view(batch_size, 1, self.detect_size + 1, num_rois) \
        #     .expand(batch_size, self.seq_per_img, self.detect_size + 1, num_rois).contiguous() \
        #     .view(seq_batch_size, self.detect_size + 1, num_rois)

        sim_mat_static = F.softmax(sim_mat_static, dim=1)

        if self.test_mode:
            cls_pred = 0
            cls_loss = torch.zeros(1).to(self.device)
        else:
            sim_target = utils.sim_mat_target(
                overlaps, gt_boxes[:, :, 5].data)  # B, num_box, num_rois
            sim_mask = (sim_target > 0)
            # if not eval_obj_ground:
            #     masked_sim = torch.gather(sim_mat_static, 1, sim_target)
            #     masked_sim = torch.masked_select(masked_sim, sim_mask)
            #     cls_loss = F.binary_cross_entropy(masked_sim, masked_sim.new(masked_sim.size()).fill_(1))
            # else:
            #     # region classification accuracy
            #     sim_target_masked = torch.masked_select(sim_target, sim_mask)
            #     sim_mat_masked = torch.masked_select(
            #         torch.max(sim_mat_static, dim=1)[1].unsqueeze(1).expand_as(sim_target), sim_mask)
            #     cls_pred = torch.stack((sim_target_masked, sim_mat_masked), dim=1).data

            if sim_mask.sum() == 0:
                cls_loss, cls_pred = torch.zeros(1).to(
                    sim_target.device), torch.zeros(1).to(sim_target.device)
            else:
                masked_sim = torch.gather(sim_mat_static, 1, sim_target)
                masked_sim = torch.masked_select(masked_sim, sim_mask)
                cls_loss = F.binary_cross_entropy(
                    masked_sim, masked_sim.new(masked_sim.size()).fill_(1))
                # region classification accuracy
                sim_target_masked = torch.masked_select(sim_target, sim_mask)
                sim_mat_masked = torch.masked_select(
                    torch.max(sim_mat_static, dim=1)[1].unsqueeze(1).expand_as(sim_target), sim_mask)
                cls_pred = torch.stack(
                    (sim_target_masked, sim_mat_masked), dim=1).data

        if not self.enable_BUTD:
            loc_input = proposals.new(batch_size, num_rois, 5)
            loc_input[:, :, :4] = proposals.data[:, :, :4] / 720.
            loc_input[:, :, 4] = proposals.data[:, :, 4] * \
                1. / self.num_sampled_frm
            loc_feats = self.loc_fc(loc_input)  # encode the locations
            label_feat = sim_mat_static.permute(0, 2, 1).contiguous()
            pool_feats = torch.cat((F.layer_norm(pool_feats, [pool_feats.size(-1)]),
                                    F.layer_norm(
                                        loc_feats, [loc_feats.size(-1)]),
                                    F.layer_norm(label_feat, [label_feat.size(-1)])), 2)

        # ========================================================

        # replicate the feature to map the seq size.
        fc_feats = fc_feats.view(batch_size, 1, self.fc_feat_size) \
            .expand(batch_size, self.seq_per_img, self.fc_feat_size) \
            .contiguous().view(-1, self.fc_feat_size)
        pool_feats = pool_feats.view(batch_size, 1, num_rois, self.pool_feat_size) \
            .expand(batch_size, self.seq_per_img, num_rois, self.pool_feat_size) \
            .contiguous().view(-1, num_rois, self.pool_feat_size)
        g_pool_feats = g_pool_feats.view(batch_size, 1, num_rois, self.vis_encoding_size) \
            .expand(batch_size, self.seq_per_img, num_rois, self.vis_encoding_size) \
            .contiguous().view(-1, num_rois, self.vis_encoding_size)
        pnt_mask = pnt_mask.view(batch_size, 1, num_rois + 1).expand(batch_size, self.seq_per_img, num_rois + 1) \
            .contiguous().view(-1, num_rois + 1)
        overlaps = overlaps.view(batch_size, 1, num_rois, overlaps.size(2)) \
            .expand(batch_size, self.seq_per_img, num_rois, overlaps.size(2)) \
            .contiguous().view(-1, num_rois, overlaps.size(2))

        return fc_feats, conv_feats, pool_feats, g_pool_feats, pnt_mask, overlaps, sample_idx_mask, cls_pred, cls_loss

    def forward(self, segs_feat, proposals, num, mask_boxes, region_feats, gt_boxes, overlaps,
                sample_idx, eval_obj_ground=False, replicate_feat=True):
        """

        :param img: (batch_size, 3, width, height)
        :param proposals: (batch_size, max num of proposals allowed (200), 6), 6-dim: (4 points for the box, proposal index to be used for self.itod to get detection, proposal score)
        :param seq: (batch_size, # of captions for this img, max caption length, 4), 4-dim: (category class, binary class, fine-grain class, from self.wtoi)
        :param num: (batch_size, 3), 3-dim: (# of captions for this img, # of proposals, # of GT bboxs)
        :param mask_boxes: (batch_size, # of captions for this img, max num of GT bbox allowed (100), max caption length)
        :return:
        """
        batch_size = segs_feat.size(0)
        fc_feats, conv_feats, pool_feats, g_pool_feats, pnt_mask, overlaps_expanded, sample_idx_mask, \
            cls_pred, cls_loss = self.get_conv_pooled_feats(segs_feat, proposals, mask_boxes, num, region_feats, gt_boxes,
                                                            overlaps, sample_idx, eval_obj_ground,
                                                            replicate_feat=replicate_feat)

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        pool_feats = proj_masking(
            pool_feats, self.pool_embed, (pnt_mask[:, 1:] == 0).float())

        # properly mask out supposedly empty ROI pooled features
        p_pool_feats = proj_masking(
            pool_feats, self.ctx2pool_fc, (pnt_mask[:, 1:] == 0).float())

        if self.att_input_mode in ('both', 'featmap'):
            conv_feats_splits = torch.split(conv_feats, 2048, 2)
            conv_feats = torch.cat([m(c) for (m, c) in zip(
                self.att_embed, conv_feats_splits)], dim=2)
            conv_feats = conv_feats.permute(0, 2,
                                            1).contiguous()  # inconsistency between Torch TempConv and PyTorch Conv1d
            conv_feats = self.att_embed_aux(conv_feats)
            conv_feats = conv_feats.permute(0, 2,
                                            1).contiguous()  # inconsistency between Torch TempConv and PyTorch Conv1d

            # TODO: call self.context_enc.flatten_parameters()?
            self.context_enc.flatten_parameters()
            conv_feats = self.context_enc(conv_feats)[0]

            conv_feats = conv_feats.masked_fill(sample_idx_mask, 0)
            conv_feats = conv_feats.view(batch_size, 1, self.t_attn_size, self.rnn_size) \
                .expand(batch_size, self.seq_per_img, self.t_attn_size, self.rnn_size) \
                .contiguous().view(-1, self.t_attn_size, self.rnn_size)
            # self.rnn_size (1024) -> self.att_hid_size (512)
            p_conv_feats = self.ctx2att_fc(conv_feats)
        else:
            # dummy
            conv_feats = pool_feats.new(1, 1).fill_(0)
            p_conv_feats = pool_feats.new(1, 1).fill_(0)

        return fc_feats, conv_feats, p_conv_feats, pool_feats, p_pool_feats, g_pool_feats, pnt_mask, overlaps_expanded, cls_pred, cls_loss
