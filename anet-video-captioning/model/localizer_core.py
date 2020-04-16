

import torch.nn as nn

from model.modules import SoftAttention

class LocalizerNoLSTMCore(nn.Module):
    """Core for attention localizer
    contains Top-Down Attention LSTM cell with visual sentinel
    """
    def __init__(self, opts):
        super(LocalizerNoLSTMCore, self).__init__()

        self.opts = opts
        self.soft_attn = SoftAttention(opts.input_encoding_size, opts.att_hid_size, temp=opts.localizer_softmax_temp)

    def forward(self, embedded_word, fc_feats, conv_feats, p_conv_feats, pool_feats, p_pool_feats, attn_mask, state,
                consistent_decoder_state, proposal_frame_mask=None, with_sentinel=False):
        """

        :param embedded_word:
        :param fc_feats: (batch_size, d-dim feature), representing the global representation of an image
        :param conv_feats: (batch_size, width x height, d-dim feature)
        :param p_conv_feats: projected conv_feats
        :param pool_feats: (batch_size, number of ROIs, d-dim feature), some ROIs could be empty
        :param p_pool_feats: projected pool_feats
        :param att_mask: mask for attention masking
        :param pnt_mask: mask for pointer network
        :param state: hidden and cell states of the LSTM(s)
        :return:
        """
        # given each embedded word, we perform attention (localization) on ROIs
        attn_input = embedded_word

        # some of the ROI features are empty; we use masking
        localized_weighted_feat, localized_prob, _ = self.soft_attn(
            attn_input, p_pool_feats, context=pool_feats, mask=attn_mask, proposal_frame_mask=proposal_frame_mask, with_sentinel=with_sentinel)

        localized_conv, _, _ = self.soft_attn(attn_input, p_conv_feats, context=conv_feats)

        return localized_weighted_feat, localized_conv, localized_prob, state