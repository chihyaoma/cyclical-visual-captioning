
import torch
import torch.nn as nn

from model.modules import SoftAttention, AdditiveSoftAttention


class TopDownDecoderCore(nn.Module):
    def __init__(self, opts):
        super(TopDownDecoderCore, self).__init__()

        self.opts = opts

        self.att_lstm = nn.LSTMCell(opts.input_encoding_size + opts.rnn_size * 2, opts.rnn_size)


        self.i2h_2 = nn.Linear(opts.rnn_size * 2, opts.rnn_size)
        self.h2h_2 = nn.Linear(opts.rnn_size, opts.rnn_size)

        self.localied_fc = nn.Linear(opts.rnn_size, opts.att_hid_size)

        if opts.softattn_type == 'additive':
            self.soft_attn = AdditiveSoftAttention(opts.rnn_size, opts.att_hid_size, temp=opts.softmax_temp)
        else:
            self.soft_attn = SoftAttention(opts.rnn_size, opts.att_hid_size, temp=opts.softmax_temp)

        self.lang_lstm = nn.LSTMCell(opts.rnn_size * 2, opts.rnn_size)
        self.dropout = nn.Dropout(opts.drop_prob_lm)

    def forward(self, embedded_word, fc_feats, conv_feats, p_conv_feats, pool_feats, p_pool_feats, pnt_mask,
                state, proposal_frame_mask=None, with_sentinel=False):
        """
        Two-layered attentional LSTM as decoder for caption generation

        :param embedded_word:
        :param fc_feats:
        :param localized_weighted_feats: list of (batch_size, d-dim)
        :param pnt_mask:
        :param state:
        :return:
        """
        batch_size = embedded_word.size(0)
        prev_h = state[0][-1]  # language LSTM previous output hidden state

        if self.opts.global_img_in_attn_lstm:
            attn_lstm_input = torch.cat([prev_h, fc_feats, embedded_word], dim=1)
        else:
            attn_lstm_input = torch.cat([prev_h, embedded_word], dim=1)

        h_attn, c_attn = self.att_lstm(attn_lstm_input, (state[0][0], state[1][0]))

        # attention selection on set of ROIs, we exclude the first column of pnt_mask,
        # because we do not have fake bbox from sentinel in baseline model
        weighted_pool_feat, roi_attn, frame_masked_attn = self.soft_attn(
            h_attn, p_pool_feats, context=pool_feats, mask=pnt_mask, proposal_frame_mask=proposal_frame_mask, with_sentinel=with_sentinel)
        attn_conv, _, _ = self.soft_attn(h_attn, p_conv_feats, context=conv_feats)

        # lang_lstm_input = torch.cat([attn, h_attn], dim=1)
        lang_lstm_input = torch.cat([weighted_pool_feat + attn_conv, h_attn], dim=1)

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))
        output = self.dropout(h_lang)

        state = (torch.stack([h_attn, h_lang]), torch.stack([c_attn, c_lang]))

        return output, state, roi_attn, frame_masked_attn, weighted_pool_feat


class AttenedDecoderCore(nn.Module):
    def __init__(self, opts, att_lstm, lang_lstm):
        super(AttenedDecoderCore, self).__init__()

        self.opts = opts

        self.att_lstm = att_lstm

        if opts.softattn_type == 'additive':
            self.soft_attn = AdditiveSoftAttention(opts.rnn_size, opts.att_hid_size, temp=opts.softmax_temp)
        else:
            self.soft_attn = SoftAttention(opts.rnn_size, opts.att_hid_size, temp=opts.softmax_temp)

        self.lang_lstm = lang_lstm

        self.dropout = nn.Dropout(opts.drop_prob_lm)

    def forward(self, embedded_word, fc_feats, weighted_pool_feat, attn_conv, state, with_sentinel=False):
        """
        Two-layered attentional LSTM as decoder for caption generation

        :param embedded_word:
        :param fc_feats:
        :param localized_weighted_feats: list of (batch_size, d-dim)
        :param pnt_mask:
        :param state:
        :return:
        """
        prev_h = state[0][-1]  # language LSTM previous output hidden state

        if self.opts.global_img_in_attn_lstm:
            attn_lstm_input = torch.cat([prev_h, fc_feats, embedded_word], dim=1)
        else:
            attn_lstm_input = torch.cat([prev_h, embedded_word], dim=1)

        h_attn, c_attn = self.att_lstm(attn_lstm_input, (state[0][0], state[1][0]))

        lang_lstm_input = torch.cat([weighted_pool_feat + attn_conv, h_attn], dim=1)

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))
        output = self.dropout(h_lang)

        state = (torch.stack([h_attn, h_lang]), torch.stack([c_attn, c_lang]))

        return output, state
