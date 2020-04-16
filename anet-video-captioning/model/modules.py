
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftAttention(nn.Module):
    """
    Soft Attention module
    """

    def __init__(self, rnn_hidden_size, attn_hidden_size, temp=1):
        super(SoftAttention, self).__init__()
        self.softmax = nn.Softmax(dim=1)

        self.h2attn = nn.Linear(rnn_hidden_size, attn_hidden_size)

        self.temp = temp

        # this min_value is used to prevent in the case that,
        # when the mask is all empty, softmax will result in NaN
        self.min_value = -1e8

    def forward(self, h, proj_context, context=None, mask=None, proposal_frame_mask=None, with_sentinel=False):
        """Propagate h through the network.

        h: batch x dim (concat(img, action))
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        """
        attn_h = self.h2attn(h)

        # Get attention
        attn = torch.bmm(proj_context, attn_h.unsqueeze(2)
                         ).squeeze(2)  # batch x seq_len

        attn = attn / self.temp

        if mask is not None:
            if with_sentinel:
                attn.data.masked_fill_(mask.data, -float('inf'))
            else:
                # without sentinel, we need to use a very small value for masking,
                # because there are corner cases where a image has no ROI proposals
                # masking with -inf will thus result in NaN.
                attn.data.masked_fill_(mask.data, self.min_value)

        if proposal_frame_mask is not None:
            # this `frame_masked_attn` is only used to computing (supervised) attention loss
            # since our proposed method does not rely on supervision, we will not update the model
            # based on this loss
            frame_masked_attn = attn.clone()
            if with_sentinel:
                frame_masked_attn.data.masked_fill_(
                    proposal_frame_mask.data, -float('inf'))
            else:
                # without sentinel, we need to use a very small value for masking,
                # because there are corner cases where a image has no ROI proposals
                # masking with -inf will thus result in NaN.
                frame_masked_attn.data.masked_fill_(
                    proposal_frame_mask.data, self.min_value)

        attn = self.softmax(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x seq_len

        if context is not None:
            weighted_context = torch.bmm(
                attn3, context).squeeze(1)  # batch x dim
        else:
            weighted_context = torch.bmm(
                attn3, proj_context).squeeze(1)  # batch x dim

        if proposal_frame_mask is not None:
            return weighted_context, attn, frame_masked_attn
        else:
            return weighted_context, attn, None


class AdditiveSoftAttention(nn.Module):
    """
    Soft Attention module
    """

    def __init__(self, rnn_hidden_size, attn_hidden_size, temp=1):
        super(AdditiveSoftAttention, self).__init__()
        self.softmax = nn.Softmax(dim=1)

        self.rnn_size = rnn_hidden_size
        self.att_hid_size = attn_hidden_size

        self.h2attn = nn.Linear(rnn_hidden_size, attn_hidden_size)
        self.alpha_net = nn.Linear(attn_hidden_size, 1)

        self.temp = temp

        # this min_value is used to prevent in the case that,
        # when the mask is all empty, softmax will result in NaN
        self.min_value = -1e8

    def forward(self, h, proj_context, context=None, mask=None, proposal_frame_mask=None, with_sentinel=False):
        """Propagate h through the network.

        h: batch x dim (concat(img, action))
        context: batch x seq_len x dim
        mask: batch x seq_len indices to be masked
        """
        attn_size = proj_context.size(1)

        attn_h = self.h2attn(h)
        attn_h = attn_h.unsqueeze(1)
        dot = proj_context + attn_h
        dot = torch.tanh(dot)
        dot = dot.view(-1, self.att_hid_size)
        dot = self.alpha_net(dot)
        attn = dot.view(-1, attn_size)

        # Get attention
        # attn = torch.bmm(proj_context, attn_h.unsqueeze(2)).squeeze(2)  # batch x seq_len

        # attn = attn / self.temp

        if mask is not None:
            if with_sentinel:
                attn.data.masked_fill_(mask.data, -float('inf'))
            else:
                # without sentinel, we need to use a very small value for masking,
                # because there are corner cases where a image has no ROI proposals
                # masking with -inf will thus result in NaN.
                attn.data.masked_fill_(mask.data, self.min_value)

        if proposal_frame_mask is not None:
            # this `frame_masked_attn` is only used to computing (supervised) attention loss
            # since our proposed method does not rely on supervision, we will not update the model
            # based on this loss
            frame_masked_attn = attn.clone()
            if with_sentinel:
                frame_masked_attn.data.masked_fill_(
                    proposal_frame_mask.data, -float('inf'))
            else:
                # without sentinel, we need to use a very small value for masking,
                # because there are corner cases where a image has no ROI proposals
                # masking with -inf will thus result in NaN.
                frame_masked_attn.data.masked_fill_(
                    proposal_frame_mask.data, self.min_value)

        attn = self.softmax(attn)
        attn3 = attn.unsqueeze(1)  # batch x 1 x seq_len

        if context is not None:
            weighted_context = torch.bmm(
                attn3, context).squeeze(1)  # batch x dim
        else:
            weighted_context = torch.bmm(
                attn3, proj_context).squeeze(1)  # batch x dim

        if proposal_frame_mask is not None:
            return weighted_context, attn, frame_masked_attn
        else:
            return weighted_context, attn, None


def proj_masking(feat, projector, mask=None):
    """Universal projector and masking"""
    proj_feat = projector(feat.view(-1, feat.size(2)))
    proj_feat = proj_feat.view(feat.size(0), feat.size(1), -1)
    if mask is not None:

        # check that there are at least one element not masked for each sample (row),
        # this is more strict but won't work for features from NBT because
        # some images do not have any regional proposal features
        # assert 0 not in mask.sum(1)

        assert mask.sum() != 0  # check that not all the elements across all samples are set to 0
        return proj_feat * mask.unsqueeze(2).expand_as(proj_feat)
    else:
        return proj_feat
