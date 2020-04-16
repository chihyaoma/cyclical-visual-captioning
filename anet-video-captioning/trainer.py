import time

import torch

from misc.utils import AverageMeter


class Trainer():
    def __init__(self, opts, model, xe_criterion, optimizer, train_loader, val_loader):
        self.opts = opts
        self.optimizer = optimizer

        # define loss criterion
        self.xe_criterion = xe_criterion

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, epoch, tb_logger=None):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        lm_losses = AverageMeter()
        attn_losses = AverageMeter()
        cls_losses = AverageMeter()

        # switch to train mode
        self.model.train()

        # if not finetuning the backbone ConvNet, we change it to evaluation mode
        if not self.opts.finetune_cnn:
            self.model.module.roi_feat_extractor.cnn.eval()

        end = time.time()
        for iter, input in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            seg_feat, iseq, gts_seq, num, proposals, bboxs, box_mask, seg_id, region_feat, frm_mask, sample_idx, ppl_mask = input

            proposals = proposals[:, :max(int(max(num[:, 1])), 1), :]
            ppl_mask = ppl_mask[:, :max(int(max(num[:, 1])), 1)]
            bboxs = bboxs[:, :max(int(max(num[:, 2])), 1), :]
            box_mask = box_mask[:, :, :max(int(max(num[:, 2])), 1), :]
            frm_mask = frm_mask[:, :max(int(max(num[:, 1])), 1), :max(int(max(num[:, 2])), 1)]
            region_feat = region_feat[:, :max(int(max(num[:, 1])), 1), :]

            # move inputs to device, e.g., cpu or gpu
            segs_feat = seg_feat.float().to(self.device)  # TODO: not sure why `seg_feat` was in double format
            input_seqs = iseq.to(self.device)
            gt_seqs = gts_seq.to(self.device)
            input_num = num.to(self.device)
            input_ppls = proposals.to(self.device)
            mask_ppls = ppl_mask.to(self.device)
            pnt_mask = torch.cat((mask_ppls.new(mask_ppls.size(0), 1).fill_(0), mask_ppls), dim=1)
            gt_bboxs = bboxs.to(self.device)
            mask_bboxs = box_mask.to(self.device)
            mask_frms = frm_mask.to(self.device)
            ppls_feat = region_feat.to(self.device)
            sample_idx = sample_idx.type(input_seqs.type())

            lm_loss, att2_loss, ground_loss, cls_loss = self.model(segs_feat, input_seqs, gt_seqs, input_num,
                                                                   input_ppls, gt_bboxs, mask_bboxs, ppls_feat,
                                                                   mask_frms,
                                                                   sample_idx, pnt_mask, 'MLE')

            loss = self.opts.lm_loss_weight * lm_loss + \
                   self.opts.w_att2 * att2_loss + \
                   self.opts.w_cls * cls_loss

            # record losses
            tb_step = self.opts.batch_size * self.opts.seq_per_img
            losses.update(loss.item(), tb_step)
            lm_losses.update(lm_loss.item(), tb_step)
            attn_losses.update(att2_loss.item(), tb_step)
            cls_losses.update(cls_loss.item(), tb_step)

            # zero the gradients before backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'LM loss {lm_loss.val:.4f} ({lm_loss.avg:.4f})\t'
                  'Attn loss {attn_loss.val:.4f} ({attn_loss.avg:.4f})\t'
                  'Cls loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'.format(
                epoch, iter, len(self.train_loader) - 1, batch_time=batch_time,
                data_time=data_time, lm_loss=lm_losses, attn_loss=attn_losses, cls_loss=cls_losses))

        if tb_logger:
            pass

        return

    def eval(self, epoch, tb_logger=None):
        batch_time = AverageMeter()
        self.model.eval()

        with torch.no_grad():
            end = time.time()
            for iter, input in enumerate(self.val_loader):

                # if self.opts.vis_attn:
                #     seg_feat, iseq, gts_seq, num, proposals, bboxs, box_mask, seg_id, seg_show, seg_dim_info, region_feat, frm_mask, sample_idx, ppl_mask = input
                # else:
                seg_feat, iseq, gts_seq, num, proposals, bboxs, box_mask, seg_id, region_feat, frm_mask, sample_idx, ppl_mask = input

                proposals = proposals[:, :max(int(max(num[:, 1])), 1), :]
                ppl_mask = ppl_mask[:, :max(int(max(num[:, 1])), 1)]
                region_feat = region_feat[:, :max(int(max(num[:, 1])), 1), :]

                segs_feat = seg_feat.float().to(self.device)  # TODO: not sure why `seg_feat` was in double format
                input_num = num.to(self.device)
                input_ppls = proposals.to(self.device)
                mask_ppls = ppl_mask.to(self.device)
                pnt_mask = torch.cat((mask_ppls.new(mask_ppls.size(0), 1).fill_(0), mask_ppls), dim=1)
                ppls_feat = region_feat.to(self.device)
                # sample_idx = sample_idx.type(input_num.type())
                eval_opt = {'sample_max': 1, 'beam_size': self.opts.beam_size, 'inference_mode': True}
                dummy = input_ppls.new(input_ppls.size(0)).byte().fill_(0)

        return
