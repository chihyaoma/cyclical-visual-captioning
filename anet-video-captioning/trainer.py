import os
import sys
import time
import json
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn

import misc.utils as utils
from misc.utils import AverageMeter

# hack to allow the imports of evaluation repos
_SCRIPTPATH_ = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, os.path.join(_SCRIPTPATH_, 'tools/densevid_eval'))
sys.path.insert(0, os.path.join(
    _SCRIPTPATH_, 'tools/densevid_eval/coco-caption'))
sys.path.insert(0, os.path.join(_SCRIPTPATH_, 'tools/anet_entities/scripts'))

from evaluate import ANETcaptions
from eval_grd_anet_entities import ANetGrdEval


class Trainer():
    def __init__(self, opts, dataset, model, optimizer, train_loader, val_loader):
        self.opts = opts

        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def train(self, epoch, tb_logger=None):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        lm_losses = AverageMeter()
        attn_losses = AverageMeter()
        cls_losses = AverageMeter()
        lm_recon_losses = AverageMeter()

        lm_recon_loss = torch.zeros(1).to(self.device)

        # switch to train mode
        self.model.train()

        end = time.time()
        for step, input in enumerate(self.train_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            seg_feat, iseq, gts_seq, num, proposals, bboxs, box_mask, seg_id, \
                region_feat, frm_mask, sample_idx, ppl_mask = input

            proposals = proposals[:, :max(int(max(num[:, 1])), 1), :]
            ppl_mask = ppl_mask[:, :max(int(max(num[:, 1])), 1)]
            bboxs = bboxs[:, :max(int(max(num[:, 2])), 1), :]
            box_mask = box_mask[:, :, :max(int(max(num[:, 2])), 1), :]
            frm_mask = frm_mask[:, :max(
                int(max(num[:, 1])), 1), :max(int(max(num[:, 2])), 1)]
            region_feat = region_feat[:, :max(int(max(num[:, 1])), 1), :]

            # move inputs to device, e.g., cpu or gpu
            segs_feat = seg_feat.float().to(self.device)
            input_seqs = iseq.to(self.device)
            gt_seqs = gts_seq.to(self.device)
            input_num = num.to(self.device)
            input_ppls = proposals.to(self.device)
            mask_ppls = ppl_mask.to(self.device)
            pnt_mask = torch.cat(
                (mask_ppls.new(mask_ppls.size(0), 1).fill_(0), mask_ppls), dim=1)
            gt_bboxs = bboxs.to(self.device)
            mask_bboxs = box_mask.to(self.device)
            mask_frms = frm_mask.to(self.device)
            ppls_feat = region_feat.to(self.device)
            sample_idx = sample_idx.type(input_seqs.type())

            if self.opts.att_model == 'cyclical':
                training_output = self.model(segs_feat, input_seqs, gt_seqs,
                                             input_num, input_ppls, gt_bboxs,
                                             mask_bboxs, ppls_feat, mask_frms,
                                             sample_idx, pnt_mask)

                if self.opts.train_decoder_only:
                    lm_loss, att2_loss, ground_loss, cls_loss = training_output
                else:
                    lm_loss, att2_loss, ground_loss, cls_loss, lm_recon_loss = training_output

            else:
                raise ValueError(
                    'Unknown att_model: {}'.format(self.opts.att_model))

            loss = self.opts.xe_loss_weight * lm_loss + \
                self.opts.w_att2 * att2_loss + \
                self.opts.w_cls * cls_loss
            loss += self.opts.caption_consistency_loss_weight * lm_recon_loss

            # record losses
            tb_step = self.opts.batch_size * self.opts.seq_per_img
            # losses.update(loss.item(), tb_step)
            lm_losses.update(lm_loss.item(), tb_step)
            attn_losses.update(att2_loss.item(), tb_step)
            cls_losses.update(cls_loss.item(), tb_step)

            # zero the gradients before backward pass
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.opts.grad_clip)
            self.optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if step % self.opts.disp_interval == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'LM Loss {lm_loss.val:.4f} ({lm_loss.avg:.4f})\t'
                      'Attn Loss {attn_loss.val:.4f} ({attn_loss.avg:.4f})\t'
                      'Cls Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                      'Recon Loss {recon_loss.val:.4f} ({recon_loss.avg:.4f})\t'
                      .format(epoch, step, len(self.train_loader) - 1,
                              batch_time=batch_time, data_time=data_time,
                              lm_loss=lm_losses, attn_loss=attn_losses,
                              cls_loss=cls_losses, recon_loss=lm_recon_losses))

        if tb_logger:
            tb_logger.add_scalar('train/learning_rate',
                                 self.optimizer.param_groups[0]['lr'], epoch)
            tb_logger.add_scalar('train/lm_loss', lm_losses.avg, epoch)
            tb_logger.add_scalar('train/attn_loss', attn_losses.avg, epoch)
            tb_logger.add_scalar('train/cls_loss', cls_losses.avg, epoch)
            tb_logger.add_scalar('train/lm_recon_loss',
                                 lm_recon_losses.avg, epoch)

        return

    def eval(self, epoch, tb_logger=None):
        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.model.eval()

        num_show = 0
        predictions = defaultdict(list)
        raw_caption_file = json.load(open(self.opts.input_raw_cap))

        if self.opts.eval_obj_grounding:
            grd_output = defaultdict(dict)

            lemma_det_dict = {self.opts.wtol[key]: idx for key,
                              idx in self.opts.wtod.items() if key in self.opts.wtol}
            print('{} classes have the associated lemma word!'.format(
                len(lemma_det_dict)))

        with torch.no_grad():
            end = time.time()
            for step, input in enumerate(self.val_loader):
                # measure data loading time
                data_time.update(time.time() - end)

                seg_feat, iseq, gts_seq, num, proposals, bboxs, box_mask, \
                    seg_id, region_feat, frm_mask, sample_idx, ppl_mask = input

                proposals = proposals[:, :max(int(max(num[:, 1])), 1), :]
                ppl_mask = ppl_mask[:, :max(int(max(num[:, 1])), 1)]
                region_feat = region_feat[:, :max(int(max(num[:, 1])), 1), :]

                segs_feat = seg_feat.float().to(self.device)
                input_num = num.to(self.device)
                input_ppls = proposals.to(self.device)
                mask_ppls = ppl_mask.to(self.device)
                pnt_mask = torch.cat(
                    (mask_ppls.new(mask_ppls.size(0), 1).fill_(0), mask_ppls), dim=1)
                ppls_feat = region_feat.to(self.device)
                # sample_idx = sample_idx.type(input_num.type())
                eval_opt = {
                    'sample_max': 1, 'beam_size': self.opts.beam_size, 'inference_mode': True}
                dummy = input_ppls.new(input_ppls.size(0)).byte().fill_(0)

                input_seqs = iseq.to(self.device)
                gt_seqs = gts_seq.to(self.device)
                input_ppls = proposals.to(self.device)
                gt_bboxs = bboxs.to(self.device)
                mask_bboxs = box_mask.to(self.device)
                mask_frms = frm_mask.to(self.device)

                batch_size = input_ppls.size(0)

                if self.opts.att_model == 'cyclical':
                    seq, att2_weights, sim_mat = self.model(segs_feat, input_seqs, gt_seqs, input_num,
                                                            input_ppls, gt_bboxs, mask_bboxs, ppls_feat, mask_frms,
                                                            sample_idx, pnt_mask,
                                                            True)
                else:
                    raise ValueError(
                        'Unknown att_model: {}'.format(self.opts.att_model))

                # save localization results on generated sentences
                if self.opts.eval_obj_grounding:
                    assert self.opts.beam_size == 1, 'only support beam_size is 1'

                    att2_ind = torch.max(att2_weights.view(batch_size, att2_weights.size(1),
                                                           self.opts.num_sampled_frm, self.opts.num_prop_per_frm), dim=-1)[1]
                    obj_bbox_att2 = torch.gather(input_ppls.view(-1, self.opts.num_sampled_frm, self.opts.num_prop_per_frm, 7)
                                                 .permute(0, 2, 1, 3).contiguous(), 1,
                                                 att2_ind.unsqueeze(-1).expand((batch_size,
                                                                                att2_ind.size(
                                                                                    1), self.opts.num_sampled_frm,
                                                                                input_ppls.size(-1))))  # Bx20x10x7

                    for i in range(seq.size(0)):
                        vid_id, seg_idx = seg_id[i].split('_segment_')
                        seg_idx = str(int(seg_idx))
                        tmp_result = {'clss': [], 'idx_in_sent': [],
                                      'bbox_for_all_frames': []}

                        for j in range(seq.size(1)):
                            if seq[i, j].item() != 0:
                                lemma = self.opts.wtol[self.opts.itow[str(
                                    seq[i, j].item())]]
                                if lemma in lemma_det_dict:
                                    tmp_result['bbox_for_all_frames'].append(
                                        obj_bbox_att2[i, j, :, :4].tolist())
                                    tmp_result['clss'].append(
                                        self.opts.itod[lemma_det_dict[lemma]])
                                    # redundant, for the sake of output format
                                    tmp_result['idx_in_sent'].append(j)
                            else:
                                break
                        grd_output[vid_id][seg_idx] = tmp_result

                sents = utils.decode_sequence(self.dataset.itow, self.dataset.itod, self.dataset.ltow, self.dataset.itoc,
                                              self.dataset.wtod, seq.data, self.opts.vocab_size, self.opts)

                for k, sent in enumerate(sents):
                    vid_idx, seg_idx = seg_id[k].split('_segment_')
                    seg_idx = int(seg_idx)

                    predictions[vid_idx].append(
                        {'sentence': sent,
                         'timestamp': raw_caption_file[vid_idx]['timestamps'][seg_idx]})

                    if num_show < 20:
                        print('segment %s: %s' % (seg_id[k], sent))
                        num_show += 1

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      .format(epoch, step, len(self.val_loader) - 1, batch_time=batch_time, data_time=data_time))

        lang_stats = defaultdict(float)
        if self.opts.language_eval:
            print('Total videos to be evaluated %d' % (len(predictions)))

            submission = 'results/' + 'densecap-' + \
                self.opts.val_split + '-' + self.opts.id + '.json'
            dense_cap_all = {'version': 'VERSION 1.0', 'results': predictions,
                             'external_data': {'used': 'true',
                                               'details': 'Visual Genome for Faster R-CNN pre-training'}}
            with open(submission, 'w') as f:
                json.dump(dense_cap_all, f)

            references = self.opts.densecap_references
            verbose = self.opts.densecap_verbose
            tious_lst = [0.3, 0.5, 0.7, 0.9]
            evaluator = ANETcaptions(ground_truth_filenames=references,
                                     prediction_filename=submission,
                                     tious=tious_lst,
                                     max_proposals=1000,
                                     verbose=verbose)
            evaluator.evaluate()

            for m, v in evaluator.scores.items():
                lang_stats[m] = np.mean(v)

            print('\nResults Summary (lang eval):')
            print('Printing language evaluation metrics...')
            for m, s in lang_stats.items():
                print('{}: {:.3f}'.format(m, s * 100))
            print('\n')

            if tb_logger:
                for metric, score in lang_stats.items():
                    tb_logger.add_scalar(
                        'lang_eval/{}/{}'.format(self.opts.val_split, metric), score, epoch)

        if self.opts.test_mode and self.opts.eval_obj_grounding:
            print('*' * 62)
            print('*  [WARNING] Grounding eval unavailable for the test set!\
        *\n*            Please submit your results to the eval server!  *')
            print('*' * 62)

        if self.opts.eval_obj_grounding:
            # write attention results to file
            attn_file = 'results/attn-gen-sent-results-' + \
                self.opts.val_split + '-' + self.opts.id + '.json'
            with open(attn_file, 'w') as f:
                json.dump(
                    {'results': grd_output,
                     'eval_mode': 'gen',
                     'external_data': {'used': True,
                                       'details': 'Object detector pre-trained on Visual Genome on object detection task.'}
                     },
                    f)

            if not self.opts.test_mode:
                # offline eval
                evaluator = ANetGrdEval(reference_file=self.opts.grd_reference, submission_file=attn_file,
                                        split_file=self.opts.split_file, val_split=[
                                            self.opts.val_split],
                                        iou_thresh=0.5)

                print('\nResults Summary (generated sent):')
                print(
                    'Printing attention accuracy on generated sentences, per class and per sentence, respectively...')
                prec_all, recall_all, f1_all, prec_all_per_sent, rec_all_per_sent, \
                    f1_all_per_sent = evaluator.grd_eval(mode='all')
                prec_loc, recall_loc, f1_loc, prec_loc_per_sent, rec_loc_per_sent, \
                    f1_loc_per_sent = evaluator.grd_eval(mode='loc')
            else:
                print('*'*62)
                print('*  [WARNING] Grounding eval unavailable for the test set!\
        *\n*            Please submit your result files under directory *\
        \n*            results/ to the eval server!                    *')
                print('*'*62)

            if tb_logger:
                tb_logger.add_scalar(
                    'grounding_{}/Prec_all'.format(self.opts.val_split), prec_all, epoch)
                tb_logger.add_scalar(
                    'grounding_{}/recall_all'.format(self.opts.val_split), recall_all, epoch)
                tb_logger.add_scalar(
                    'grounding_{}/f1_all'.format(self.opts.val_split), prec_all, epoch)
                tb_logger.add_scalar(
                    'grounding_{}/prec_loc'.format(self.opts.val_split), prec_loc, epoch)
                tb_logger.add_scalar(
                    'grounding_{}/recall_loc'.format(self.opts.val_split), recall_loc, epoch)
                tb_logger.add_scalar(
                    'grounding_{}/f1_loc'.format(self.opts.val_split), f1_loc, epoch)

        return lang_stats
