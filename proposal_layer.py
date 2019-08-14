from __future__ import absolute_import
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F



DEBUG = False

class _ProposalLayer(nn.Module):
    def __init__(self, feat_stride, anchors, din):
        super(_ProposalLayer, self).__init__()

        self._feat_stride = feat_stride
        self._anchors = torch.from_numpy(anchors).float()
        self._num_anchors = anchors.shape[0]   # e.g. self._num_anchors=4

        # define the convrelu layers processing input feature map
        self.RPN_Conv = nn.Conv2d(din, 128, 2, 1, 1, bias=True)
        # define bg/fg classifcation score layer

        self.nc_score_out = len(anchors) * 2  # 2(positive/negative) * 4
        self.RPN_cls_score = nn.Conv2d(128, self.nc_score_out, kernel_size=2, stride=1, padding=0)

        # define anchor box offset prediction layer
        self.nc_bbox_out = len(anchors) * 4  # 4(x/y/w/h) * 4
        self.RPN_bbox_pred = nn.Conv2d(128, self.nc_bbox_out, kernel_size=2, stride=1, padding=0)


    def forward(self, input):   # input: (base_feature, central_pos,im_info, gt_boxes)
        base_feat= input[0]   # base_feat.shape=(batch_size * num_frames,128,11,11)
        central_pos=input[1]
        im_info = input[2]
        gt_boxes = input[3]

        # return feature map after conv_relu layer
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True)  # rpn_conv1.shape=(batch_size * num_frames,128,12,12)

        # get rpn classification score
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)  # rpn_cls_score.shape=(batch_size * num_frames,8,11,11)

        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape,
                                    self.nc_score_out)  # rpn_cls_prob.shape=(batch_size * num_frames,8,11,11)

        # get rpn offsets to the anchor boxes
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)  # rpn_bbox_pred.shape=(batch_size * num_frames,16,11,11)

        # the first set of _num_anchors channels are bg probs, the second set are the fg probs.
        # So here "scores" fetch the last 4 lines in dim=1, which refers to the foreground.
        scores = rpn_cls_prob[:, self._num_anchors:, :, :]
        bbox_deltas = rpn_bbox_pred
        batch_size = bbox_deltas.size(0)


        feat_height, feat_width = scores.size(2), scores.size(3)
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(scores).float()   # shifts.shape=(121,4)

        A = self._num_anchors  # A=4
        K = shifts.size(0)  # K=121

        self._anchors = self._anchors.type_as(scores)
        anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)

        proposals = anchors.view(1, A * K, 4).expand(batch_size, A * K, 4)  # proposals.shape=(batch_size * num_frames, 484,4)
        # Here, we obtain 4 anchors starting from every feature map point.
        # Note that anchors[:,:,2:4]= right-down point +1


        # Transpose and reshape predicted bbox transformations to get them into the same order as the anchors:
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous()
        bbox_deltas = bbox_deltas.view(batch_size, -1, 4)   # bbox_deltas.shape=(batch_size * num_frames,121,4)
        # delta_loss=torch.abs(bbox_deltas).mean()   # L1 regularization of bbox_deltas


        dx_1 = bbox_deltas[:, :, 0::4]  # dx.shape=(batch_size * num_frames,121, 1)
        dy_1 = bbox_deltas[:, :, 1::4]
        dx_2 = bbox_deltas[:, :, 2::4]
        dy_2 = bbox_deltas[:, :, 3::4]

        proposals_delta=proposals.new(batch_size,A*K,4).zero_()
        proposals_delta[:, :, 0::4]=proposals[:, :, 0::4]  + dx_1
        proposals_delta[:, :, 1::4] =proposals[:, :, 1::4]  + dy_1
        proposals_delta[:, :, 2::4] =proposals[:, :, 2::4]  + dx_2
        proposals_delta[:, :, 3::4] =proposals[:, :, 3::4]  + dy_2
        proposals_delta *= im_info


        scores_keep = scores.view(-1,K,4)   # scores_keep.shape=(batch_size(64) * num_frames(300),121,4)
        proposals_keep = proposals_delta   # proposals_keep.shape=(batch_size(64) * num_frames(300),484,4)
        scores_first, order_first = torch.max(scores_keep,dim=2)
        scores_second, order_second = torch.max(scores_first, dim=1)

        output = scores.new(batch_size, 9).zero_()  # 4 for anchors, 4 for boxes and last dimension for softmax score
        log_loss = 0
        curr_video=0
        for i in range(batch_size):
            box_id=order_second[i]*4+order_first[i][order_second[i]]
            score=scores_second[i]
            box = proposals_keep[i, box_id]
            anchor=(box/im_info).int()

            # calculate ground truth box loss
            if curr_video<central_pos.shape[0] and i==300*curr_video+central_pos[curr_video]:
                gt_box=gt_boxes[curr_video]
                if self.IoU(gt_box,box)>0.7:
                    log_loss-=torch.log(score+1.0e-5)
                else:
                    log_loss-=torch.log(1-score+1.0e-5)
                curr_video+=1

            output[i, :4] = anchor
            output[i, 4:8] = box
            output[i, -1] = score

        # log_loss/=gt_boxes.shape[0]
        proposal_loss=log_loss
        return output, proposal_loss

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(input_shape[0],
                   int(d),
                   int(float(input_shape[1] * input_shape[2]) / float(d)),
                   input_shape[3])
        return x




    def IoU(self, box1, box2):
        xi1 = max(box1[0], box2[0])
        yi1 = max(box1[1], box2[1])
        xi2 = max(box1[2], box2[2])
        yi2 = max(box1[3], box2[3])
        inter_area = (xi2 - xi1) * (yi2 - yi1)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        iou = inter_area / union_area
        return iou