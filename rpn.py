from __future__ import absolute_import
import torch.nn as nn
import numpy as np
from .proposal_layer import _ProposalLayer
from torch.autograd import Variable
import torch


class _RPN(nn.Module):
    """ region proposal network """
    def __init__(self, din):
        super(_RPN, self).__init__()
        self.din = din  # get depth of input feature map, e.g., 512
        self.feat_stride = 1
        self.anchors = np.array([(0, 0, 4, 4), (0, 0, 8, 8),
                                 (0, 0, 4, 8), (0, 0, 6, 8)])  # anchor:(start_x,start_y,cols, rows)

        # define proposal layer
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchors, self.din)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0
        self.theta = 1

    def forward(self, base_feat, central_pos,im_info, gt_boxes):
        """
        :param base_feat: shape=(batch_size * num_frames,128,11,11)
        :param central_pos: shape=(batch_size,)
        :param im_info: 10.2
        :param gt_boxes: shape=(batch_size,4)
        """

        output,proposal_loss = self.RPN_proposal((base_feat ,central_pos,im_info,gt_boxes))   # output.shape=(batch_size * num_frames, 9)

        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None

            # compute classification loss
            self.rpn_loss_cls = proposal_loss   # log-loss

            # compute bbox regression loss
            pred_box=output.new(central_pos.shape[0],4).zero_()
            for i in range(central_pos.shape[0]):
                pred_box[i,:]=output[central_pos[i]+300*i,4:8]
            delta_box=pred_box-gt_boxes
            self.rpn_loss_box = self.smooth_l1_loss(delta_box)

            print("rpn_loss_cls={:.3f},   rpn_loss_box={:.3f}".format(self.rpn_loss_cls[0], self.rpn_loss_box[0]))

        else:
            pred_box_top3 = output.new(3, 4).zero_().cpu()
            for i in range(3):
                pred_box_top3[i, :] = output[central_pos[i] + 300 * i, 4:8]
            print(pred_box_top3)   # the first 5 pred boxes in test batch

        rpn_loss = self.rpn_loss_cls + self.theta * self.rpn_loss_box
        return output, rpn_loss


    def smooth_l1_loss(self,delta_boxes):
        delta_box_points=delta_boxes.view(-1,)
        batch_size = delta_box_points.shape[0]
        loss=Variable(torch.Tensor([0])).cuda()
        for point in delta_box_points:
            if abs(point)<3:
                loss+=point**2
            else:
                loss+=abs(point)
        return loss/batch_size