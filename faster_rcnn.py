import torch.nn as nn
from faster_model.utils.config import cfg
from faster_model.rpn.rpn import _RPN
import torch

class _fasterRCNN(nn.Module):
    def __init__(self):
        super(_fasterRCNN, self).__init__()
        self.deconv_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=1, stride=2)
        self.deconv_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=1, stride=2, padding=1)

        # define rpn
        self.RCNN_rpn = _RPN(din=128)
        # self.available_anchors = [(0, 0, 2, 2), (0, 0, 2, 3), (0, 0, 3, 2), (0, 0, 3, 3),
        #                           (1, 0, 3, 2), (1, 0, 4, 2), (1, 0, 3, 3), (1, 0, 4, 3),
        #                           (2, 0, 4, 2), (2, 0, 4, 3),
        #                           (0, 1, 2, 3), (0, 1, 3, 3), (0, 1, 3, 3), (0, 1, 3, 4),
        #                           (1, 1, 3, 3), (1, 1, 4, 3), (1, 1, 3, 4), (1, 1, 4, 4),
        #                           (2, 1, 4, 3), (2, 1, 4, 4),
        #                           (0, 2, 2, 4), (0, 2, 3, 4),
        #                           (1, 2, 3, 4), (1, 2, 4, 4),
        #                           (2, 2, 4, 4)]
        self.linear_rcnn=nn.Linear(2048,1024)
        self.relu=nn.LeakyReLU()

    def forward(self, base_feat, central_pos,im_info, gt_boxes):
        # feed base feature map tp RPN to obtain rois
        batch_size=base_feat.shape[0]
        base_feat=base_feat.view(-1,512,4,4)  # base_feat.shape=(batch_size * num_frames, 512, 4, 4)
        deconv_base_feat=self.deconv_2(self.deconv_1(base_feat))  # deconv_base_feat.shape=(batch_size * num_frames,128,11,11)

        rois, rpn_loss = self.RCNN_rpn(deconv_base_feat, central_pos,im_info, gt_boxes)
        anchor_areas=rois[:,:4]  # anchor_areas.shape=(batch_size * num_frames, 4)

        # if it is training phase, then use ground truth bboxes for refining
        '''
        if self.training:
            "*** Add some ground truth bboxes here ***"
        '''

        rcnn_output=deconv_base_feat.new(deconv_base_feat.shape[0],128,4,4).zero_()
        not_in = 0
        for i in range(anchor_areas.shape[0]):
            anchor_area=tuple(anchor_areas[i].cpu().detach().numpy())
            anchor_area=(int(anchor_area[0]),int(anchor_area[1]),int(anchor_area[2]),int(anchor_area[3]))
            if not self.check(anchor_area):
                not_in+=1
                anchor_area=(1,1,10,10)
            pre_pooling=deconv_base_feat[i,:,anchor_area[0]:anchor_area[2],anchor_area[1]:anchor_area[3]]
            if pre_pooling.shape[1:]==(4,4):
                after_pooling=pre_pooling
                rcnn_output[i]=after_pooling
            if pre_pooling.shape[1:]==(8,8):
                pooling_layer=nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
                after_pooling=pooling_layer.forward(pre_pooling)
                rcnn_output[i] = after_pooling
            if pre_pooling.shape[1:]==(4,8):
                pooling_layer = nn.MaxPool1d(kernel_size=2,stride=2)
                after_pooling = pooling_layer.forward(pre_pooling)
                rcnn_output[i] = after_pooling
            if pre_pooling.shape[1:]==(6,8):
                pooling_layer = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2),padding=(1,0))
                after_pooling = pooling_layer.forward(pre_pooling)
                rcnn_output[i] = after_pooling
            if pre_pooling.shape[1:]==(9,9):
                pooling_layer=nn.MaxPool2d(kernel_size=(3,3),stride=(2,2))
                after_pooling=pooling_layer.forward(pre_pooling)
                rcnn_output[i] = after_pooling
        print("num of random anchor: ",not_in)

        rcnn_output=rcnn_output.view(batch_size,-1,128,4,4)   # rcnn_output.shape=(batch_size, 300, 128, 4, 4)
        rcnn_output=self.relu(self.linear_rcnn(rcnn_output.view(batch_size,-1,2048)))  # rcnn_output.shape=(batch_size, 300, 1024)

        return rcnn_output, rpn_loss



    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_proposal.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_proposal.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_proposal.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)

    def check(self,anchor):
        for rect in anchor:
            if rect<0 or rect>=11:
                return False
        anchor_shape=(anchor[2]-anchor[0],anchor[3]-anchor[1])
        if anchor_shape not in [(4,4),(8,8),(4,8),(6,8)]:
            return False
        return True


    def create_architecture(self):
        self._init_weights()