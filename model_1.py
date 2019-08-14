import torch.nn.functional as F
from util.graph_definition import *
from faster_model.faster_rcnn import faster_rcnn
import torch


class selfAttn(nn.Module):
    def __init__(self, feature_size, hidden_size, num_desc):
        super(selfAttn, self).__init__()
        self.linear_1 = nn.Linear(feature_size, hidden_size, bias=False)
        self.linear_2 = nn.Linear(hidden_size, num_desc, bias=False)
        self.num_desc = num_desc
        self.bn = nn.BatchNorm1d(feature_size)

    def forward(self, model_input):   # model_input.shape=(batch_size,300,256)
        reshaped_input = model_input
        s1 = F.tanh(self.linear_1(reshaped_input))
        # Each row of A represents different layer of information for the video.
        A = F.softmax(self.linear_2(s1), dim=1)   # A.shape=(batch_size, 300,50)
        M = self.bn(torch.bmm(model_input.permute(0, 2, 1), A)).permute(0, 2, 1).contiguous()
        # AAT = torch.bmm(A.permute(0, 2, 1), A)
        # I = Variable(torch.eye(self.num_desc)).cuda()
        # P = torch.norm(AAT - I, 2)
        # penal = P * P / model_input.shape[0]

        return M, A


class conv_lstm(nn.Module):
    def __init__(self, hidden_size, kernel, stride, nb_filter, input_size):
        super(conv_lstm, self).__init__()
        self.conv = nn.Sequential(nn.Conv1d(in_channels=input_size, out_channels=nb_filter,
                                               kernel_size=kernel, stride=stride),
                                  nn.ReLU(),
                                  nn.BatchNorm1d(nb_filter))
        self.lstm = create_model(model='skip_lstm',input_size=nb_filter,hidden_size=hidden_size,num_layers=1)
        self.hidden_size = hidden_size

    def forward(self, input):
        input = self.conv(input.permute(0, 2, 1))
        input = input.permute(0, 2, 1)
        output = self.lstm(input)
        output, hx, updated_state = split_rnn_outputs('skip_lstm', output)
        return output[:, -1, :]



class Scoring(nn.Module):
    def __init__(self, map_size,im_info):
        super(Scoring, self).__init__()
        self.im_info=im_info
        self.feat_size=map_size

        conv_input = 128
        self.conv = nn.Sequential(nn.Conv1d(in_channels=self.feat_size, out_channels=conv_input,kernel_size=1,stride=1),
                                  nn.ReLU(),
                                  nn.BatchNorm1d(conv_input))
        hidden_size = 256
        self.scale1 = conv_lstm(hidden_size, kernel=2, stride=1, nb_filter=256, input_size=self.feat_size)
        self.scale2 = conv_lstm(hidden_size, kernel=4, stride=2, nb_filter=256, input_size=self.feat_size)
        self.scale3 = conv_lstm(hidden_size, kernel=8, stride=4, nb_filter=256, input_size=self.feat_size)
        self.linear_skip1 = nn.Linear(hidden_size, 64)
        self.linear_skip2 = nn.Linear(hidden_size, 64)
        self.linear_skip3 = nn.Linear(hidden_size, 64)

        self.attn = selfAttn(conv_input, 64, 50)
        self.lstm = nn.LSTM(input_size=conv_input, hidden_size=hidden_size, num_layers=1, batch_first=True)

        self.linear_attn = nn.Linear(hidden_size, 64)
        self.linear_merge = nn.Linear(64*4, 64)

        self.reg = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.6)

        # build rpn model
        self.faster_RCNN=faster_rcnn._fasterRCNN()
        self.faster_RCNN.create_architecture()


    def forward(self, base_feat,central_pos,gt_boxes):
        # loss for central frames, and rois for all images
        rcnn_output, rpn_loss = self.faster_RCNN(base_feat,central_pos,
                                                 self.im_info, gt_boxes)  # rcnn_output.shape=(b_size,300,1024)

        model_input = rcnn_output.permute(0, 2, 1)
        model_input = self.conv.forward(model_input).permute(0, 2, 1)   # model_input.shape=(b_size,300,128)
        attn_output, atten_matrix = self.attn.forward(model_input)  # attn_output.shape=(b_size,50,128)
        attn, _ = self.lstm(attn_output)   # How to improve this attention model for a better interpretability?
        attn = attn[:, -1, :]


        M_output=  torch.cat([self.relu(self.linear_skip1(self.scale1(rcnn_output))),
                              self.relu(self.linear_skip2(self.scale2(rcnn_output))),
                              self.relu(self.linear_skip3(self.scale3(rcnn_output)))], 1)

        output = torch.cat([M_output, self.relu(self.linear_attn(attn))], 1)
        output=self.relu(self.linear_merge(output))

        return self.reg(output),rpn_loss

    def loss(self, regression, actuals):
        regr_loss_fn = nn.MSELoss()
        return regr_loss_fn(regression, actuals)


if __name__ == '__main__':
    model=Scoring(map_size=1024,c3d_size=4096,im_info=28)
    base_feat=torch.rand((2,300,512,4,4))
    gt_boxes=torch.rand((2,4))
    result,penal,rpn_loss=model(base_feat,gt_boxes)
    print result.shape,penal,rpn_loss