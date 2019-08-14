from dataloader import videoDataset
from model_1 import Scoring
import torch
import torch.utils.data as data
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
from scipy.stats import spearmanr as sr
import matplotlib.pyplot as plt
import os
import time


# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

train_loss=[]
test_loss=[]
epoch_num=50
def train_shuffle(min_mse=200, max_corr=0):
    trainset = videoDataset(fea_map_root="figure_skating/central_frames_1",c3d_root="figure_skating/c3d_new",
                            label="figure_skating/train_dataset.txt", suffix=".npy", train=True,PCS=False)
    trainLoader = torch.utils.data.DataLoader(trainset,batch_size=32, shuffle=True, num_workers=0)
    testset = videoDataset(fea_map_root="figure_skating/central_frames_1",c3d_root="figure_skating/c3d_new",
                           label="figure_skating/test_dataset.txt", suffix='.npy', train=False,PCS=False)
    testLoader = torch.utils.data.DataLoader(testset,batch_size=32, shuffle=False, num_workers=0)

    # build the model
    scoring = Scoring(map_size=1024,im_info=10)
    if torch.cuda.is_available():
        scoring.cuda()                # turn the model into gpu
    for para in scoring.named_parameters():
        print para[0],para[1].shape

    total_params = sum(p.numel() for p in scoring.parameters() if p.requires_grad)
    optimizer = optim.SGD(params=scoring.parameters(), lr=1.5e-3,weight_decay=0.01)
    scheduler=lr_scheduler.ExponentialLR(optimizer,gamma=0.98)

    for epoch in range(epoch_num):
        print("Epoch:  " + str(epoch) + "; Total Params: %d" % total_params)
        total_regr_loss = 0
        total_sample = 0
        for i, (feature_map,central_pos, scores, boxes) in enumerate(trainLoader):
            start_time = time.time()
            if torch.cuda.is_available():
                scores = scores.float().view(-1, 1)
                scores = Variable(scores).cuda()
                feature_map=Variable(feature_map).cuda()
                central_pos=Variable(central_pos).cuda()
                boxes=Variable(boxes).cuda()
                # c3d_feat=Variable(c3d_feat).cuda()

            logits, rpn_loss= scoring(feature_map,central_pos,boxes)   # features.shape=(batch_size,512,4,4)
            mse=scoring.loss(logits, scores)
            print("mse={:.3f}  rpn_loss={}".format(mse,rpn_loss))
            regr_loss = mse + rpn_loss

            # back propagation
            optimizer.zero_grad()        # PyTorch accumulates the gradients, so clean it every step of backward.
            regr_loss.backward()
            optimizer.step()
            total_regr_loss += regr_loss.data.item() * scores.shape[0]
            total_sample += scores.shape[0]

            end_time=time.time()
            time_spent=end_time-start_time
            print("{}/13 train-batch have done, time spent {:.2f}s".format(i+1,time_spent))
            print("*********************************")

        loss=total_regr_loss / total_sample
        train_loss.append(loss)
        print("Train Loss:{:.3f}" .format(loss))

        ### the rest is used to evaluate the model with the test dataset ###
        scoring.eval()   # turn the model into evaluation mode(for batch-normalization / dropout layer)
        val_pred = []
        val_sample = 0
        val_loss = 0
        val_truth = []
        for j, (feature_map,central_pos,scores, boxes) in enumerate(testLoader):
            val_truth.append(scores.numpy())
            if torch.cuda.is_available():
                scores = Variable(scores).cuda()
                scores = scores.float().view(-1, 1)
                feature_map = Variable(feature_map).cuda()
                central_pos = Variable(central_pos).cuda()
                boxes = Variable(boxes).cuda()
                # c3d_feat = Variable(c3d_feat).cuda()

            regression,_,= scoring(feature_map,central_pos,boxes)
            val_pred.append(regression.data.cpu().numpy())
            regr_loss = scoring.loss(regression, scores)
            val_loss += (regr_loss.data.item()) * scores.shape[0]
            val_sample += scores.shape[0]
            print("{}/4 test-batches have done".format(j+1))

        val_truth = np.concatenate(val_truth)
        val_pred = np.concatenate(val_pred)
        val_sr, _= sr(val_truth, val_pred)
        if val_loss / val_sample < min_mse:
            torch.save(scoring.state_dict(), 'faster_rcnn.pt')
        min_mse = min(min_mse, val_loss / val_sample)
        max_corr = max(max_corr, val_sr)
        loss=val_loss / val_sample
        test_loss.append(loss)
        print("Val Loss: {:.2f} Correlation: {:.2f} Min Val Loss: {:.2f} Max Correlation: {:.2f}"
              .format(loss, val_sr, min_mse, max_corr))
        print '\n'

        scoring.train()                    # turn back to train mode
        scheduler.step()



min_mse = 200
max_corr = 0
curr_time=time.asctime(time.localtime(time.time()))
print("Running start at",curr_time)
train_shuffle(min_mse, max_corr)

print test_loss
# plot
plt.title("New model: PCS")
plt.subplot(211)
plt.plot(range(0,epoch_num),train_loss,'r-')
plt.legend('train_loss')
plt.subplot(212)
plt.plot(range(0,epoch_num),test_loss,'g-')
plt.legend('test_loss')

plt.savefig("01.png")