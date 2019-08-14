import os
import torch
import torch.utils.data as data
import numpy as np
import random


def default_loader(path):
    feature = np.load(path)
    return feature


def fea_transform(fea, max_frames):
    num_frames = fea.shape[0]
    fea=np.reshape(fea,(-1,512,4,4))
    if num_frames >= max_frames:
        start_idx = random.choice(range(num_frames - max_frames))
        new_fea = fea[start_idx:start_idx + max_frames, :]
        central_pos=min(max_frames-1,max((num_frames>>1)-start_idx,0))   # in case num_frames>2 * max_frames
    else:
        raise ValueError("max_frames={} is too large.".format(max_frames))
    return new_fea,central_pos

def c3d_transform(c3d_feat,max_frames):
    num_frames = c3d_feat.shape[0]
    if num_frames >= max_frames:
        start_idx = random.choice(range(num_frames - max_frames))
        new_fea = c3d_feat[start_idx:start_idx + max_frames, :]
    else:
        raise ValueError("max_frames={} is too large.".format(max_frames))
    return new_fea

class videoDataset(data.Dataset):
    def __init__(self, fea_map_root,c3d_root, label, target_transform=None,
                 suffix=".binary", loader=default_loader, data=None, PCS=True,train=True):
        self.boxes=[]
        if data is not None:
            videos = data
        else:
            fh = open(label)
            videos = []
            if train:
                for line in fh.readlines():
                    video_id, tes, pcs, failure,gt_box = line.strip().split(' ')
                    positions = gt_box[1:-1].split(',')
                    gt_box = torch.Tensor((int(positions[0]), int(positions[1]), int(positions[2]), int(positions[3])))
                    self.boxes.append(gt_box)
                    if PCS:
                        pcs = float(pcs)
                        videos.append((video_id, pcs))
                    else:
                        tes = float(tes)
                        videos.append((video_id, tes))
            else:
                for line in fh.readlines():
                    video_id, tes, pcs, failure= line.strip().split(' ')
                    if PCS:
                        pcs = float(pcs)
                        videos.append((video_id, pcs))
                    else:
                        tes = float(tes)
                        videos.append((video_id, tes))
        self.fea_map_root=fea_map_root
        self.c3d_root=c3d_root
        self.videos = videos
        self.fea_transform = lambda x:fea_transform(x, 300)
        # self.c3d_transform=lambda x: c3d_transform(x,300)
        self.target_transform = target_transform
        self.loader = loader
        self.suffix = suffix
        self.train=train

    def __getitem__(self, index):
        video_id, score = self.videos[index]
        if not video_id.endswith(self.suffix):
            base_name = video_id + "_central"+self.suffix
            # c3d_name=video_id+self.suffix
        fea_map=self.loader(os.path.join(self.fea_map_root, base_name))
        # c3d_feat=self.loader(os.path.join(self.c3d_root, c3d_name))

        fea_map ,central_pos= self.fea_transform(fea_map)   # fea_map.shape=(num_frames,512,4,4),central_feat=(512,4,4)
        # c3d_feat= self.c3d_transform(c3d_feat)  # c3d_feat.shape=(num_frames,4096)

        if self.train:
            box=self.boxes[index]
        else:
            box=torch.rand((4))    # In test mode, box is of no use. So we just randomly pick one for simplicity.

        return fea_map,central_pos,score, box


    def __len__(self):
        return len(self.videos)
