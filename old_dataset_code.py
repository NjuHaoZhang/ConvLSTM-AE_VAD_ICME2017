# 转发服务器上的8000端口数据到本地的8001端口： ssh -L 8001:127.0.0.1:8000 zh@10.21.25.237
# 本地访问：127.0.0.1:8001
# tensorboard --logdir="./summary" --port=8000
#
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#
# from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
from torchvision import transforms, utils
#
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from collections import OrderedDict
import os
import glob
import cv2 # pip install opencv-python

rng = np.random.RandomState(2017)


def np_load_frame(filename, resize_height, resize_width):
    """
    Load image path and convert it to numpy.ndarray. Notes that the color channels are BGR and the color space
    is normalized from [0, 255] to [-1, 1].

    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """
    image_decoded = cv2.imread(filename)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized

import math
class MyDataset(data.Dataset):
    def __init__(self, video_folder, time_steps=5, num_pred=0, resize_height=256, resize_width=256):
        # num_pred: 用于向后预测几帧
        #
        self.dir = video_folder # 一个dataset比如avenue/frames_dir
        self.videos = OrderedDict() # aveneue目录下有多个子目录，子目录下是xxx.jpg
        self._resize_height = resize_height
        self._resize_width = resize_width
        self.setup()
        #
        video_info_list = list(self.videos.values())
        self._num_videos = len(video_info_list)
        self._clip_length = time_steps + num_pred
        self._clip_num = [] # 统计每个子目录中有多少个clip，方便确定index
        for vid in range(self._num_videos): # 遍历每一个子目录（目录下有jpg）
            # print("_num_videos: ", self._num_videos)
            video_info = video_info_list[vid]  # 当前某个子目录，包含很多jpg
            # print("video_info: ", video_info)
            # print("vid: ", vid)
            num = video_info['length'] - self._clip_length + 1 # 充分利用每一帧构成clip
            (self._clip_num).append(num)
        #
        self._cnt_cn = []  # 记录每个子目录的 index(上界 + 1)
        iterNum = self._num_videos  # 当前这个dataset(比如avenue)有多少个子目录(i.e.16)
        for idx in range(iterNum):
            if idx == 0:
                num = self._clip_num[idx]
                self._cnt_cn.append(num)
            else:
                num = self._cnt_cn[idx - 1] + self._clip_num[idx]
                self._cnt_cn.append(num)

    def __getitem__(self, index):#返回的是tensor(一个video clip)
        # clip = self.clips[index]
        # return clip
        vid, start, end = self.get_vid_start_end(index)
        clip = self.get_video_clips(vid, start, end)
        # 再来组织model需要的X,Y
        X = clip
        Y = self.get_Y_clips(vid, start, end)
        return X,Y

    def __len__(self):
        return sum(self._clip_num) # 统计一共有多少个clip

    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video
            # ['frame'] 也是存放 frames的path，直到需要才读入内存
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
            self.videos[video_name]['frame'].sort()
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame'])

    def get_vid_start_end(self, index):
        iterNum = self._num_videos # 当前这个dataset(比如avenue)有多少个子目录(i.e.16)
        for idx in range(iterNum): # TODO 这个循环查找可以通过一个 hashmap来加速（空间换时间）
            if index < self._cnt_cn[idx]: # 不能等于，因为Index从 0 计数
                break
        if idx == 0:
            delta_clip = index
        else:
            delta_clip = index - self._cnt_cn[idx-1] # 在 idx 这个子目录里面处于第几个clip, 注意结果是index
        # start = delta_clip * self._clip_length # 后来经过debug发现这个错了！！！
        start = delta_clip  # clip index 与 frame index的对应关系是：相等，因为我是overlap-wise的从连续帧抽取尽可能多的clip
        end = start + self._clip_length
        vid_names = list(self.videos) # 获取所有子目录的name
        return vid_names[idx], start, end # 注意其实是：[start, end)

    def get_video_clips(self, video, start, end):
        # assert video in self.videos, 'video = {} must in {}!'.format(video, self.videos.keys())
        # assert start >= 0, 'start = {} must >=0!'.format(start)
        # assert end <= self.videos[video]['length'], 'end = {} must <= {}'.format(video, self.videos[video]['length'])

        batch = []
        for i in range(start, end): # 注意其实是：[start,end)
            # print("video, start, end: ", video, start, end) # 0,0,5
            # print("self.videos[video]['frame'][i]: ", self.videos)
            # print("self.videos[video]['frame'][i]: ", self.videos[video]['frame'][i])
            image = np_load_frame(self.videos[video]['frame'][i], self._resize_height, self._resize_width)
            batch.append(image)
        # print("image_shape: ", image.shape) # (256,256,3)
        # print("batch_shape: ", np.array(batch).shape) # (5, 256, 256, 3)
        return np.concatenate(batch, axis=2) # axis=0,1,2分别表示依据第0,1,2维来拼接数组

    def get_Y_clips(self, video, start, end):
        # assert video in self.videos, 'video = {} must in {}!'.format(video, self.videos.keys())
        # assert start >= 0, 'start = {} must >=0!'.format(start)
        # assert end <= self.videos[video]['length'], 'end = {} must <= {}'.format(video, self.videos[video]['length'])

        batch = []
        for i in range(start, end):
            if i == start:
                image = np_load_frame(self.videos[video]['frame'][i],
                                  self._resize_height, self._resize_width)
                batch.append(image)
            else:
                # 先重构过去一帧
                image_pre = np_load_frame(self.videos[video]['frame'][i - 1],
                                          self._resize_height, self._resize_width)
                batch.append(image_pre)
                # 再重构当前帧
                image = np_load_frame(self.videos[video]['frame'][i],
                                      self._resize_height, self._resize_width)
                batch.append(image)
        return np.concatenate(batch, axis=2) #

# kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
# train_loader = torch.utils.data.DataLoader(
#     MyDataset(images, labels), batch_size=args.batch_size, shuffle=True, **kwargs)

# 下面是测试子程序
def test_1():
    video_folder = "/home/zh/Papers_Code/ano_pred_cvpr2018_sist/ano_pred_cvpr2018/" \
                   "Data/avenue/training/frames"
    mds = MyDataset(video_folder, time_steps=5, num_pred=0,
                    resize_height=256, resize_width=256)
    # print(len(mds))
    # print("keys: ", mds.videos.keys())
    print("mds[0]: ", mds[0]) # 第一个clip

def test___init__():
    video_folder = "/home/zh/Papers_Code/ano_pred_cvpr2018_sist/ano_pred_cvpr2018/" \
                   "Data/avenue/training/frames"
    mds = MyDataset(video_folder, time_steps=5, num_pred=0,
                    resize_height=256, resize_width=256)
    video_info_list = list(mds.videos.values()) # 获得所有子目录对应的dict
    print("video_info_list: ", video_info_list)
    vid_names = list(mds.videos)  # 获取所有子目录的name
    print("vid_names: ", vid_names)
    v01,v16 = vid_names[0], vid_names[-1]
    frame_num_v01, frame_num_v16 = mds.videos[v01]['length'], mds.videos[v16]['length']
    print("frame_num_v01, frame_num_v16: ", frame_num_v01, frame_num_v16)
    # 01有 1364 帧， 16有 244 帧
    assert frame_num_v01 == 1364,"frame_num_v01出错"
    assert frame_num_v16 == 244, "frame_num_v16出错"
    #
    clip_num = mds._clip_num
    c01, c16 = clip_num[0], clip_num[-1]
    print("c01, c16: ", c01, c16)
    # c01 = 1364 - 5 + 1 = 1360, c16 = 244 - 5 + 1 = 240
    assert c01 == 1360, " c01 出错"
    assert c16 == 240, " c16 出错"
    #
    cnt_cn = mds._cnt_cn
    print("clip_num: ", clip_num)
    print("cnt_cn: ", cnt_cn)

def test_get_vid_start_end():
    # 测试第一个子目录：aveneue/frames/01: 1364帧，5秒的时间窗口，所以是 1364-5+1=1362个clip
    #
    video_folder = "/home/zh/Papers_Code/ano_pred_cvpr2018_sist/ano_pred_cvpr2018/" \
                   "Data/avenue/training/frames"
    mds = MyDataset(video_folder, time_steps=5, num_pred=0,
                    resize_height=256, resize_width=256)
    cnt_cn = mds._cnt_cn
    clip_num = mds._clip_num
    print("clip_num: ", clip_num)
    print("cnt_cn: ", cnt_cn)
    vid_name, start, end = mds.get_vid_start_end(0) #
    # assertCountEqual([vid_name, start, end], ["01", 0, 5]), TODO 调通
    print("vid_name, start, end if index == 0: ", vid_name, start, end)
    vid_name, start, end = mds.get_vid_start_end(1)  #
    # assert [vid_name, start, end] == ["01", 5, 10], "index=1 error"
    print("vid_name, start, end if index == 1: ", vid_name, start, end)
    vid_name, start, end = mds.get_vid_start_end(1360)  #
    # assert [vid_name, start, end] == ["02", 0, 5], "index=1360 error"
    print("vid_name, start, end if index == 1360: ", vid_name, start, end)
    vid_name, start, end = mds.get_vid_start_end(8174)  #
    # assert [vid_name, start, end] == ["06", 1502, 1507], "index=8174 error"
    print("vid_name, start, end if index == 8174: ", vid_name, start, end)
    # 再测试几组，TODO
    vid_name, start, end = mds.get_vid_start_end(15263)  # 最后一个clip
    # assert [vid_name, start, end] == ["16", 1195, 1200], "index=15263 error"
    print("vid_name, start, end if index == 15263: ", vid_name, start, end)

def test___getitem__():
    from PIL import Image

    # 测试取，T=1的(X,Y); T=6的pair TODO
    video_folder = "/home/zh/Papers_Code/ano_pred_cvpr2018_sist/ano_pred_cvpr2018/" \
                   "Data/avenue/training/frames"
    mds = MyDataset(video_folder, time_steps=5, num_pred=0,
                    resize_height=256, resize_width=256)
    T =1 # 也就是 clip 的 index
    X_1, Y_1 = mds[T] # 开始的5帧
    print("shape of X_1, Y_1", X_1.shape, Y_1.shape) #(256, 256, 15) (256, 256, 27)
    # X_1 = Image.fromarray()
    # print("X_1: ", X_1)
    # Y_1 = Image.fromarray(Y_1[:][:][0:3])
    # X_1.save("X_1.png")
    # Y_1.save(Y_1.png)

def test_batchLoader():
    pass # 加载器方式访问X,Y

# TODO: 除了如同上面这样组织clips(根据clip index 返回 frame-level's start and end)
# 还可以直接rand: 只要有足够的随机性，理论上每次rand 出来的 clip都不相同
# 所以这种方式方式下的 __getitem__的index就不具有先后关系约束了，而是作为随机数种子？
# 每次__getitem__(index)，通过 rand 得到一对(vid_name, start, end)得到一个clip
# TODO，这个好像pytorch有支持：torch.utils.data.RandomSampler（但必须控制顺序采样）
# torch.utils.data.BatchSampler这个也可以，但是不支持overlap式的采样，只能
# 把源data分成若干个non-overlap mini-batch

if __name__ == '__main__':
    # test_1()
    # test___init__()
    # test_get_vid_start_end()
    test___getitem__()