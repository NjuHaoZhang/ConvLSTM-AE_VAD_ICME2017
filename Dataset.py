# 转发服务器上的8000端口数据到本地的8001端口： ssh -L 8001:127.0.0.1:8000 zh@10.21.25.237
# 本地访问：127.0.0.1:8001
# tensorboard --logdir="./summary" --port=8000
#
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

# 这里暂时要手动维护， for mean of frames
from config import Config_ped2_testing as Config

# 这个是：做归一化的 data augmentation
# def np_load_frame(filename, resize_height, resize_width, channel=3):
#     """
#     Load image path and convert it to numpy.ndarray.
#     Notes that the color channels are BGR and
#     the color space is normalized from [0, 225] to [-1, 1].
#
#     :param filename: the full path of image
#     :param resize_height: resized height
#     :param resize_width: resized width
#     :return: numpy.ndarray
#     """
#
#     image_decoded = cv2.imread(filename)
#     # print("image: ", image_decoded.shape)
#     if channel == 1: # 转为 灰度图 并reshape
#         image_decoded = cv2.cvtColor(image_decoded, cv2.COLOR_RGB2GRAY) # 转为 gray image
#         # print("image: ", image_decoded.shape)
#     image_resized = cv2.resize(image_decoded, (resize_width, resize_height)) # resize
#     image_resized = np.reshape(image_resized, (resize_width, resize_height, channel)) # reshape
#     image_resized = image_resized.astype(dtype=np.float32)
#     image_resized = (image_resized / 127.5) - 1.0 # for [0,255]
#     # image_resized = (image_resized / 112) - 1.0 ## 标准化 [-1,1} # 这个理解错了
#     return image_resized


# 这个的 data augmentation涉及的，submean, resize,
# 这个版本暂时，直接使用 wxl 做好的 mean.npy, 更全面的计算 mean,submean函数版本， TODO
def np_load_frame(filename, resize_height, resize_width, channel=1):
    """
    Load image path and convert it to numpy.ndarray.
    Notes that the color channels are BGR and
    the color space is normalized from [0, 225] to [-1, 1].

    :param filename: the full path of image
    :param resize_height: resized height
    :param resize_width: resized width
    :return: numpy.ndarray
    """

    image_decoded = cv2.imread(filename)
    # print("image: ", image_decoded.shape) # pixel value in [0,255]
    if channel == 1: # 转为 灰度图 并reshape
        image_decoded = cv2.cvtColor(image_decoded, cv2.COLOR_RGB2GRAY) # 转为 gray image
        # print("image: ", image_decoded.shape)
    image_resized = cv2.resize(image_decoded, (resize_width, resize_height)) # resize
    image_resized = np.reshape(image_resized, (resize_width, resize_height, channel)) # reshape
    image_resized = image_resized.astype(dtype=np.float32)
    # image_resized = (image_resized / 127.5) - 1.0 # for [256,256]
    # 归一化 转化为 标准化 ， (x_i - x_mean) / (x_max - x_min)
    mean = get_mean(Config.dataset_name)
    # print("mean: ", mean) # pixel value in [0,255]
    # print("image_resized-befor sub mean: ", image_resized)
    image_resized = (image_resized - mean) / 255 # 去中心化 {好像有去背景的效果},;除以255 [0,255]转为 [0,1]
    # print("image_resized: ", image_resized)
    return image_resized

def get_mean(dataset_name):
    filePath = os.path.join('data', dataset_name + "_mean_225_gray.npy")
    # print("fileName: ", filePath)
    res = np.load(filePath)
    # print("shape of res: ", res.shape)
    return res


import math
class MyDataset(data.Dataset):
    def __init__(self, video_folder, time_steps=5, num_pred=0, resize_height=256, resize_width=256, channel=3):
        # num_pred: 用于向后预测几帧
        #
        self.dir = video_folder # 一个dataset比如avenue/frames_dir
        self.videos = OrderedDict() # aveneue目录下有多个子目录，子目录下是xxx.jpg
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._channel = channel # 默认是3
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
        start = delta_clip # clip index 与 frame index的对应关系是：相等，因为我是overlap-wise的从连续帧抽取尽可能多的clip
        end = start + self._clip_length
        vid_names = list(self.videos) # 获取所有子目录的name
        return vid_names[idx], start, end # 注意其实是：[start, end)

    def get_video_clips(self, video, start, end):
        # assert video in self.videos, 'video = {} must in {}!'.format(video, self.videos.keys())
        # assert start >= 0, 'start = {} must >=0!'.format(start)
        # assert end <= self.videos[video]['length'], 'end = {} must <= {}'.format(video, self.videos[video]['length'])

        batch = []
        for idx in range(start, end): # 注意其实是：[start,end)
            # print("video, start, end: ", video, start, end) # 0,0,5
            # print("self.videos[video]['frame'][i]: ", self.videos)
            # print("self.videos[video]['frame'][i]: ", self.videos[video]['frame'][i])
            image = np_load_frame(self.videos[video]['frame'][idx], self._resize_height, self._resize_width,
                                  channel=self._channel)
            batch.append(image)
        # print("image_shape: ", image.shape) # (256,256,3)
        # print("batch_shape: ", np.array(batch).shape) # (5, 256, 256, 3)
        # return np.concatenate(batch, axis=0) # axis=0,1,2分别表示依据第0,1,2维来拼接数组
        return np.array(batch)

    def get_Y_clips(self, video, start, end):
        # assert video in self.videos, 'video = {} must in {}!'.format(video, self.videos.keys())
        # assert start >= 0, 'start = {} must >=0!'.format(start)
        # assert end <= self.videos[video]['length'], 'end = {} must <= {}'.format(video, self.videos[video]['length'])

        batch = []
        for idx in range(start, end):
            if idx == start:
                image = np_load_frame(self.videos[video]['frame'][idx],
                                  self._resize_height, self._resize_width,
                                      channel=self._channel)
                batch.append(image)
            else:
                # 先重构过去一帧
                image_pre = np_load_frame(self.videos[video]['frame'][idx - 1],
                                          self._resize_height, self._resize_width,
                                          channel=self._channel)
                batch.append(image_pre)
                # 再重构当前帧
                image = np_load_frame(self.videos[video]['frame'][idx],
                                      self._resize_height, self._resize_width,
                                      channel=self._channel)
                batch.append(image)
        # return np.concatenate(batch, axis=2) #
        return np.array(batch)

# 因为 tesing mode 与 Training 不同，所以需要为 Testing 建一个 Dataset
# Tesing 要求 扫描每帧有且仅有 1 次，以后重构代码，再把两个Dataset想办法合并
class Dataset_Testing(data.Dataset):
    def __init__(self, video_folder, time_steps=5, num_pred=0, resize_height=256, resize_width=256, channel=3):
        # num_pred: 用于向后预测几帧
        #
        self.dir = video_folder # 一个dataset比如avenue/frames_dir
        self.videos = OrderedDict() # aveneue目录下有多个子目录，子目录下是xxx.jpg
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._channel = channel
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
            num = math.ceil(video_info['length'] / self._clip_length) # 2.5->3
            # 最后剩余几帧单独组成一个变长(不是T帧)的seq
            # 每个clip 都是 non-voerlap，与Training 中 (overlap充分利用每一帧)不同
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
        vid_names = list(self.videos)  # 获取所有子目录的name
        #
        iterNum = self._num_videos # 当前这个dataset(比如avenue)有多少个子目录(i.e.16)
        for idx in range(iterNum): # TODO 这个循环查找可以通过一个 hashmap来加速（空间换时间）
            if index < self._cnt_cn[idx]: # 不能等于，因为Index从 0 计数
                break
        if idx == 0:
            delta_clip = index
        else:
            delta_clip = index - self._cnt_cn[idx-1] # 在 idx 这个子目录里面处于第几个clip, 注意结果是index
        # start of clip-index 和 start of frame-index的对应关系不是 相等，
        start = delta_clip * self._clip_length # 画图可得：0->len->2*len->3*len
        end = start + self._clip_length
        # 处理异常：end 超出 vid_names[idx] 这个子目录的帧数目
        len_vid = self.videos[vid_names[idx]]['length']
        if end > len_vid: # 牢牢记住是：[start,end)，所以 end本来就取不到
            end = len_vid # 所以 end==len_vid是合理的下标
            # 这个异常处理的意义是：处理之前整除之后,落单的若干帧
            # 组成一个特殊的seq, seq_len 不是 T=10， 是一个小于10的数字
            # 只要 batch_size == 1 就可以处理 seq_len 变长的输入序列
        return vid_names[idx], start, end # 注意其实是：[start, end)

    def get_video_clips(self, video, start, end):
        # assert video in self.videos, 'video = {} must in {}!'.format(video, self.videos.keys())
        # assert start >= 0, 'start = {} must >=0!'.format(start)
        # assert end <= self.videos[video]['length'], 'end = {} must <= {}'.format(video, self.videos[video]['length'])

        batch = []
        for idx in range(start, end): # 注意其实是：[start,end)
            # print("video, start, end: ", video, start, end) # 0,0,5
            # print("self.videos[video]['frame'][i]: ", self.videos)
            # print("self.videos[video]['frame'][i]: ", self.videos[video]['frame'][i])
            image = np_load_frame(self.videos[video]['frame'][idx], self._resize_height, self._resize_width,
                                  channel=self._channel)
            batch.append(image)
        # print("image_shape: ", image.shape) # (256,256,3)
        # print("batch_shape: ", np.array(batch).shape) # (5, 256, 256, 3)
        # return np.concatenate(batch, axis=0) # axis=0,1,2分别表示依据第0,1,2维来拼接数组
        return np.array(batch)

    def get_Y_clips(self, video, start, end):
        # assert video in self.videos, 'video = {} must in {}!'.format(video, self.videos.keys())
        # assert start >= 0, 'start = {} must >=0!'.format(start)
        # assert end <= self.videos[video]['length'], 'end = {} must <= {}'.format(video, self.videos[video]['length'])

        batch = []
        for idx in range(start, end):
            if idx == start:
                image = np_load_frame(self.videos[video]['frame'][idx],
                                  self._resize_height, self._resize_width, channel=self._channel)
                batch.append(image)
            else:
                # 先重构过去一帧
                image_pre = np_load_frame(self.videos[video]['frame'][idx - 1],
                                          self._resize_height, self._resize_width, channel=self._channel)
                batch.append(image_pre)
                # 再重构当前帧
                image = np_load_frame(self.videos[video]['frame'][idx],
                                      self._resize_height, self._resize_width, channel=self._channel)
                batch.append(image)
        # return np.concatenate(batch, axis=2) #
        return np.array(batch)


###################################################################################

# 下面是 TODO 部分
# def get_dataloader(opt):
#     myDataset = MyDataset(video_folder, time_steps=TIME_STEPS, num_pred=NUM_PRED,
#                           resize_height=new_weight, resize_width=new_height)
#     # TODO: data augmentation
#     dataloader = DataLoader(myDataset, batch_size=BATCH_SIZE, shuffle=True,
#                             num_workers=num_workers)
#     dataloader = data.DataLoader(dataset,
#                                  batch_size=opt.batch_size,
#                                  shuffle=opt.shuffle,
#                                  num_workers=opt.num_workers,
#                                  collate_fn=create_collate_fn(dataset.padding, dataset.end))
#     return dataloader


##################################################################################

# 下面是测试子程序：for Dataset_training
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
    print("vid_name, start, end if index == 8174: ", vid_name, start, end)
    # 再测试几组，TODO
    vid_name, start, end = mds.get_vid_start_end(15263)  # 最后一个clip
    print("vid_name, start, end if index == 15263: ", vid_name, start, end)
    vid_name, start, end = mds.get_vid_start_end(3970)  #
    print("vid_name, start, end if index == 3970: ", vid_name, start, end)

def test___getitem__():
    from PIL import Image

    # 测试取，T=1的(X,Y); T=6的pair TODO
    video_folder = "/home/zh/Papers_Code/ano_pred_cvpr2018_sist/ano_pred_cvpr2018/" \
                   "Data/avenue/training/frames"
    mds = MyDataset(video_folder, time_steps=5, num_pred=0,
                    resize_height=256, resize_width=256)
    idx = 3970 # 也1,2,3,4,5 就是 clip 的 index
    X_1, Y_1 = mds[idx] # 开始的5帧
    print("shape of X_1, Y_1", X_1.shape, Y_1.shape)

def test_batchLoader():
    video_folder = "/home/zh/Papers_Code/ano_pred_cvpr2018_sist/ano_pred_cvpr2018/" \
                   "Data/avenue/training/frames"
    mds = MyDataset(video_folder, time_steps=5, num_pred=0,
                    resize_height=256, resize_width=256)

    dataloader = DataLoader(mds, batch_size=4,
                            shuffle=True, num_workers=4)
    for i_batch, (X,Y) in enumerate(dataloader):
        print("i_batch: ", i_batch)
        print("the shape of X,Y: ", X.shape, Y.shape)
        if i_batch == 3:
            break

################################################################################
# for Dataset_Testing

def test_get_vid_start_end_Dataset_Testting():
    # 测试第一个子目录：aveneue/frames/01: 1364帧，5秒的时间窗口，所以是 1364-5+1=1362个clip
    #
    video_folder = "/home/zh/Papers_Code/ano_pred_cvpr2018_sist/ano_pred_cvpr2018/" \
                   "Data/avenue/testing/frames"
    mds = Dataset_Testing(video_folder, time_steps=10, num_pred=0,
                    resize_height=225, resize_width=225)
    cnt_cn = mds._cnt_cn
    clip_num = mds._clip_num
    print("clip_num: ", clip_num)
    print("cnt_cn: ", cnt_cn)
    vid_name, start, end = mds.get_vid_start_end(265)  #
    print("vid_name, start, end if index == 265: ", vid_name, start, end)
    vid_name, start, end = mds.get_vid_start_end(951)  #
    print("vid_name, start, end if index == 295: ", vid_name, start, end)
    X = mds.get_video_clips(vid_name, start, end)
    print("X shape: ", X.shape)
    vid_name, start, end = mds.get_vid_start_end(1127)  #
    print("vid_name, start, end if index == 1127: ", vid_name, start, end)
    # vid_name, start, end = mds.get_vid_start_end(0) #
    # # assertCountEqual([vid_name, start, end], ["01", 0, 5]), TODO 调通
    # print("vid_name, start, end if index == 0: ", vid_name, start, end)
    # vid_name, start, end = mds.get_vid_start_end(1)  #
    # # assert [vid_name, start, end] == ["01", 5, 10], "index=1 error"
    # print("vid_name, start, end if index == 1: ", vid_name, start, end)
    # vid_name, start, end = mds.get_vid_start_end(1360)  #
    # # assert [vid_name, start, end] == ["02", 0, 5], "index=1360 error"
    # print("vid_name, start, end if index == 1360: ", vid_name, start, end)
    # vid_name, start, end = mds.get_vid_start_end(8174)  #
    # print("vid_name, start, end if index == 8174: ", vid_name, start, end)
    # # 再测试几组，TODO
    # vid_name, start, end = mds.get_vid_start_end(15263)  # 最后一个clip
    # print("vid_name, start, end if index == 15263: ", vid_name, start, end)
    # vid_name, start, end = mds.get_vid_start_end(3970)  #
    # print("vid_name, start, end if index == 3970: ", vid_name, start, end)

def test_Dataset_Testing():
    video_folder = "/home/zh/Papers_Code/ano_pred_cvpr2018_sist/" \
                   "ano_pred_cvpr2018/" \
                   "Data/avenue/testing/frames"
    mds = Dataset_Testing(video_folder, time_steps=10, num_pred=0,
                    resize_height=225, resize_width=225)

    # 1 就能支持变长的 seq_len, 否则在拼接成大矩阵时，会报 seq_len 这个 维度不一致的错误
    dataloader = DataLoader(mds, batch_size=1,
                            shuffle=False, num_workers=4)
    for i_batch, (X,Y) in enumerate(dataloader):
        print("i_batch: ", i_batch)
        print("the shape of X,Y: ", X.shape, Y.shape)
        # if i_batch == 3:
        #     break

def test_total_frames_len():
    video_folder = "/home/zh/Papers_Code/ano_pred_cvpr2018_sist/" \
                   "ano_pred_cvpr2018/" \
                   "Data/avenue/testing/frames"
    mds = Dataset_Testing(video_folder, time_steps=10, num_pred=0,
                          resize_height=225, resize_width=225)
    # 统计帧数
    vid_names = list(mds.videos)  # 获取所有子目录的name
    cnt = 0
    for vn in vid_names:
        cnt += mds.videos[vn]['length']
    print("total frames: ", cnt) # 15324

def test_Dataset_Testing_Y_lens():
    video_folder = "/home/zh/Papers_Code/ano_pred_cvpr2018_sist/" \
                   "ano_pred_cvpr2018/" \
                   "Data/avenue/testing/frames"
    mds = Dataset_Testing(video_folder, time_steps=10, num_pred=0,
                    resize_height=225, resize_width=225)

    # 1 就能支持变长的 seq_len, 否则在拼接成大矩阵时，会报 seq_len 这个 维度不一致的错误
    dataloader = DataLoader(mds, batch_size=1,
                            shuffle=False, num_workers=4)
    cnt = 0 # 统计 X 到底有多少帧，Y需要跟踪转化代码
    for i_batch, (X,Y) in enumerate(dataloader):
        tmp = X.shape[1] # 每次 X 的 seq_len
        cnt+= tmp
    print("X 一共有： {} 帧".format(cnt)) # 15324 ，没有问题，等于 gt_frames_len
    # gt一共有 15324 帧
    # 所以问题出现在 Y的转化过程中

def test_get_mean():
    dataset_name = "avenue"
    res = get_mean(dataset_name)
    print("res: ", res.shape)
    #
    dataset_name = "ped1"
    res = get_mean(dataset_name)
    print("res: ", res.shape)
    #
    dataset_name = "ped2"
    res = get_mean(dataset_name)
    print("res: ", res)

if __name__ == '__main__':
    # test_1()
    # test___init__()
    # test_get_vid_start_end()
    # test___getitem__() #
    # test_batchLoader()
    ########################################
    # test Dataset_Testing
    # test_get_vid_start_end_Dataset_Testting()
    # test_Dataset_Testing()
    # test_total_frames_len()
    # test_Dataset_Testing_Y_lens()
    # pass
    #########################################
    # test get_mean()
    test_get_mean()