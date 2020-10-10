#
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import time
#
# 自定义utils
from convlstm import ConvLSTM,ConvLSTMCell
from config import Config_ped2_testing as Config
#
torch.manual_seed(1)    # reproducible


class ConvLSTMAE(nn.Module):
    def __init__(self, opt):
        super(ConvLSTMAE, self).__init__()

        #
        self.opt = opt # 配置参数
        #
        self.Conv3 = nn.Sequential(
            # TODO：图片的channel不是1，但是作者网络设置是1，有冲突？
            nn.Conv2d(self.opt.channel, 128, 7, 4, 3),
            nn.ReLU(),
            nn.Conv2d(128, 256, 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.ReLU(),
        )
        # 下面的ConvLSTM和本论文作者的code有一些差异，等最后再调节这个！
        height, width = 15, 15 # TODO 每次都要根据Conv3输出的x的w,h来设置
        self.ConvLSTM = ConvLSTM(
                         input_size=(height, width), # Conv3后tensor的w,h
                         input_dim=512, # Conv3后tensor的channel
                         hidden_dim=[64, 64, 128],
                         kernel_size=(3, 3),
                         num_layers=3,
                         batch_first=True,
                         bias = False, # 作者论文
                         return_all_layers = False # 只返回最后一层
                         )
        self.Deconv3_1 = nn.Sequential(
            nn.ConvTranspose2d(128, 256, 3, 2, 1), # TODO input_channel根据ConvLSTM的output改
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 5, 2, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, self.opt.channel, 7, 4, 3),
            nn.Tanh(),
        )
        self.Deconv3_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 256, 3, 2, 1), # TODO input_channel根据ConvLSTM的output改
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 5, 2, 2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, self.opt.channel, 7, 4, 3),
            nn.Tanh(),
        )

    def forward(self, x):
        # print("x shape: ", x.size())
        # 这里其实应该要根据 x的shape来确定 t,b,c,w,h
        seq_len, b, c, w, h = x.shape[1], x.shape[0], x.shape[4], x.shape[2], x.shape[3]
        x = x.view(-1, c, h, w) # 因为conv2d不支持5-d tensor故 reshape 成 (n,c,w,h)
        # 上面一行代码有个隐含的bug: 当 Batch_size设置的不合适(就是total_num/batch_size不是整数，
        # 最后一个batch的长度并不是batchsize), 这里 reshape 就会报错， 所以第一维改为：自动推断
        # print("89行的 x size: ", x.size())
        x = self.Conv3(x)
        # print("self.Conv3(x) return x.shape is : ", x.size()) # w,h=15x15,c=512
        x = x.view(-1, seq_len, 512, 15, 15) # TODO reshape，以便输入ConvLSTM, batch frist
        # 上面一行代码第一维，防止出现某个batch长度不是 batch_size，而报错，改为自动推断
        # print("93行的 5-D x'size is:", x.size())
        output_list, hidden_state_list = self.ConvLSTM(x)
        # print("output_list, hidden_state_list 's size():",
        #       np.array(output_list).shape, np.array(hidden_state_list).shape)
        # print("output_list, hidden_state_list 's len", len(output_list), len(hidden_state_list))
        # print("output_list[0]'s size(): ", output_list[0].size()) #[8, 10, 128, 15, 15]
        output = output_list[0] # 因为只用到最后一层的output, [8, 10, 128, 15, 15]
        output = output.permute(1, 0, 2, 3, 4) # 8,10 调换顺序为 10,8
        # print("output's size: ", output.size()) # 10是时间窗口
        # print("output[0]'size: ", output[0].size())
        #
        y_list = []
        for t in range(seq_len): # 每个时间步单独处理
            if t == 0:
                y = self.Deconv3_1(output[t]) # 只取这个时间步的batch_tensor
                y_list.append(y)
            else:
                y_1 = self.Deconv3_1(output[t])
                y_list.append(y_1)
                y_2 = self.Deconv3_2(output[t])
                y_list.append(y_2)
        #
        return y_list # 注意要与下面的Y保持相同的shape

    def states(self):
        opt_state_dict = {attr: getattr(self.opt, attr)
                          for attr in dir(self.opt)
                          if not attr.startswith('__')}
        return {
            'state_dict': self.state_dict(),
            'opt': opt_state_dict
        }
    #
    def save(self, path=None, **kwargs):
        if path is None:
            path = '{prefix}{dataset}_{time}.pkl'.format(prefix=self.opt.prefix,
                                                      dataset=self.opt.dataset_name ,
                                                    time=time.strftime('%m%d_%H%M'))
            # mkdir(path) # 函数里面已经实现了目录已存在的处理逻辑
            # TODO：这里需要实时修改 opt.model_ckpt 吗?
        states = self.states()
        states.update(kwargs)
        torch.save(states, path)
        return path

    def load(self, path, load_opt=False):
        data = torch.load(path, map_location=lambda s, l: s)
        state_dict = data['state_dict']
        self.load_state_dict(state_dict)

        if load_opt:
            for k, v in data['opt'].items():
                setattr(self.opt, k, v)

        return self

    def get_optimizer(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr)

##########################################################################################################
# 一些辅助函数，最终要搬入 utils.py
def mkdir(path):
    # 引入模块
    import os
    # 去除首位空格
    path=path.strip()
    # 去除尾部 \ 符号
    path=path.rstrip("\\")
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists=os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        print(path+' 创建成功')
        # 创建目录操作函数
        # os.makedirs(path)
        os.mknod(path) # 创建文件
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path+' 目录已存在')
        return False