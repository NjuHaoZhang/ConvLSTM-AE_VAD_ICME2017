# 转发服务器上的8000端口数据到本地的8001端口： ssh -L 8001:127.0.0.1:8000 zh@10.21.25.237
# 本地访问：127.0.0.1:8001
# tensorboard --logdir="./summary" --port=8000
#
import os
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
from utils import Visualizer
from torchnet import meter
import tqdm
from tensorboardX import SummaryWriter

# 自定义utils
#
# from config import Config_avenue_training as Config # avenue_training
# from config import Config_ped1_training as Config # training ped1
from config import Config_ped2_training as Config # training ped2
#
from Dataset import MyDataset
from convlstm import ConvLSTM,ConvLSTMCell
from model import ConvLSTMAE
from utils import TensorboardX_utils
#
torch.manual_seed(1)    # reproducible
#
# os.environ["CUDA_VISIBLE_DEVICES"] = '3' # pytorch 有更简单的办法，to("cuda:0")
device_idx = Config.device_idx
device = torch.device("cuda:" + device_idx)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 很奇怪的bug: 只能支持 cuda:0，其他1,2,3都会报错

# 传递给model 的配置信息
opt = Config()
#
# Training相关的配置参数
EPOCH = opt.EPOCH
BATCH_SIZE = opt.BATCH_SIZE
TIME_STEPS = opt.TIME_STEPS     # TODO：还是 9 ？caffe code 有一个 spilce op 没看懂！
NUM_PRED = opt.NUM_PRED         # 用于预测的，比如预测生成1帧
LR = opt.LR                     # TODO 本网络各个组件的学习率是不同的，此处只是一个通用的lr
# 配置参数
video_folder = opt.video_folder
num_workers = opt.num_workers
new_weight, new_height = opt.new_weight, opt.new_height     # 与作者caffe code 保持一致
shuffle = opt.shuffle

def train(epoch, loss_meter, writer):
    # 数据
    myDataset = MyDataset(video_folder, time_steps=TIME_STEPS, num_pred=NUM_PRED,
                          resize_height=new_weight, resize_width=new_height,
                          channel=opt.channel)
    # TODO: data augmentation
    dataloader = DataLoader(myDataset, batch_size=BATCH_SIZE, shuffle=shuffle,
                            num_workers=num_workers)

    # 模型
    model = ConvLSTMAE(opt).train()
    if opt.model_ckpt:
        # TODO 注意这里的 losd, 其实还有一个参数：map_location （如果GPU在load处报错，考虑这个报错）
        model.load(opt.model_ckpt)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR) # 要不要改成和论文一致呢？ TODO
    # TODO：具体设置下各个参数的初始化，以及，优化的学习率，以及其他超参
    # criterion = nn.MSELoss().to(device)  # EuclideanLoss
    criterion = nn.MSELoss().to(device)  #
    #
    model.to(device)
    # 统计
    loss_meter.reset()

    cnt_batch = 0
    for i_batch, (X, Y) in tqdm.tqdm(enumerate(dataloader)):
        cnt_batch = cnt_batch + 1  # 手动记录 batch_num
        # 训练
        # TODO: 对 X，Y做一些预处理
        # print("X,Y size() is : ", X.size(), Y.size())
        X, Y = X.to(device), Y.to(device)  # the input of model need to(cuda)        #
        # y_list = convLSTMAE(X).to(device) # model need to cuda(gpu)
        y_list = model(X)  #
        # print("size of y_list[0]: ", y_list[18].size())
        y_hat = torch.stack(y_list, 0)
        # print("y_hat size(): ", y_hat.size()) # TODO 好坑这里，由于
        # 计算图,tensor,自动求导的缘故，不能直接执行 np.catxxx()
        # print("Y size(): ", Y.size())
        Y = Y.permute(1, 0, 4, 3, 2)
        # print("new Y size(): ", Y.size())
        # TODO 其实Dataset一开始就应该转型为 [b,t,c,w,h]

        loss = criterion(y_hat, Y)  # mean square error
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        # loss_meter.add(loss.data[0])

        # 可视化
        if (i_batch + 1) % opt.plot_every == 0:
            # if os.path.exists(opt.debug_file):
            #     ipdb.set_trace()
            # loss绘图
            # vis.plot('loss', loss_meter.value()[0])
            # 控制台输出 Loss
            print(' epoch: ', epoch, ' | i_batch: ', i_batch,
                  ' | train loss: %.4f' % loss.data)
            #
            writer.add_scalar("train_loss", loss.item(), cnt_batch)

    model.save()  # 每个 epoch 完毕保存下模型
    # 注意一件比较坑爹的事：pytorch的save默认竟然不会创建path包含的文件夹
    # 所以诸如 checkpoints这样的父目录需要自己创建

# eval TODO
def eval(epoch, loss_meter):
    # 数据
    myDataset = MyDataset(video_folder, time_steps=TIME_STEPS, num_pred=NUM_PRED,
                          resize_height=new_weight, resize_width=new_height)
    # TODO: data augmentation
    dataloader = DataLoader(myDataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=num_workers)
    # TODO：可视化相关功能

    # 模型
    model = ConvLSTMAE(opt)
    model = model.load(opt.model_ckpt).eval()  # TODO：model_ckpt 根据 train中的设置填写
    model.to(device)
    criterion = nn.MSELoss().to(device)  # EuclideanLoss

    for i_batch, (X, Y) in tqdm.tqdm(enumerate(dataloader)):
        # 计算 metrics
        # TODO: 对 X，Y做一些预处理
        # print("X,Y size() is : ", X.size(), Y.size())
        X, Y = X.to(device), Y.to(device)  # the input of model need to(cuda)
        #
        # y_list = convLSTMAE(X).to(device) # model need to cuda(gpu)
        y_list = model(X)  #
        # print("size of y_list[0]: ", y_list[18].size())
        y_hat = torch.stack(y_list, 0)
        # print("y_hat size(): ", y_hat.size()) # TODO 好坑这里，由于
        # 计算图,tensor,自动求导的缘故，不能直接执行 np.catxxx()
        # print("Y size(): ", Y.size())
        Y = Y.permute(1, 0, 4, 3, 2)
        # print("new Y size(): ", Y.size())
        loss = criterion(y_hat, Y)  # mean square error

        # 可视化
        if (i_batch + 1) % opt.plot_every == 0:
            # if os.path.exists(opt.debug_file):
            #     ipdb.set_trace()
            # loss绘图
            # vis.plot('loss', loss_meter.value()[0])
            print(' i_batch: ', i_batch,
                  ' | train loss: %.4f' % loss.data)
            # 保存 model : 每个epoch完毕就 save 一次

if __name__ == '__main__':

    #  TODO: 训练集划分

    # 每个 epoch 做一次train & eval
    loss_meter_train = meter.AverageValueMeter()
    loss_meter_eval = meter.AverageValueMeter()

    # TODO：可视化相关功能
    # vis_tool = TensorboardX_utils(logdir=opt.dataset_name, comment="ConvLSTM-AE", model=model)
    with SummaryWriter(log_dir=opt.dataset_name, comment="ConvLSTM-AE") as writer:
        for epoch in range(EPOCH):
            train(epoch, loss_meter_train, writer)
            # eval(epoch, loss_meter_eval)

        # add_graph： 在训练完毕后执行
        model = ConvLSTMAE(opt)
        # model = model.load(opt.model_ckpt).eval()  # 应该不需要这两个
        # model.to(device)
        shape = [1,opt.TIME_STEPS, opt.channel, opt.new_weight, opt.new_height]
        # print("shape: ", shape)
        model_input = torch.rand(shape)
        writer.add_graph(model, input_to_model=model_input)



