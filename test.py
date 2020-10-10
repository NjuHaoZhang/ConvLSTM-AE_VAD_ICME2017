# 转发服务器上的8000端口数据到本地的8001端口： ssh -L 8001:127.0.0.1:8000 zh@10.21.25.237
# 本地访问：127.0.0.1:8001
# tensorboard --logdir="./summary" --port=8000
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
from torchnet import meter
import tqdm
from sklearn import metrics
import scipy.io as scio
from tensorboardX import SummaryWriter

# 自定义utils
#
# from config import Config_avenue_testing as Config # for testing avenue
# from config import Config_ped1_testing as Config # for testing ped1
from config import Config_ped2_testing as Config # for testing ped2


#
from Dataset import MyDataset, Dataset_Testing
from convlstm import ConvLSTM,ConvLSTMCell
from model import ConvLSTMAE
#
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device_idx = Config.device_idx
device = torch.device("cuda:" + device_idx)
# CUDA_VISIBLE_DEVICES=1 python test.py
#
torch.manual_seed(1)    # reproducible
#
opt = Config()

# Hyper Parameters
EPOCH = opt.EPOCH #
BATCH_SIZE = opt.BATCH_SIZE
TIME_STEPS = opt.TIME_STEPS # TODO：还是 9 ？caffe code 有一个 spilce op 没看懂！
NUM_PRED = opt.NUM_PRED # 用于预测的，比如预测生成1帧
# 配置参数
video_folder = opt.video_folder
gt_path = opt.gt_path
data_set_name = opt.dataset_name
#
num_workers = opt.num_workers
new_weight, new_height = opt.new_weight, opt.new_height # 与作者caffe code 保持一致


def test():
    # 数据
    myDataset = Dataset_Testing(video_folder, time_steps=TIME_STEPS, num_pred=NUM_PRED,
                          resize_height=new_weight, resize_width=new_height,
                                channel=opt.channel)
    # TODO: data augmentation
    # batch_size 设为1 就能避免 矩阵并行加速导致(seq_len 维度 不一致)的报错
    # 千万注意：shuffle=False
    dataloader = DataLoader(myDataset, batch_size=1, shuffle=False,
                            num_workers=num_workers)
    # TODO：可视化相关功能
    writer = SummaryWriter(log_dir=opt.dataset_name, comment="ConvLSTM-AE")
    # 模型
    model = ConvLSTMAE(opt)
    #
    # TODO 注意这里的 losd, 其实还有一个参数：map_location （如果GPU在load处报错，考虑这个报错）
    model = model.load(opt.model_ckpt).eval()  # TODO：model_ckpt 根据 train中的设置填写
    # model = model.load(opt.model_ckpt, map_location="1").eval()  # TODO：model_ckpt 根据 train中的设置填写
    # print("model state: ", model.state_dict())
    #
    model.to(device)
    # add_graph
    shape = [1, opt.TIME_STEPS, opt.new_weight, opt.new_height, opt.channel]
    print("shape: ", shape)
    model_input = torch.rand(shape).to(device)
    writer.add_graph(model, input_to_model=model_input)  # 暂时只查看图，loss todo
    writer.close()

    # 计算 AUC，ROC
    score = np.array([], dtype=np.float32)  # normal score(异常事件分数越低)
    loss_total_np = np.array([])
    for i_batch, (X, Y) in tqdm.tqdm(enumerate(dataloader)):
        # 计算 metrics
        # TODO: 对 X，Y做一些预处理
        # print("X,Y size() is : ", X.size(), Y.size())
        X, Y = X.to(device), Y.to(device)  # the input of model need to(cuda)
        # print("X.shape, Y.shape: ", X.size(), Y.size())
        #
        # print("76 test!")
        y_list = model(X)  #
        # print("size of y_list[0]: ", y_list[18].size())
        y_hat = torch.stack(y_list, 0)
        # print("y_hat size(): ", y_hat.size()) # 这里有个坑，好坑，由于
        # 计算图,tensor,自动求导的缘故，不能直接执行 np.concatenate(),要
        # 优先选用pytorch的官方 tensor library: torch.stack
        #
        # print("Y size(): ", Y.size())
        Y = Y.permute(1, 0, 4, 3, 2)
        # print("new Y size(): ", Y.size())

        # 开始计算两个 tensor 之间的差距
        dis_Y_y_hat = calcu_distance(Y, y_hat) # [batch_size, time_step] list
        # 支持每次 dis_Y_y_hat shape 不同
        loss_total_np = np.append(loss_total_np, np.array(dis_Y_y_hat))

    print("loss_total_np shape: ", loss_total_np.shape)

    # 根据 score and label 计算 AUC
    label = get_gt(gt_path, data_set_name, video_folder)
    print("label: ", label.shape)
    #
    loss_total_np = loss_total_np.flatten() # 直接展平
    print("loss_total_np shape: ", loss_total_np.shape)
    np_min, np_max = np.min(loss_total_np), np.max(loss_total_np)
    print("np_min, np_max: ", np_min, np_max)
    score = (loss_total_np - np_min) / (np_max - np_min)
    score = 1 - score #
    print("score: ", score.shape)

    res = calcu_roc(score, label) # TODO 很奇怪，竟然没有根据 score和阈值来判定异常

    # 我怀疑是calcu_roc里面的sklearn 自适应使用了阈值
    # # 可视化
    # if (i_batch + 1) % opt.plot_every == 0:
    #     # if os.path.exists(opt.debug_file):
    #     #     ipdb.set_trace()
    #     # loss绘图
    #     # vis.plot('loss', loss_meter.value()[0])
    #     print(' i_batch: ', i_batch,
    #           ' | train loss: %.4f' % loss.data)

def get_gt(gt_path, data_set_name, video_folder):
    import glob

    gt = scio.loadmat(gt_path, squeeze_me=True)['gt']
    # print(gt.shape) # (21,),因为有21个Testing 子目录
    # print("gt[0]: ", gt[0]) # 我感觉像是：[start,end]组成的2-D tensor
    # print("gt.ndim: ", gt.ndim) # 1 ??
    # 下面是 pixel-wise mask gt (暂时我先不处理这个)
    if data_set_name == 'exit' or data_set_name == 'enter_authors':
        mask = np.ones((225, 225, 1), dtype=np.uint8)
        mask[165: 195, 125: 215, 0] = 0
    else:
        mask = np.ones((225, 225, 1), dtype=np.uint8) # 全1 的像素矩阵(gray)
    #
    label = np.array([], dtype=np.int32)
    videos = glob.glob(os.path.join(video_folder, '*'))
    # print("len(videos): ", len(videos))
    # print("videos: ", videos)
    i = 0
    for video in sorted(videos): # 遍历 21 个子目录
        # print("os.path.join(video_name, '*.jpg'): ", os.path.join(video, '*.jpg'))
        frames_path = glob.glob(os.path.join(video, '*.jpg'))
        length = len(frames_path) # 当前这个子目录的帧的数目
        # print("length: ", length)
        video_label = np.zeros(length, dtype=np.float32)
        if gt.ndim == 2: # 存在第一个维度，是batch ? 后面再说
            gt = gt.reshape(-1, gt.shape[0], gt.shape[1])
        gt[i] = np.reshape(gt[i], (2, -1)) # 2作为行数，是start和end
        # print("gt.shape: ", gt.shape)
        # print("gt[{}] shape: {}".format(i, gt[i].shape))
        for j in range(gt[i].shape[1]): # 因为gt[i].shape[0]是行数2：第一行是start list,第二行是end
            for k in range(gt[i][0, j] - 1, gt[i][1, j]): # 遍历所有列数，即每一个异常帧
                video_label[k] = 1
        # print("video_label: ", video_label)
        label = np.append(label, video_label)
        # print("len(label): ", len(label))
        i += 1 # 计数器自增
    # print("label: ", label.shape, type(label))
    # print("label[0]: ", label[0])
    # print("label: ", label)
    return label # 当前dataset的所有帧的gt(frame-level)

def calcu_distance(Y, y_hat):
    # print("Y, y_hat device: ", Y.device, y_hat.device)
    # 必须要先展平这个高维 tensor
    y_hat = y_hat.view(y_hat.shape[1], y_hat.shape[0], -1) # (b,t,flatten_size)
    # print("new y_hat size: ", y_hat.size()) # (b,t,vec)
    Y = Y.contiguous().view(Y.shape[1], Y.shape[0], -1)  # 这里有个坑，contiguous
    # print("new Y size: ", Y.size())
    res = np.array([], dtype=np.float)
    iter_num = y_hat.shape[0]  # batch_size
    # print("iter_num: ", iter_num)
    batch_tmp = []
    for it in range(iter_num):
        res = torch.norm((Y[it] - y_hat[it]), p=2, dim=1).cpu()  # 按照行计算，即每帧
        # print("res: ", res.shape, res)
        # 把这(2*T-1)个loss重新拼接为：T个loss
        res = res.tolist()  # tensor to list
        idx, tmp = 0, []
        # print("res type: ", type(res))break
        # print("\nres: ", res, '\n', len(res))
        # 根据本论文的model网络图， res只能为奇数，不可能为偶数！
        if len(res) == 1:
            tmp.append(res[0])  # 只有这一个元素
        else: # 下面的逻辑适合 len(res)>=3的奇数
            while idx < len(res) - 1:  # 即  len-2, res倒数第二个元素的idx
                if idx == 0:
                    tmp.append(res[idx])  # 0,
                    idx += 1
                else:
                    tmp.append(res[idx] + res[idx + 1])  # (1,2) (3,4)
                    idx += 2
        # print("\ntmp: ", tmp, '\n', len(tmp))  # TODO: 早上起来check
        batch_tmp.append(tmp)
    # print("batch_tmp: ", np.array(batch_tmp).shape) # [8,10] 8个batch，seq_len是10
    return batch_tmp

def calcu_roc(pred, label):
    fpr, tpr, thresholds = metrics.roc_curve(label, pred, pos_label=0)
    print("fpr: {}\n, tpr:{}\n, thresholds:{}\n".format(fpr, tpr, thresholds))
    auc =  metrics.auc(fpr, tpr)
    print("auc: ", type(auc), auc)
    return auc

##################################################################################
# 下面是上述函数的测试小程序
def test_get_gt():
    gt_path = "/home/zh/Papers_Code/" \
              "ano_pred_cvpr2018_sist/ano_pred_cvpr2018/" \
              "Data/avenue/avenue.mat"
    data_set_name = "avenue"
    video_folder = "/home/zh/Papers_Code/ano_pred_cvpr2018_sist/" \
                   "ano_pred_cvpr2018/" \
                   "Data/avenue/testing/frames"
    get_gt(gt_path, data_set_name, video_folder)



#################################################################################
# 运行处

if __name__ == '__main__':
    # test_get_gt()
    test()