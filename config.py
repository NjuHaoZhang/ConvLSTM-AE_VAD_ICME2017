'''
本文件是配置文件
'''

# 一些公共配置
class BaseConfig:
    #
    # batch_size = 8
    # shuffle = True
    # num_workers = 4
    # rnn_hidden = 256
    # embedding_dim = 256
    # num_layers = 2
    # share_embedding_weights = False
    # debug_file = '/tmp/debugc'
    pass

###################################################################################
# 本次配置是 for avenue_training
class Config_avenue_training(BaseConfig):
    #
    prefix = 'checkpoints/'  # 模型保存前缀
    plot_every = 10
    model_ckpt = None  # 模型断点保存路径
    #
    # Training 相关的配置
    shuffle = True
    EPOCH = 10
    BATCH_SIZE = 32  # 作者 caffe的batch_size是8，比较小
    TIME_STEPS = 10  # TODO：还是 9 ？caffe code 有一个 spilce op 没看懂！
    NUM_PRED = 0  # 用于预测的，比如预测生成1帧
    LR = 0.005  # 本网络各个组件的学习率是不同的，此处只是一个通用的lr
    video_folder = "/home/zh/Papers_Code/ano_pred_cvpr2018_sist/ano_pred_cvpr2018/" \
                   "Data/" \
                   "avenue/" \
                   "training/frames"
    num_workers = 4
    new_weight, new_height = 225, 225  # 与作者caffe code 保持一致
    # Model相关的配置
    dataset_name = "avenue"
    channel = 1  # avenue虽然是RGB，但是论文是转为Gray 去 Train
    use_gpu = True
    device_idx = "3"  # GPU:2

####################################################################################

# 本次配置是 for avenue_testing
class Config_avenue_testing(BaseConfig):
    #
    plot_every = 10
    model_ckpt = "checkpoints/avenue_0425_2023.pkl"  # 模型断点保存路径
    #
    #
    shuffle = False # 必须关掉！！！
    EPOCH = 1 # testing只需要扫描数据集一次
    BATCH_SIZE = 1  # testing暂时设置 1
    TIME_STEPS = 10  # TODO：还是 9 ？caffe code 有一个 spilce op 没看懂！
    NUM_PRED = 0  # 用于预测的，比如预测生成1帧
    video_folder = "/home/zh/Papers_Code/ano_pred_cvpr2018_sist/" \
                   "ano_pred_cvpr2018/Data/" \
                   "avenue/" \
                   "testing/frames"
    num_workers = 4
    new_weight, new_height = 225, 225  # 与作者caffe code 保持一致
    #
    dataset_name = "avenue"
    gt_path = "/home/zh/Papers_Code/" \
              "ano_pred_cvpr2018_sist/ano_pred_cvpr2018/Data/" \
              "avenue/avenue.mat"
    channel = 1  # avenue是RGB image，但是论文转化为Gray
    use_gpu = True
    device_idx = "1"  # GPU:3

#############################################################################################

###################################################################################
# 本次配置是 for ped1_training
class Config_ped1_training(BaseConfig):
    #
    prefix = 'checkpoints/'  # 模型保存前缀
    plot_every = 10
    model_ckpt = None  # 模型断点保存路径
    #
    # Training 相关的配置
    shuffle = True
    EPOCH = 30
    BATCH_SIZE = 32  # 作者 caffe的batch_size是8，比较小
    TIME_STEPS = 10  # TODO：还是 9 ？caffe code 有一个 spilce op 没看懂！
    NUM_PRED = 0  # 用于预测的，比如预测生成1帧
    LR = 0.005  # 本网络各个组件的学习率是不同的，此处只是一个通用的lr
    video_folder = "/home/zh/Papers_Code/ano_pred_cvpr2018_sist/ano_pred_cvpr2018/" \
                   "Data/" \
                   "ped1/" \
                   "training/frames"
    num_workers = 4
    new_weight, new_height = 225, 225  # 与作者caffe code 保持一致
    # Model相关的配置
    dataset_name = "ped1"
    channel = 1  # ped1是gray image
    use_gpu = True
    device_idx = "2"  # GPU:2

####################################################################################
# 本次配置是 for ped2_training
class Config_ped2_training(BaseConfig):
    #
    prefix = 'checkpoints/'  # 模型保存前缀
    plot_every = 10
    model_ckpt = None  # 模型断点保存路径
    #
    #
    shuffle = True
    EPOCH = 50
    BATCH_SIZE = 32  # 作者 caffe的batch_size是8，比较小
    TIME_STEPS = 10  # TODO：还是 9 ？caffe code 有一个 spilce op 没看懂！
    NUM_PRED = 0  # 用于预测的，比如预测生成1帧
    LR = 0.001  # 本网络各个组件的学习率是不同的，此处只是一个通用的lr
    video_folder = "/home/zh/Papers_Code/ano_pred_cvpr2018_sist/" \
                   "ano_pred_cvpr2018/Data/" \
                   "ped2/" \
                   "training/frames"
    num_workers = 4
    new_weight, new_height = 225, 225  # 与作者caffe code 保持一致
    #
    dataset_name = "ped2"
    channel = 1  # ped2是gray image
    use_gpu = True
    device_idx = "3"  # GPU:2

####################################################################################

# 本次配置是 for ped1_testing
class Config_ped1_testing(BaseConfig):
    #
    plot_every = 10
    model_ckpt = "checkpoints/ped1_0426_0457.pkl"  # 模型断点保存路径
    #
    #
    shuffle = False # 必须关掉！！！
    EPOCH = 1 # testing只需要扫描数据集一次
    BATCH_SIZE = 1  # testing暂时设置 1
    TIME_STEPS = 10  # TODO：还是 9 ？caffe code 有一个 spilce op 没看懂！
    NUM_PRED = 0  # 用于预测的，比如预测生成1帧
    video_folder = "/home/zh/Papers_Code/ano_pred_cvpr2018_sist/" \
                   "ano_pred_cvpr2018/Data/" \
                   "ped1/" \
                   "testing/frames"
    num_workers = 4
    new_weight, new_height = 225, 225  # 与作者caffe code 保持一致
    #
    dataset_name = "ped1"
    gt_path = "/home/zh/Papers_Code/" \
              "ano_pred_cvpr2018_sist/ano_pred_cvpr2018/Data/" \
              "ped1/ped1.mat"
    channel = 1  # ped2是gray image
    use_gpu = True
    device_idx = "1"  # GPU:3

#############################################################################################

# 本次配置是 for ped2_testing
class Config_ped2_testing(BaseConfig):
    #
    plot_every = 10
    model_ckpt = "checkpoints/ped2_0427_1403.pkl"  # 模型断点保存路径
    #
    #
    shuffle = False # 必须关掉！！！
    EPOCH = 1 # testing只需要扫描数据集一次
    BATCH_SIZE = 1  # testing暂时设置 1
    TIME_STEPS = 10  # TODO：还是 9 ？caffe code 有一个 spilce op 没看懂！
    NUM_PRED = 0  # 用于预测的，比如预测生成1帧
    video_folder = "/home/zh/Papers_Code/ano_pred_cvpr2018_sist/" \
                   "ano_pred_cvpr2018/Data/" \
                   "ped2/" \
                   "testing/frames"
    num_workers = 4
    new_weight, new_height = 225, 225  # 与作者caffe code 保持一致
    #
    dataset_name = "ped2"
    gt_path = "/home/zh/Papers_Code/" \
              "ano_pred_cvpr2018_sist/ano_pred_cvpr2018/Data/" \
              "ped2/ped2.mat"
    channel = 1  # ped2是gray image
    use_gpu = True
    device_idx = "3"  # GPU:3

##############################################################################################