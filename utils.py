# coding:utf8
import visdom
import time
import numpy as np
from tensorboardX import SummaryWriter


class Visualizer(object):
    '''
    封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
    调用原生的visdom接口
    '''

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)

        # 画的第几个数，相当于横座标
        # 保存（’loss',23） 即loss的第23个点
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        '''
        修改visdom的配置
        '''
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        '''
        一次plot多个
        @params d: dict (name,value) i.e. ('loss',0.11)
        '''
        for k, v in d.iteritems():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.iteritems():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        '''
        self.plot('loss',1.00)
        '''
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=unicode(name),
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        '''
        self.img('input_img',t.Tensor(64,64))
        self.img('input_imgs',t.Tensor(3,64,64))
        self.img('input_imgs',t.Tensor(100,1,64,64))
        self.img('input_imgs',t.Tensor(100,3,64,64),nrows=10)
        ！！！don‘t ~~self.img('input_imgs',t.Tensor(100,64,64),nrows=10)~~！！！
        '''
        self.vis.images(img_.cpu().numpy(),
                        win=unicode(name),
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win='log_text'):
        '''
        self.log({'loss':1,'lr':0.0001})
        '''

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'), \
            info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)

class TensorboardX_utils(object):
    from tensorboardX import SummaryWriter

    def __init__(self, logdir="logdir", comment="comment", model=None, ): # log_dir=None，默认为 runs
        self._logdir = logdir
        self._comment = comment
        self.__model = model

    # 最好是一次性把所有需要打印的东西准备好，然后一次性写入
    def add_all(self, model_input): # model_input 这种还是外部传入比较方便？
        with SummaryWriter(log_dir=self._logdir, comment=self._comment) as writer:
            # add graph
            self._add_graph(writer, self.__model, model_input)
            # add text ?

    # 下面就写一些辅助函数 （尽量与 class 解耦，只写独立的函数）
    def _add_graph(self, writer, model, model_input):
        writer.add_graph(model, model_input)  # model_input 只要 shape 没问题就OK，torch.rand(shape)就OK

    # def add_some(self,input):
    #     # scalar
    #     writer.add_scalar('data/scalar1', dummy_s1[0], n_iter)
    #     # add_scalars
    #     writer.add_scalars('data/scalar_group', {'xsinx': n_iter * np.sin(n_iter),
    #                                              'xcosx': n_iter * np.cos(n_iter),
    #                                              'arctanx': np.arctan(n_iter)}, n_iter)
    #     # image
    #     writer.add_image('Image', x, n_iter)  # 后面的覆盖前面的
    #     # audio
    #     writer.add_audio('myAudio', dummy_audio, n_iter, sample_rate=sample_rate)  # 后面的覆盖前面的
    #     # text
    #     writer.add_text('Text', 'text logged at step:' + str(n_iter), n_iter)  # 后面没有覆盖前面，why ?
    #     # histogram and distribution
    #     writer.add_histogram(name, param.clone().cpu().data.numpy(), n_iter)
    #     # pr_curve
    #     writer.add_pr_curve('xoxo', np.random.randint(2, size=100), np.random.rand(100), n_iter)
    #     # embedding
    #     images = dataset.test_data[:100].float()
    #     label = dataset.test_labels[:100]
    #     features = images.view(100, 784)
    #     writer.add_embedding(features, metadata=label, label_img=images.unsqueeze(1))


