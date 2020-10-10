# 转发服务器上的8000端口数据到本地的8001端口： ssh -L 8001:127.0.0.1:8000 zh@10.21.25.237
# 本地访问：127.0.0.1:8001

import torch as t
import visdom

# 新建一个连接客户端
# 指定env = u'test1'，默认端口为8097，host是‘localhost'
vis = visdom.Visdom(env=u'test1',use_incoming_socket=False)

x = t.arange(1, 30, 0.01)
y = t.sin(x)
vis.line(X=x, Y=y, win='sinx', opts={'title': 'y=sin(x)'})

# step1: 在服务器上运行： python -m visdom.server
# step2: 运行测试的 vis(比如我这里是env=u'test1') 所在的代码段
# step3: 在浏览器中输入：server_ip:port(port在服务器端有输出，可查)，然后找到
# 'test1'这个 vis_env，即可查看本代码段所有可视化内容
# 评价：非常方便！