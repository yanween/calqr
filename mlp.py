import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
'''
    MLP的Projection算子，And算子,Or算子，Not算子都使用MLP实现，
    MLP相较于前面几个模型增加了And算子的建模，相当于Intersection 
    And算子可以用来做CenterNet
    即这四个模型的架构完全相同，只有参数不同
'''
class ProjectionMLP(nn.Module):
    def __init__(self, n_layers, entity_dim):
        super(ProjectionMLP, self).__init__()
        self.n_layers = n_layers
        for i in range(1, self.n_layers+1):
            setattr(self, "proj_layer_{}".format(i), nn.Linear(2 * entity_dim, 2 * entity_dim))
        self.last_layer = nn.Linear(2 * entity_dim, entity_dim)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=-1)
        for i in range(1, self.n_layers+1):
            # 每一轮都使用ReLu激活函数
            x = F.relu(getattr(self, "proj_layer_{}".format(i))(x))
        x = self.last_layer(x)
        return x


class AndMLP(nn.Module):
    def __init__(self, n_layers, entity_dim):
        super(AndMLP, self).__init__()
        self.n_layers = n_layers
        self.layers = []
        for i in range(1, self.n_layers + 1):
            setattr(self, "and_layer_{}".format(i), nn.Linear(2 * entity_dim, 2 * entity_dim))
        self.last_layer = nn.Linear(2 * entity_dim, entity_dim)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=-1)
        for i in range(1, self.n_layers + 1):
            x = F.relu(getattr(self, "and_layer_{}".format(i))(x))
        x = self.last_layer(x)
        return x


class OrMLP(nn.Module):
    def __init__(self, n_layers, entity_dim):
        super(OrMLP, self).__init__()
        self.n_layers = n_layers
        self.layers = []
        for i in range(1, self.n_layers + 1):
            setattr(self, "or_layer_{}".format(i), nn.Linear(2 * entity_dim, 2 * entity_dim))
        self.last_layer = nn.Linear(2 * entity_dim, entity_dim)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=-1)
        for i in range(1, self.n_layers + 1):
            x = F.relu(getattr(self, "or_layer_{}".format(i))(x))
        x = self.last_layer(x)
        return x


'''
    考虑电位(0,1)的取逆问题，有: 0->1; 1->0
    这里的信号相当于模拟信号，可以这样考虑：
        1. 将取值范围[min, max]中的任何一个值压缩到[0, 1]
        2. 考虑取逆问题
    通过线性变化，能达到电位取反的问题，同时也能解决模拟信号取反的问题
    问题在于ReLU的小于0的部分直接归0，真的是我们想要的结果吗？
    为什么不使用在小于0区间和大于0区间斜率不等的分段Linear做激活函数呢？
        mixer的激活函数使用的是GELU，虽然比我们设想的情况计算梯度更为复杂，在某些任务上确实效果更强
        但是在建模的时候模型比常规的MLP效果高出很多
'''
class NotMLP(nn.Module):
    def __init__(self, n_layers, entity_dim):
        super(NotMLP, self).__init__()
        self.n_layers = n_layers
        self.layers = []
        for i in range(1, self.n_layers + 1):
            '''
            setattr() 函数指定对象的指定属性的值。
            setattr(object, attribute, value)
            '''
            setattr(self, "not_layer_{}".format(i), nn.Linear(entity_dim, entity_dim))
        self.last_layer = nn.Linear(entity_dim, entity_dim)

    def forward(self, x):
        for i in range(1, self.n_layers + 1):
            '''
            getattr() 函数从指定的对象返回指定属性的值。
            getattr(object, attribute, default)
            '''
            x = F.relu(getattr(self, "not_layer_{}".format(i))(x))
        x = self.last_layer(x)
        return x
