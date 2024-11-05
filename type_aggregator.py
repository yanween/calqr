# -*- coding: utf-8 -*-
"""
@Time ： 2021/10/12 12:41
@Auth ： zhiweihu
"""
import torch
import torch.nn as nn
# import torch.nn.functional as F
from abc import abstractmethod
from typing import Optional


# Aggregator抽象类，由EntityTypeAggregator进行实例化
class Aggregator(nn.Module):
    # 由于这是一个抽象类，其具体的网络层由实现类定义在__init__()中，即定义在EntityTypeAggregator()的__init__()中
    def __init__(self, input_dim, output_dim, act, self_included,
                 neighbor_ent_type_samples):  # Initialize your model & define layers
        super(Aggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.self_included = self_included
        self.neighbor_ent_type_samples = neighbor_ent_type_samples

    def forward(self, self_vectors, neighbor_vectors):  # Compute output of NN
        outputs = self._call(self_vectors, neighbor_vectors)
        return outputs

    '''
        @abstractmethod：抽象方法，含abstractmethod方法的类不能实例化，
        继承了含abstractmethod方法的子类必须复写所有abstractmethod装饰的方法，未被装饰的可以不重写
        继承语法：
        class a(ABC):
            @abstractmethod
            def xxx():
                pass
                    
        class b(a):
            def xxx():
    '''

    @abstractmethod
    def _call(self, self_vectors, entity_vectors):
        pass


# 实现Aggregator抽象类
'''
    self_included表示是否concatenate entity本身信息
    temp在inductive reasoning中不需要concatenate entity信息
'''
class EntityTypeAggregator(Aggregator):
    def __init__(self, input_dim, output_dim, act=lambda x: x, self_included=True, with_sigmoid=False,
                 neighbor_ent_type_samples=32): #Initialize model & define layers
        super(EntityTypeAggregator, self).__init__(input_dim, output_dim, act, self_included, neighbor_ent_type_samples)
        self.proj_layer = HighwayNetwork(neighbor_ent_type_samples, 1, 2, activation=nn.Sigmoid())

        multiplier = 2 if self_included else 1
        self.layer = nn.Linear(self.input_dim * multiplier, self.output_dim)
        # Xavier初始化,参考 https://cloud.tencent.com/developer/article/1627511
        nn.init.xavier_uniform_(self.layer.weight)
        self.with_sigmoid = with_sigmoid

    def _call(self, self_vectors, neighbor_vectors): # 抽象类forward()方法调用抽象方法_call(),由这里实现，Compute output of NN
        neighbor_vectors = torch.transpose(neighbor_vectors, 1, 2) #问题出现在这，这里只能处理三维形状的entity tensor
        neighbor_vectors = self.proj_layer(neighbor_vectors)
        neighbor_vectors = torch.transpose(neighbor_vectors, 1, 2)
        neighbor_vectors = neighbor_vectors.squeeze(1)

        # 包括entity信息
        if self.self_included:
            self_vectors = self_vectors.view([-1, self.input_dim])
            output = torch.cat([self_vectors, neighbor_vectors], dim=-1)
        output = self.layer(output)
        output = output.view([-1, self.output_dim])
        if self.with_sigmoid: #将值映射到(0,1)
            output = torch.sigmoid(output)

        # act默认=lambda x: x ;默认返回sigmoid()的值
        return self.act(output)


'''
    highway network: 解决网络深度加深、梯度信息回流受阻，造成网络训练困难的问题
    所谓Highway网络，无非就是输入某一层网络的数据一部分经过非线性变换，另一部分直接从该网络跨过去不做任何转换，就想走在高速公路上一样，而多少的数据需要非线性变换，多少的数据可以直接跨过去，是由一个权值矩阵和输入数据共同决定的。        
'''
class HighwayNetwork(nn.Module):
    """
        Typing.Optional类
        可选类型，作用几乎和带默认值的参数等价，不同的是使用Optional会告诉你的IDE或者框架：
        这个参数除了给定的默认值外还可以是None，而且使用有些静态检查工具如mypy时，
        对 a: int =None这样类似的声明可能会提示报错，但使用a :Optional[int] = None不会。
        原文链接：https://blog.csdn.net/qq_44683653/article/details/108990873
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 n_layers: int,
                 activation: Optional[nn.Module] = None):  # Initialize your model & define layers
        super(HighwayNetwork, self).__init__()
        self.n_layers = n_layers
        '''
            neighbor_ent_type_samples, 1, 2, activation=nn.Sigmoid()
            ModuleList的作用：不是创建三层前后连接的网络，而是创建三个上下并列的网络，所以称model里面的模型为子模块。这里只是创建了一层网络。
        '''
        # self.nonlinear = nn.ModuleList(
        #     [nn.Linear(input_dim, input_dim) for _ in range(n_layers)])
        # self.gate = nn.ModuleList(
        #     [nn.Linear(input_dim, input_dim) for _ in range(n_layers)])
        # for layer in self.gate:
        #     layer.bias = torch.nn.Parameter(0. * torch.ones_like(layer.bias))
        self.final_linear_layer = nn.Linear(input_dim, output_dim)
        # self.activation = nn.ReLU() if activation is None else activation
        # self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # Compute output of NN
        # 迭代n_layers次数
        # for layer_idx in range(self.n_layers):
            # 不使用gate_value;
            # gate_values = self.sigmoid(self.gate[layer_idx](inputs)) #计算g
            # nonlinear = self.activation(self.nonlinear[layer_idx](inputs)) #计算(W_i' * H_s^i + b_i')
            # 直接使用 nonlinear + inputs
            # inputs = gate_values * nonlinear + (1. - gate_values) * inputs
            # inputs = nonlinear + inputs
        return self.final_linear_layer(inputs)


'''
    考虑修改HighwayNetwork为ResNet看看效果
    highway network保存输入的两类信息，通过权重进行整合
    使用更简单的ResNet也许训练速度更快
    现在不考虑修改highway为ResNet 目前仍旧使用highway
'''
'''
    使用ResNet进行属性叠加
'''
class ResNetwork(nn.Module):
    pass



class Match(nn.Module):
    def __init__(self, hidden_size, with_sigmoid=False): #Initialize model & define layers
        super(Match, self).__init__()
        self.map_linear = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.trans_linear = nn.Linear(hidden_size, hidden_size)
        self.with_sigmoid = with_sigmoid

    def forward(self, inputs): #Compute output of NN
        proj_p, proj_q = inputs
        # trans_q --> key
        trans_q = self.trans_linear(proj_q)
        '''
            torch.bmm()函数作用
            计算两个tensor的矩阵乘法，torch.bmm(a,b),tensor a 的size为(b,h,w),tensor b的size为(b,w,m) 
            也就是说两个tensor的第一维是相等的，然后第一个数组的第三维和第二个数组的第二维度要求一样，对于剩下的则不做要求，输出维度 （b,h,m）
            (b,h,w) * (b,w,m) --> (b,h,m)
            原文链接：https://blog.csdn.net/qq_40178291/article/details/100302375/
        '''
        # 计算p对q的注意力,即将q的信息融入p; proj_p --> query;trans_q --> key
        att_weights = proj_p.bmm(torch.transpose(trans_q, 1, 2))
        # 过softmax()得到概率
        att_norm = torch.nn.functional.softmax(att_weights, dim=-1)
        # attention * q; proj_q --> value
        # attention的前后反了? 没反，论文中在后面的向量是wanted值(input_q)
        att_vec = att_norm.bmm(proj_q)
        elem_min = att_vec - proj_p
        elem_mul = att_vec * proj_p
        all_con = torch.cat([elem_min, elem_mul], 2)
        output = self.map_linear(all_con)
        if self.with_sigmoid:
            output = torch.sigmoid(output)
        return output


class RelationTypeAggregator(nn.Module):
    def __init__(self, hidden_size, with_sigmoid=False): #Initialize model & define layers
        super(RelationTypeAggregator, self).__init__()
        self.linear = nn.Linear(2 * hidden_size, hidden_size)
        self.linear2 = nn.Linear(2 * hidden_size, 2 * hidden_size)
        self.with_sigmoid = with_sigmoid

    def forward(self, inputs): #Compute output of NN
        p, q = inputs
        lq = self.linear2(q)
        lp = self.linear2(p)
        mid = nn.Sigmoid()(lq + lp)
        output = p * mid + q * (1 - mid)
        output = self.linear(output)
        if self.with_sigmoid: #将值映射到(0,1)
            output = torch.sigmoid(output)
        return output
