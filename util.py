import numpy as np
import random
import torch
import time

def list2tuple(l):
    return tuple(list2tuple(x) if type(x)==list else x for x in l)

def tuple2list(t):
    return list(tuple2list(x) if type(x)==tuple else x for x in t)


"""
    将若干个tuple包裹的数据转化为一个list
    eg:
        x = (((1,2,3)),((4,5,6)),((7,8,9)))
        print(flatten(x))
        [1, 2, 3, 4, 5, 6, 7, 8, 9]
    (7330, (38,)) --> [7330, 38]
    这个匿名函数很高级！！！
"""
flatten=lambda l: sum(map(flatten, l),[]) if isinstance(l,tuple) else [l]

# 返回当前时间
def parse_time():
    return time.strftime("%Y.%m.%d-%H:%M:%S", time.localtime())

#设置随机种子，方便下次复现实验结果
def set_global_seed(seed):
    torch.manual_seed(seed) #设置CPU生成随机数的种子
    torch.cuda.manual_seed(seed) #为特定GPU设置种子，生成随机数
    '''
        为所有GPU设置种子，生成随机数：
        # 如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。
        torch.cuda.manual_seed_all(number)
    '''
    np.random.seed(seed) #为numpy指定随机种子
    random.seed(seed) #指定python/random包的随机种子
    #参考:https://www.pudn.com/news/6231e44134f9fa732f38a51a.html
    torch.backends.cudnn.deterministic=True #使torch中模型输入相同时，输出也相同，(即算法保持一致)方便复现

'''
    将一个包含tuple的字符串转化为tuple
'''
def eval_tuple(arg_return):
    """Evaluate a tuple string into a tuple."""
    if type(arg_return) == tuple:
        return arg_return
    if arg_return[0] not in ["(", "["]: #字符串""包裹的情况
        arg_return = eval(arg_return) #eval()函数用来执行一个字符串表达式，并返回表达式的值。
    else:
        splitted = arg_return[1:-1].split(",")
        List = []
        for item in splitted:
            try:
                item = eval(item)
            except:
                pass
            if item == "":
                continue
            List.append(item)
        arg_return = tuple(List)
    return arg_return

# 将<query_structure,query> 变为 [(query, query_structure),(query, query_structure),...]
def flatten_query(queries):
    all_queries = []
    for query_structure in queries:
        tmp_queries = list(queries[query_structure])
        all_queries.extend([(query, query_structure) for query in tmp_queries])
    return all_queries