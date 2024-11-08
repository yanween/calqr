#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader import TestDataset, TrainDataset, SingledirectionalOneShotIterator
from tensorboardX import SummaryWriter
import pickle
from collections import defaultdict

from models import KGReasoning
from util import flatten_query, parse_time, set_global_seed, eval_tuple

# import os
# os.environ['CUDA_VISIBLE_DEVICES']='4'

query_name_dict = {('e',('r',)): '1p',
                    ('e', ('r', 'r')): '2p',
                    ('e', ('r', 'r', 'r')): '3p',
                    ('e', ('r', 'r', 'r', 'r')): '4p',
                    ('e', ('r', 'r', 'r', 'r', 'r')): '5p',
                    (('e', ('r',)), ('e', ('r',))): '2i',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                    ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                    (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                    (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                    ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
                    (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
                    (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
                    (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
                    ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
                    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
                    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM'
                }
name_query_dict = {value: key for key, value in query_name_dict.items()}
all_tasks = list(name_query_dict.keys()) # ['1p', '2p', '3p', '4p', '5p']

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU', default=True)

    parser.add_argument('--do_train', action='store_true', help="do train", default=True)
    parser.add_argument('--do_valid', action='store_true', help="do valid", default=True)
    parser.add_argument('--do_test', action='store_true', help="do test", default=True)

    parser.add_argument('--data_path', type=str, default='./data/FB15k-237-long_chain', help="KG data path")
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int, help="negative entities sampled per query")
    parser.add_argument('-d', '--hidden_dim', default=400, type=int, help="embedding dimension")
    """
        什么是margin in the loss
        gamma是具体用在KGReasoning中的
    """
    parser.add_argument('-g', '--gamma', default=0.375, type=float, help="margin in the loss")
    parser.add_argument('-b', '--batch_size', default=512, type=int, help="batch size of queries")

    parser.add_argument('--drop', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--test_batch_size', default=16, type=int, help='valid/test batch size')
    parser.add_argument('-plr', '--pre_train_learning_rate', default=0.0001, type=float)
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=1, type=int, help="used to speed up torch.dataloader")
    parser.add_argument('-save', '--save_path', default='./logs/FB15k-237/gqe_baseline_test', type=str, help="no need to set manually, will configure automatically")
    parser.add_argument('--max_steps', default=300000, type=int, help="maximum iterations to train")

    parser.add_argument('--warm_up_steps', default=None, type=int, help="no need to set manually, will configure automatically")
    parser.add_argument('--save_checkpoint_steps', default=10000, type=int, help="save checkpoints every xx steps")
    parser.add_argument('--valid_steps', default=10000, type=int, help="evaluate validation queries every xx steps")
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')


    parser.add_argument('--geo', default='logic', type=str, choices=['vec', 'box', 'beta', 'cone', 'logic', 'mlp', 'mlpMixer'], help='the reasoning model, vec for GQE, box for Query2box, beta for BetaE')
    parser.add_argument('--print_on_screen', action='store_true', default=True)



    parser.add_argument('--tasks', default='1p.2p.3p.4p.5p', type=str, help="tasks connected by dot, refer to the BetaE paper for detailed meaning and structure of each task")
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('-betam', '--beta_mode', default="(1600,2)", type=str, help='(hidden_dim,num_layer) for BetaE relational projection')

    parser.add_argument('-boxm', '--box_mode', default="(none,0.02)", type=str, help='(offset activation,center_reg) for Query2box, center_reg balances the in_box dist and out_box dist')
    parser.add_argument('-cenr', '--center_reg', default=0.02, type=float, help='center_reg for ConE, center_reg balances the in_cone dist and out_cone dist')
    parser.add_argument('-logicm', '--logic_mode', default="(luk,1,1,0,1600,2)", type=str, help='(tnorm,bounded,use_att,use_gtrans,hidden_dim,num_layer)')
    parser.add_argument(
        "-nlayers",
        "--mlp_mode",
        default=1,
        type=int,
        help="Number of layers (before last) for the neural networs",
    )
    parser.add_argument('--prefix', default=None, type=str, help='prefix of the log path')
    parser.add_argument('--checkpoint_path', default=None, type=str, help='path for loading the checkpoints')
    parser.add_argument('-evu', '--evaluate_union', default="DNF", type=str, choices=['DNF', 'DM'], help='the way to evaluate union queries, transform it to disjunctive normal form (DNF) or use the De Morgan\'s laws (DM)')

    parser.add_argument('--model_mode', default="baseline", type=str, choices=['baseline', 'temp'], help='the type of model')
    """
        faithful是指什么？--表示train数据是否包含valid和test数据
    """
    parser.add_argument('--faithful', default="no_faithful", type=str, choices=['faithful', 'no_faithful'], help='faithful or not')
    parser.add_argument('--neighbor_ent_type_samples', type=int, default=32, help='number of sampled entity type neighbors')
    parser.add_argument('--neighbor_rel_type_samples', type=int, default=64, help='number of sampled relation type neighbors')

    parser.add_argument('--negation_mode', default="baseline", type=str, choices=['baseline', 'enhance'], help='the type of negation')
    parser.add_argument('--negation_together_enhance', default="together", type=str, choices=['together', 'separate'], help='the type of negation enhance')
    parser.add_argument('--enhance_mode', default="domain_only", type=str, choices=['domain_only', 'domain_range', 'domain_range_plus_origin'], help='the type of query enhance')

    parser.add_argument('--range_enhance_mode', default=None, type=str, choices=['attention_weight', 'None'], help='the type of range enhance')

    parser.add_argument('--do_pretrain', action='store_true', help="do pretrain", default=False)
    parser.add_argument('--do_prevalid', action='store_true', help="do prevalid", default=False)
    parser.add_argument('--do_pretest', action='store_true', help="do pretest", default=False)
    parser.add_argument('--pre_train_max_steps', default=500000, type=int, help="maximum iterations to pre_train")
    return parser.parse_args(args)

def save_model(model, optimizer, save_variable_list, args, save_file = 'checkpoint'):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)


    save_path = os.path.join(args.save_path, save_file)
    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        save_path
    )


def set_logger(args):
    '''
    Write logs to console and log file
    '''
    if args.do_train:
        log_file = os.path.join(args.save_path, 'train.log')
    else:
        log_file = os.path.join(args.save_path, 'test.log')

    """
        logging由logger，handler，filter，formater
        logger是提供我们记录日志的方法；handler是让我们选择日志的输出地方，如：控制台，文件，邮件发送等，一个logger添加多个handler；filter是给用户提供更加细粒度的控制日志的输出内容；formater用户格式化输出日志的信息。
        logging参考链接：http://t.csdn.cn/2xocp
    """

    logging.basicConfig(


        format='%(asctime)s %(levelname)-8s %(message)s',

        level=logging.INFO,

        datefmt='%Y-%m-%d %H:%M:%S',

        filename=log_file,

        filemode='a+'
    )

    if args.print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)



def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    metrics: {}
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))


def evaluate(model, tp_answers, fn_answers, args, dataloader, query_name_dict, mode, step, writer):
    '''
    Evaluate queries in dataloader
    :param model: model
    :param tp_answers: valid_easy_answers
    :param fn_answers: valid_hard_answers
    :param args: args
    :param dataloader: valid_dataloader
    :param query_name_dict: query_name_dict
    :param mode: 'Valid'
    :param step: step
    :param writer:
    :return: all_metrics: dict{'average_metric': float number}
    '''
    average_metrics = defaultdict(float)
    all_metrics = defaultdict(float)


    metrics = model.test_step(model, tp_answers, fn_answers, args, dataloader, query_name_dict)
    num_query_structures = 0
    num_queries = 0
    for query_structure in metrics:
        log_metrics(mode+" "+query_name_dict[query_structure], step, metrics[query_structure])
        for metric in metrics[query_structure]:
            writer.add_scalar("_".join([mode, query_name_dict[query_structure], metric]), metrics[query_structure][metric], step)
            all_metrics["_".join([query_name_dict[query_structure], metric])] = metrics[query_structure][metric]
            if metric != 'num_queries':
                average_metrics[metric] += metrics[query_structure][metric]
        num_queries += metrics[query_structure]['num_queries']
        num_query_structures += 1

    for metric in average_metrics:
        average_metrics[metric] /= num_query_structures
        writer.add_scalar("_".join([mode, 'average', metric]), average_metrics[metric], step)
        all_metrics["_".join(["average", metric])] = average_metrics[metric]
    log_metrics('%s average'%mode, step, average_metrics)

    return all_metrics



def load_data(args, tasks):
    logging.info("loading data")

    train_queries = pickle.load(open(os.path.join(args.data_path, "train-queries.pkl"), 'rb'))
    train_answers = pickle.load(open(os.path.join(args.data_path, "train-answers.pkl"), 'rb'))
    valid_queries = pickle.load(open(os.path.join(args.data_path, "valid-queries.pkl"), 'rb'))
    valid_hard_answers = pickle.load(open(os.path.join(args.data_path, "valid-hard-answers.pkl"), 'rb'))
    valid_easy_answers = pickle.load(open(os.path.join(args.data_path, "valid-easy-answers.pkl"), 'rb'))
    test_queries = pickle.load(open(os.path.join(args.data_path, "test-queries.pkl"), 'rb'))
    test_hard_answers = pickle.load(open(os.path.join(args.data_path, "test-hard-answers.pkl"), 'rb'))
    test_easy_answers = pickle.load(open(os.path.join(args.data_path, "test-easy-answers.pkl"), 'rb'))


    if args.faithful == 'faithful':

        for queries in [valid_queries, test_queries]:
            for query_structure in queries:
                train_queries[query_structure] |= queries[query_structure]

        for answers in [valid_hard_answers, valid_easy_answers, test_hard_answers, test_easy_answers]:
            for query in answers:
                train_answers.setdefault(query, set())
                train_answers[query] |= answers[query]


    """
        这里的evaluate_union是在干嘛？
        query_structure就是main.py一开始定义的name_query_dict的value：
        即query_name_dict中的key
        query_name_dict = {('e',('r',)): '1p', 
                ('e', ('r', 'r')): '2p',
                ('e', ('r', 'r', 'r')): '3p',
                ('e', ('r', 'r', 'r', 'r')): '4p',
                ('e', ('r', 'r', 'r', 'r', 'r')): '5p',
                (('e', ('r',)), ('e', ('r',))): '2i',
                (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
                (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
                (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
                (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
                ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
                ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
                ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM'
            } 
    """
    for name in all_tasks:
        if 'u' in name:
            name, evaluate_union = name.split('-')
        else:
            evaluate_union = args.evaluate_union


        if name not in tasks or evaluate_union != args.evaluate_union:


            query_structure = name_query_dict[name if 'u' not in name else '-'.join([name, evaluate_union])]
            if query_structure in train_queries:
                del train_queries[query_structure]
            if query_structure in valid_queries:
                del valid_queries[query_structure]
            if query_structure in test_queries:
                del test_queries[query_structure]

    """
        queries的data_structure: dict{<class 'tuple'>,<class 'set'>}
        eg:
            ('e', ('r',)) :  {(3292, (144,)), (3147, (34,))......}
            即dict{query_structure,query : {query1,query2,...}}

        answers的data_structure: dict{<class 'tuple'>,<class 'set'>}
        eg: 
            (7330, (38,)) :  {8936, 3250, 2773}
            即dict{query : {answer1,answer2,...}}  
    """
    return train_queries, train_answers, valid_queries, valid_hard_answers, valid_easy_answers, test_queries, test_hard_answers, test_easy_answers


def main(args):
    set_global_seed(args.seed)

    tasks = args.tasks.split('.')
    for task in tasks:
        if 'n' in task and args.geo in ['box', 'vec']:
            assert False, "Q2B and GQE cannot handle queries with negation"
    if args.evaluate_union == 'DM':
        assert args.geo == 'beta' or args.geo == 'cone' or args.geo == 'logic', "only BetaE supports modeling union using De Morgan's Laws"

    cur_time = parse_time()
    if args.prefix is None:
        prefix = 'logs'
    else:
        prefix = args.prefix


    print ("overwritting args.save_path")
    args.save_path = os.path.join(args.save_path, args.tasks, args.geo)
    if args.geo in ['box']:
        tmp_str = "g-{}-mode-{}".format(args.gamma, args.box_mode)
    elif args.geo in ['vec']:
        tmp_str = "g-{}".format(args.gamma)
    elif args.geo == 'beta':
        tmp_str = "g-{}-mode-{}".format(args.gamma, args.beta_mode)
    elif args.geo == 'cone':
        tmp_str = "g-{}-mode-{}".format(args.gamma, args.center_reg)
    elif args.geo == 'logic':
        tmp_str = "g-{}-mode-{}".format(args.gamma, args.logic_mode)
    elif args.geo == "mlp":
        tmp_str = "g-{}-layers-{}".format(args.gamma, args.mlp_mode)
    elif args.geo == "mlpMixer":
        tmp_str = "g-{}-layers-{}".format(args.gamma, args.mlp_mode)


    if args.checkpoint_path is not None:
        args.save_path = args.checkpoint_path

    else:
        args.save_path = os.path.join(args.save_path, tmp_str, cur_time)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    print("logging to", args.save_path)
    if not args.do_train:
        writer = SummaryWriter('./logs-debug/unused-tb')
    else:
        writer = SummaryWriter(args.save_path)
    set_logger(args)


    with open('%s/stats.txt'%args.data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])
        ntype = int(entrel[2].split(' ')[-1])

    args.nentity = nentity
    args.nrelation = nrelation


    logging.info('-------------------------------'*3)
    logging.info('Geo: %s' % args.geo)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)
    logging.info('#max steps: %d' % args.max_steps)
    logging.info('Evaluate unoins using: %s' % args.evaluate_union)
    logging.info('negation_mode: %s' % args.negation_mode)
    logging.info('negation_together_enhance: %s' % args.negation_together_enhance)
    logging.info('enhance_mode: %s' % args.enhance_mode)
    logging.info('range_enhance_mode: %s' % args.range_enhance_mode)

    train_queries, train_answers, valid_queries, valid_hard_answers, valid_easy_answers, test_queries, test_hard_answers, test_easy_answers = load_data(args, tasks)

    logging.info("Training info:")

    if args.do_train:
        for query_structure in train_queries:
            logging.info(query_name_dict[query_structure]+": "+str(len(train_queries[query_structure])))
        """
            defaultdict是属于collections 模块下的一个工厂函数，用于构建字典对象，接收一个函数（可调用）对象为作为参数。参数返回的类型是什么，key对应value就是什么类型。
            即设定dict{}中value类型
            且通过访问dict[key]时若key不存在于dict中则返回默认值
            默认值和设定的数据类型有关，具体如下：
            int:0
            set: set()
            str: ''
            list: []
        """
        train_path_queries = defaultdict(set)
        train_other_queries = defaultdict(set)

        path_list = ['1p', '2p', '3p', '4p', '5p']
        for query_structure in train_queries:
            if query_name_dict[query_structure] in path_list:

                train_path_queries[query_structure] = train_queries[query_structure]
            else:
                train_other_queries[query_structure] = train_queries[query_structure]
        train_path_queries = flatten_query(train_path_queries)

        train_path_iterator = SingledirectionalOneShotIterator(DataLoader(
                                    TrainDataset(train_path_queries, nentity, nrelation, args.negative_sample_size, train_answers),
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=args.cpu_num,
                                    collate_fn=TrainDataset.collate_fn
                                ))

        if len(train_other_queries) > 0:
            train_other_queries = flatten_query(train_other_queries)
            train_other_iterator = SingledirectionalOneShotIterator(DataLoader(
                                        TrainDataset(train_other_queries, nentity, nrelation, args.negative_sample_size, train_answers),
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=args.cpu_num,
                                        collate_fn=TrainDataset.collate_fn
                                    ))
        else:
            train_other_iterator = None

    if args.do_pretrain:
        train_1p_queries = defaultdict(set)
        train_1p_queries[('e', ('r',))] = train_queries[('e', ('r',))]
        train_1p_queries = flatten_query(train_1p_queries)


        train_1p_iterator = SingledirectionalOneShotIterator(DataLoader(
            TrainDataset(train_1p_queries, nentity, nrelation, args.negative_sample_size, train_answers),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.cpu_num,
            collate_fn=TrainDataset.collate_fn
        ))




    logging.info("Validation info:")
    if args.do_valid:
        for query_structure in valid_queries:
            logging.info(query_name_dict[query_structure]+": "+str(len(valid_queries[query_structure])))
        valid_queries = flatten_query(valid_queries)
        valid_dataloader = DataLoader(
            TestDataset(
                valid_queries,
                args.nentity,
                args.nrelation,
            ),
            batch_size=args.test_batch_size,
            num_workers=args.cpu_num,
            collate_fn=TestDataset.collate_fn
        )


    logging.info("Test info:")
    if args.do_test:
        for query_structure in test_queries:
            logging.info(query_name_dict[query_structure]+": "+str(len(test_queries[query_structure])))
        test_queries = flatten_query(test_queries)
        test_dataloader = DataLoader(
            TestDataset(
                test_queries,
                args.nentity,
                args.nrelation,
            ),
            batch_size=args.test_batch_size,
            num_workers=args.cpu_num,
            collate_fn=TestDataset.collate_fn
        )
    entity2type, relation2type = build_kg(args.data_path, args.neighbor_ent_type_samples, args.neighbor_rel_type_samples)

    model = KGReasoning(
        nentity=nentity,
        nrelation=nrelation,
        ntype=ntype,
        entity2type=entity2type,
        relation2type=relation2type,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        geo=args.geo,
        use_cuda = args.cuda,
        box_mode=eval_tuple(args.box_mode),
        beta_mode = eval_tuple(args.beta_mode),
        center_reg=args.center_reg,
        logic_mode=eval_tuple(args.logic_mode),
        mlp_mode=args.mlp_mode,
        model_mode = args.model_mode,
        test_batch_size=args.test_batch_size,
        query_name_dict = query_name_dict,
        drop=args.drop,
        neighbor_ent_type_samples=args.neighbor_ent_type_samples,
        neighbor_rel_type_samples=args.neighbor_rel_type_samples,
        negation_mode=args.negation_mode,
        negation_together_enhance=args.negation_together_enhance,
        enhance_mode=args.enhance_mode,
        range_enhance_mode=args.range_enhance_mode
    )

    logging.info('Model Parameter Configuration:')
    num_params = 0
    for name, param in model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
        if param.requires_grad:
            num_params += np.prod(param.size())
    logging.info('Parameter Number: %d' % num_params)


    if args.cuda:
        model = model.cuda()


    if args.do_pretrain:
        current_learning_rate = args.pre_train_learning_rate
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=current_learning_rate
        )

        warm_up_steps = args.pre_train_max_steps // 2
        logging.info('Start PreTraining...')
        logging.info('pretrain_learning_rate = %f' % current_learning_rate)
        logging.info('pre_train_max_steps = %d' % args.pre_train_max_steps)
        logging.info('batch_size = %d' % args.batch_size)
        logging.info('hidden_dim = %d' % args.hidden_dim)
        logging.info('gamma = %f' % args.gamma)
        pre_training_logs = []

        max_1p_valid_HITS3 = 0.0
        for step in range(0, args.pre_train_max_steps):






            log = model.train_step(model, optimizer, train_1p_iterator, args, step)
            for metric in log:
                writer.add_scalar('path_' + metric, log[metric], step)

            pre_training_logs.append(log)


            if step >= warm_up_steps:

                current_learning_rate = current_learning_rate / 5
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 1.5


            if step % args.valid_steps == 0 and step > 0:
                if args.do_prevalid:
                    logging.info('Pretrain model Evaluating on Valid Dataset...')

                    valid_all_metrics = evaluate(model, valid_easy_answers, valid_hard_answers, args,
                                                 valid_dataloader, query_name_dict, 'Valid', step, writer)


                    wanted_metric = 'HITS3'

                    if valid_all_metrics[
                        "_".join(["1p", wanted_metric])] > max_1p_valid_HITS3:
                        save_variable_list = {
                            'step': step,
                            'current_learning_rate': current_learning_rate,
                            'warm_up_steps': warm_up_steps
                        }
                        save_file = 'pretrain_optimal_checkpoint'
                        save_model(model, optimizer, save_variable_list, args, save_file)

                        max_1p_valid_HITS3 = valid_all_metrics["_".join(["1p", wanted_metric])]

                    if args.do_pretest:
                        logging.info('Pretrain model Evaluating on Test Dataset...')
                        test_all_metrics = evaluate(model, test_easy_answers, test_hard_answers, args,
                                                    test_dataloader,
                                                    query_name_dict, 'Test', step, writer)


            if step % args.log_steps == 0:
                metrics = {}
                for metric in pre_training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in pre_training_logs]) / len(pre_training_logs)

                log_metrics('Training average', step, metrics)
                pre_training_logs = []


    if args.do_pretest:
        logging.info('Evaluating on Test Dataset...')
        test_all_metrics = evaluate(model, test_easy_answers, test_hard_answers, args, test_dataloader,
                                    query_name_dict, 'Test', step, writer)

    logging.info("PreTraining finished!!")


    if args.do_train:

        current_learning_rate = args.learning_rate

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=current_learning_rate
        )

        warm_up_steps = args.max_steps // 2


    if args.checkpoint_path is not None:
        logging.info('Loading checkpoint %s...' % args.checkpoint_path)
        checkpoint = torch.load(os.path.join(args.checkpoint_path, 'checkpoint'))
        init_step = checkpoint['step']
        model.load_state_dict(checkpoint['model_state_dict'])

        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.geo)
        init_step = 0

    step = init_step
    if args.geo == 'box':
        logging.info('box mode = %s' % args.box_mode)
    elif args.geo == 'beta':
        logging.info('beta mode = %s' % args.beta_mode)
    elif args.geo == 'cone':
        logging.info('cone mode = %s' % args.center_reg)
    elif args.geo == 'logic':
        logging.info('logic mode = %s (tnorm,bounded,use_att,use_gtrans,hidden_dim,num_layer)' % args.logic_mode)
    logging.info('tasks = %s' % args.tasks)
    logging.info('init_step = %d' % init_step)
    if args.do_train:
        logging.info('Start Training...')
        logging.info('learning_rate = %f' % current_learning_rate)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('finetune_train_max_steps = %d' % args.max_steps)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)

    if args.do_train:
        training_logs = []

        max_average_valid_HITS3 = 0.0
        for step in range(init_step, args.max_steps):


            if step == 2*args.max_steps//3:
                args.valid_steps *= 4

            log = model.train_step(model, optimizer, train_path_iterator, args, step)
            for metric in log:
                writer.add_scalar('path_'+metric, log[metric], step)
            if train_other_iterator is not None:
                log = model.train_step(model, optimizer, train_other_iterator, args, step)
                for metric in log:
                    writer.add_scalar('other_'+metric, log[metric], step)
                log = model.train_step(model, optimizer, train_path_iterator, args, step)

            training_logs.append(log)


            if step >= warm_up_steps:

                current_learning_rate = current_learning_rate / 5
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 1.5


            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step,
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(model, optimizer, save_variable_list, args)


            if step % args.valid_steps == 0 and step > 0:
                if args.do_valid:
                    logging.info('Evaluating on Valid Dataset...')

                    valid_all_metrics = evaluate(model, valid_easy_answers, valid_hard_answers, args, valid_dataloader, query_name_dict, 'Valid', step, writer)


                    wanted_metric = 'HITS3'

                    if valid_all_metrics["_".join(["average", wanted_metric])] > max_average_valid_HITS3:
                        save_variable_list = {
                            'step': step,
                            'current_learning_rate': current_learning_rate,
                            'warm_up_steps': warm_up_steps
                        }
                        save_file = 'optimal_checkpoint'
                        save_model(model, optimizer, save_variable_list, args, save_file)

                        max_average_valid_HITS3 = valid_all_metrics["_".join(["average", wanted_metric])]

                if args.do_test:
                    logging.info('Evaluating on Test Dataset...')
                    test_all_metrics = evaluate(model, test_easy_answers, test_hard_answers, args, test_dataloader, query_name_dict, 'Test', step, writer)


            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)

                log_metrics('Training average', step, metrics)
                training_logs = []


        save_variable_list = {
            'step': step,
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(model, optimizer, save_variable_list, args)

    try:
        print (step)
    except:
        step = 0

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        test_all_metrics = evaluate(model, test_easy_answers, test_hard_answers, args, test_dataloader, query_name_dict, 'Test', step, writer)

    logging.info("Training finished!!")


"""
    返回 entity2types, relation2types，即entity和relation的type属性
    entity2types: 存放所有entity的type,每个entity存放neighbor_ent_type_samples个type
    [
        [entity_type,entity_type,...],   
        [entity_type,entity_type,...],
        ...
    ]
    
    relation2types: 存放所有relation的type,每个relation存放neighbor_rel_type_samples个type
    [
        [relation_type,relation_type,...],   
        [relation_type,relation_type,...],
        ...
    ]
"""
def build_kg(data_path, neighbor_ent_type_samples, neighbor_rel_type_samples):
    entity_type_mapping = np.load(data_path + '/entity_type.npy', allow_pickle=True)

    entity2types = []
    for i in range(len(entity_type_mapping)):
        sampled_types = np.random.choice(entity_type_mapping[i], size=neighbor_ent_type_samples,

                                             replace=len(entity_type_mapping[i]) < neighbor_ent_type_samples)
        entity2types.append(sampled_types)

    relation_type_mapping = np.load(data_path + '/relation_type.npy', allow_pickle=True)

    relation2types = []
    for i in range(len(relation_type_mapping)):
        sampled_types = np.random.choice(relation_type_mapping[i], size=neighbor_rel_type_samples,
                                         replace=len(relation_type_mapping[i]) < neighbor_rel_type_samples)
        relation2types.append(sampled_types)
    return entity2types, relation2types


if __name__ == '__main__':
    main(parse_args())