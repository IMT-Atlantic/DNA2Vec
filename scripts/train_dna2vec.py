#!/usr/bin/env python3

### 官方卷 通用
import sys
sys.path.extend(['.', '..'])
# 路径模式匹配
import glob
# 日志记录
import logbook
from logbook.compat import redirect_logging
# 增强了argparse，支持从配置文件或命令行读取参数
import configargparse
import numpy as np
# 第三方库Biopython，用于生物信息序列的处理和解析
from Bio import SeqIO
# 第三方库gensim，用于自然语言处理中特别是词向量的生成
from gensim.models import word2vec

### 自定义卷
# 自定义模块，用于基准测试时间
from attic_util.time_benchmark import Benchmark
# 自定义模块，工具函数
from attic_util import util
# 自定义模块，用于日志的多路传输
from attic_util.tee import Tee
# 自定义模块，用于生成和处理k-mer直方图
from dna2vec.histogram import Histogram
# 自定义模块，DNA序列生成和处理
from dna2vec.generators import SeqGenerator, KmerSeqIterable, SeqMapper, SeqFragmenter
# 自定义模块，不同的k-mer分片方法
from dna2vec.generators import DisjointKmerFragmenter, SlidingKmerFragmenter

# 无效参数处理
class InvalidArgException(Exception):
    pass

class Learner:
    # 初始化对象 设置参数并记录
    def __init__(self, out_fileroot, context_halfsize, gensim_iters, vec_dim):
        self.logger = logbook.Logger(self.__class__.__name__)
        assert(word2vec.FAST_VERSION >= 0)
        self.logger.info('word2vec.FAST_VERSION (should be >= 0): {}'.format(word2vec.FAST_VERSION))
        self.model = None
        self.out_fileroot = out_fileroot
        self.context_halfsize = context_halfsize
        self.gensim_iters = gensim_iters
        self.use_skipgram = 1
        self.vec_dim = vec_dim

        self.logger.info('Context window half size: {}'.format(self.context_halfsize))
        self.logger.info('Use skipgram: {}'.format(self.use_skipgram))
        self.logger.info('gensim_iters: {}'.format(self.gensim_iters))
        self.logger.info('vec_dim: {}'.format(self.vec_dim))

    # 生成序列数据训练word2vec模型 kmer作为单词进行训练
    # kmer作为输入进行训练
    def train(self, kmer_seq_generator):
        self.model = word2vec.Word2Vec(
            sentences=kmer_seq_generator,
            vector_size=self.vec_dim,
            window=self.context_halfsize,
            min_count=5,
            workers=4,
            sg=self.use_skipgram,
            epochs=self.gensim_iters)

        # self.logger.info(model.vocab)

    # 写入权重 就是我们后来加载的那个
    def write_vec(self):
        out_filename = '{}.w2v'.format(self.out_fileroot)
        self.model.wv.save_word2vec_format(out_filename, binary=False)

def run_main(args, inputs, out_fileroot):
    logbook.info(' '.join(sys.argv))
    # 非调试情况下日志级别设计为INFO
    if not args.debug:
        import logging
        logging.getLogger('gensim.models.word2vec').setLevel(logging.INFO)

    # 设计随机数种子
    np.random.seed(args.rseed)

    # 创建基准测试对象 用于时间测量
    benchmark = Benchmark()

    # 基于前置参数进行kmer长度的选择 因为我们这里的参数设计是一个动态的过程
    # 或许可以考虑把参数变成一个可以学习的过程 当检查到有合适的模式的时候改变成更优秀的权重体系？
    if args.kmer_fragmenter == 'disjoint':
        kmer_fragmenter = DisjointKmerFragmenter(args.k_low, args.k_high)
    elif args.kmer_fragmenter == 'sliding':
        kmer_fragmenter = SlidingKmerFragmenter(args.k_low, args.k_high)
    else:
        raise InvalidArgException('Invalid kmer fragmenter: {}'.format(args.kmer_fragmenter))

    logbook.info('kmer fragmenter: {}'.format(args.kmer_fragmenter))

    # 创建直方图对象 用于统计
    histogram = Histogram()
    # 原始module在dna2vec.generators
    kmer_seq_iterable = KmerSeqIterable(
        args.rseed_trainset,
        SeqGenerator(inputs, args.epochs), # 序列生成器
        SeqMapper(), # 序列映射器
        SeqFragmenter(), # 序列分片器
        kmer_fragmenter, # k-mer分片器
        histogram, # 用于统计k-mer分布的直方图
    )

    # 创建kmer生成器 使用输入的文件、映射器、分片器和直方图对象
    learner = Learner(out_fileroot, args.context, args.gensim_iters, args.vec_dim)
    learner.train(kmer_seq_iterable)
    learner.write_vec()

    # 打印kmer统计信息
    histogram.print_stat(sys.stdout)

    # 打印运行时长基准数据
    benchmark.print_time()

def main():
    # 命令行参数解析 使用configargparse解析命令行参数
    argp = configargparse.get_argument_parser()
    argp.add('-c', is_config_file=True, help='config file path')
    argp.add_argument('--kmer-fragmenter', help='disjoint or sliding', choices=['disjoint', 'sliding'], default='sliding')
    argp.add_argument('--vec-dim', help='vector dimension', type=int, default=12)
    argp.add_argument('--rseed', help='general np.random seed', type=int, default=7)
    argp.add_argument('--rseed-trainset', help='random seed for generating training data', type=int, default=123)
    argp.add_argument('--inputs', help='FASTA files', nargs='+', required=True)
    argp.add_argument('--k-low', help='k-mer start range (inclusive)', type=int, default=5)
    argp.add_argument('--k-high', help='k-mer end range (inclusive)', type=int, default=5)
    argp.add_argument('--context', help='half size of context window (the total size is 2*c+1)', type=int, default=4)
    argp.add_argument('--epochs', help='number of epochs', type=int, default=1)
    argp.add_argument('--gensim-iters', help="gensim's internal iterations", type=int, default=1)
    argp.add_argument('--out-dir', help="output directory", default='../dataset/dna2vec/results')
    argp.add_argument('--debug', help='', action='store_true')
    args = argp.parse_args()

    # 根据是否启用调试模式，设置输出目录和日志级别
    if args.debug:
        out_dir = '/tmp'
        log_level = 'DEBUG'
    else:
        out_dir = args.out_dir
        log_level = 'INFO'

    # 处理输入文件，支持通配符展开，将匹配到的文件名列表添加到inputs中
    inputs = []
    for s in args.inputs:
        inputs.extend(list(glob.glob(s)))

    # 计算输入文件的大小并生成输出文件的根路径
    mbytes = util.estimate_bytes(inputs) // (10 ** 6)
    out_fileroot = util.get_output_fileroot(
        out_dir,
        'dna2vec',
        'k{}to{}-{}d-{}c-{}Mbp-{}'.format(
            args.k_low,
            args.k_high,
            args.vec_dim,
            args.context,
            mbytes * args.epochs,  # total Mb including epochs
            args.kmer_fragmenter))


    # 打开一个文本文件用于保存摘要信息，并将标准输出重定向到文件中记录日志
    # 调用run_main函数执行主要流程
    out_txt_filename = '{}.txt'.format(out_fileroot)
    with open(out_txt_filename, 'w') as summary_fptr:
        with Tee(summary_fptr):
            logbook.StreamHandler(sys.stdout, level=log_level).push_application()
            redirect_logging()
            run_main(args, inputs, out_fileroot)

if __name__ == '__main__':
    main()
