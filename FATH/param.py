import argparse

from utils import *

def parse_args():
    desc = 'Few shot adversarial learning of realistic neural talking head models'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--dataset', type=str, default='voxceleb', help='dataset name')
    
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', help='Directory name of the checkpoints')
    parser.add_argument('--sample_dir', type=str, default='samples', help='Directory name to save the samples on training')
    parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory name to save training logs')

    parser.add_argument('--epoch', type=int, default=10, help='the number of epochs to run')
    parser.add_argument('--iteration', type=int, default=10000, help='the number of training iterations')
    parser.add_argument('--batch_size', type=int, default=32, help='The number of batch size')
    parser.add_argument('--print_freq', type=int, default=500, help='The number of image_print_freq')
    parser.add_argument('--save_freq', type=int, default=500, help='The number of chpt_save_freq')

    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--c_dim', type=int, default=3, help='The dimention of channel')
    parser.add_argument('--e_dim', type=int, default=512, help='The dimention of embedding')
    parser.add_argument('--k', type=int, default=9, help='The number of sampling frame in video')
    parser.add_argument('--n_video', type=int, default=8, help='The number of index in dataset')

    parser.add_argument('--g_lr', type=float, default=0.00005)
    parser.add_argument('--d_lr', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.0)
    parser.add_argument('--beta2', type=float, default=0.9)

    return check_args(parser.parse_args())

def check_args(args):
    check_folder(args.checkpoint_dir)
    check_folder(args.sample_dir)
    check_folder(args.result_dir)
    check_folder(args.log_dir)

    try:
        assert args.epoch >= 1
    except:
        print('number of epoch error')

    try:
        assert args.batch_size >= 1
    except:
        print('batch size error')

    return args
