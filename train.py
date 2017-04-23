import argparse
import random
import os
import time

import torch.utils.data
import torchvision.utils as vutils

from models.cycleGAN import CycleGAN
from data.pairs_dataset import UnalignedDataLoader
from util.visualisations import Visualizer

parser = argparse.ArgumentParser('CycleGAN train')
parser.add_argument('--dir_A', default='/Users/taras/datasets/horse2zebra/A')
parser.add_argument('--dir_B', default='/Users/taras/datasets/horse2zebra/B')
parser.add_argument('--width', default=128, type=int)
parser.add_argument('--height', default=128, type=int)
parser.add_argument('--load_size', default=142, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--lr', default=2e-4, type=float)
parser.add_argument('--beta1', default=0.5, type=float)
parser.add_argument('--pool_size', default=50, type=int)
parser.add_argument('--input_nc', default=3, type=int)
parser.add_argument('--output_nc', default=3, type=int)
parser.add_argument('--use_lsgan', default=True, type=bool)
parser.add_argument('--train', default=True, type=bool)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--num_gpu', default=1, type=int)
parser.add_argument('--save_path', default='./results', type=str)
parser.add_argument('--save_interval', default=1000, type=int)
parser.add_argument('--vis_interval', default=500, type=int)
parser.add_argument('--log_interval', default=50, type=int)
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument('--shuffle', action='store_true')
parser.add_argument('--display_id', default=1, type=int)
parser.add_argument('--backward_type', default='separate')
parser.add_argument('--continue_epoch', default=99, type=int)
args = parser.parse_args()

try:
    os.makedirs(args.save_path)
except OSError as e:
    pass


def set_random_seed(seed, cuda):
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


set_random_seed(args.seed, args.cuda)

data_loader = UnalignedDataLoader(args)
dataset = data_loader.load_data()
visualizer = Visualizer(args)


def train():
    model = CycleGAN(params=args)
    model.train()
    if args.continue_epoch > -1:
        model.load_parameters(args.continue_epoch)

    for e in range(args.epochs):
        e_begin = time.time()
        for batch_idx, inputs in enumerate(dataset):
            model.set_inputs(inputs)
            model.optimize_parameters()
            e_fraction_passed = batch_idx * args.batch_size / len(dataset.data_loader_A)
            if batch_idx % args.log_interval == 0:
                err = model.get_errors()
                visualizer.plot_errors(err, e, e_fraction_passed)
                desc = model.get_errors_string()
                print('Epoch:[{}/{}] Batch:[{:10d}/{}] '.format(e, args.epochs,
                                                                batch_idx * args.batch_size,
                                                                len(dataset.data_loader_A)), desc)
            if batch_idx % args.vis_interval == 0:
                imAB_gen_file = os.path.join(args.save_path, 'imAB_gen_{}_{}.jpg'.format(e, batch_idx))
                vutils.save_image(model.get_AB_images_triple(), imAB_gen_file, normalize=True)
            if batch_idx % args.save_interval == 0:
                model.save_parameters(e)
        e_end = time.time()
        e_time = e_end - e_begin
        print('End of epoch [{}/{}] Time taken: {:.4f} sec.'.format(e, args.epochs, e_time))

    print('saving final model paramaters')
    model.save_parameters(args.epochs)


if __name__ == '__main__':
    train()
