import argparse
import os

import torch
import torchvision.utils as vutils

from models.cycleGAN import CycleGAN
from data.pairs_dataset import UnalignedDataLoader
from util.visualisations import Visualizer

parser = argparse.ArgumentParser('CycleGAN models test')
parser.add_argument('--dir_A', default='/Users/taras/datasets/horse2zebra/A', type=str)
parser.add_argument('--dir_B', default='/Users/taras/datasets/horse2zebra/B', type=str)
parser.add_argument('--width', default=512, type=int)
parser.add_argument('--height', default=512, type=int)
parser.add_argument('--load_size', default=512, type=int)
parser.add_argument('--save_path', default='./results_fused', type=str)
parser.add_argument('--shuffle', default=False, action='store_true')
parser.add_argument('--load_epoch', default=56, type=int)
parser.add_argument('--cuda', default=False, action='store_true')
parser.add_argument('--use_lsgan', default=True, type=bool)
parser.add_argument('--backward_type', default='separate')
parser.add_argument('--train', default=False, type=bool)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument('--input_nc', default=3, type=int)
parser.add_argument('--output_nc', default=3, type=int)
args = parser.parse_args()

model = CycleGAN(args)
model.load_parameters(args.load_epoch)
model.print_model_desription()
data_loader = UnalignedDataLoader(args)
dataset = data_loader.load_data()

for batch_idx, inputs in enumerate(dataset):
    model.set_inputs(inputs)
    model.test_model()

    imAB_gen_file = os.path.join(args.save_path, 'imAB_gen_{}_test.jpg'.format(batch_idx))
    vutils.save_image(model.get_AB_images_triple(), imAB_gen_file, normalize=True)
