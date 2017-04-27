import argparse
import os

import torchvision.utils as vutils

from models.cycleGAN import CycleGAN
from data.pairs_dataset import UnalignedDataLoader

parser = argparse.ArgumentParser('CycleGAN models test')
parser.add_argument('--data_root', default='/Users/taras/datasets/horse2zebra', type=str)
parser.add_argument('--width', default=1024, type=int)
parser.add_argument('--height', default=1024, type=int)
parser.add_argument('--load_size', default=1024, type=int)
parser.add_argument('--save_path', default='./results_fused', type=str)
parser.add_argument('--shuffle', default=False, action='store_true')
parser.add_argument('--load_epoch', default=99, type=int)
parser.add_argument('--cuda', default=True, action='store_true')
parser.add_argument('--num_gpu', default=1, type=int)
parser.add_argument('--use_lsgan', default=True, type=bool)
parser.add_argument('--backward_type', default='separate')
parser.add_argument('--train', default=False, type=bool)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument('--input_nc', default=3, type=int)
parser.add_argument('--output_nc', default=3, type=int)
parser.add_argument('--identity', default=0, type=int)
parser.add_argument('--num_test_iterations', default=5, type=int)
parser.add_argument('--phase', default='test', type=str)
args = parser.parse_args()

for k,v in vars(args).items():
    print('{} = {}'.format(k,v))

test_dir = os.path.join(args.save_path,'test')

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
if not os.path.exists(test_dir):
    os.makedirs(test_dir)

model = CycleGAN(args)
model.load_parameters(args.load_epoch)
model.print_model_desription()
data_loader = UnalignedDataLoader(args)
dataset = data_loader.load_data()

for batch_idx, inputs in enumerate(dataset):
    if batch_idx >= args.num_test_iterations:
        break
    model.set_inputs(inputs)
    model.test_model()

    imAB_gen_file = os.path.join(test_dir, 'imAB_gen_{}_{}_{}_test.jpg'.format(batch_idx, args.height, args.width))
    vutils.save_image(model.get_AB_images_triple(), imAB_gen_file, normalize=True)
    print('precessed item with idx: {}'.format(batch_idx))
