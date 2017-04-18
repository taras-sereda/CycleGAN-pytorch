import argparse
import random
import os
import time

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.utils as vutils
import matplotlib.pyplot as plt

from models.cycleGAN import CycleGAN

parser = argparse.ArgumentParser('CycleGAN')
parser.add_argument('--dataset_dir_A', default='./data')
parser.add_argument('--dataset_dir_B', default='./data')
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

dataset_A = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.dataset_dir_A,
                         transform=transforms.Compose([
                             transforms.Scale(size=(args.load_size, args.load_size)),
                             transforms.RandomCrop(size=(args.height, args.width)),
                             transforms.ToTensor()
                         ])),
    num_workers=args.num_workers,
    shuffle=True)

dataset_B = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.dataset_dir_B,
                         transform=transforms.Compose([
                             transforms.Scale(size=(args.load_size, args.load_size)),
                             transforms.RandomCrop(size=(args.height, args.width)),
                             transforms.ToTensor()
                         ])),
    num_workers=args.num_workers,
    shuffle=True)


def show(img):
    npimg = img.numpy()
    plt.imshow(npimg.transpose(1, 2, 0))


def train():
    model = CycleGAN(params=args)
    model.train()
    for e in range(args.epochs):
        e_begin = time.time()
        for batch_idx, (dataA, dataB) in enumerate(zip(dataset_A, dataset_B)):
            model.netG_A.zero_grad()
            real_A_cpu, _ = dataA
            real_B_cpu, _ = dataB
            model.real_A.data.resize_(real_A_cpu.size()).copy_(real_A_cpu)
            model.real_B.data.resize_(real_B_cpu.size()).copy_(real_B_cpu)
            model.optimizeParameters()

            if batch_idx % args.log_interval == 0:
                desc = model.getErrorsDecription()
                print('Epoch:[{}/{}] Batch:[{:10d}/{}] '.format(e, args.epochs,
                                                                batch_idx * len(real_A_cpu),
                                                                len(dataset_A.dataset)), desc)
            if batch_idx % args.vis_interval == 0:
                imAB_gen_file = os.path.join(args.save_path, 'imAB_gen_{}_{}.jpg'.format(e, batch_idx))
                vutils.save_image(model.getABImagesTriple(), imAB_gen_file)
        e_end = time.time()
        e_time = e_end - e_begin
        print('End of epoch [{}/{}] Time taken: {:.4f} sec.'.format(e, args.epochs, e_time))
    model_file = os.path.join(args.save_path, 'model_{}.pth'.format(e))
    torch.save(model.state_dict(), model_file)


if __name__ == '__main__':
    train()
