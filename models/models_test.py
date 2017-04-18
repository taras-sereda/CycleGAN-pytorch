import unittest
import random
import time
from easydict import EasyDict as edict
import numpy as np
import torch
from torch.autograd import Variable

from models.generators import GeneratorCNN
from models.discriminators import DiscriminatorCNN
from models.cycleGAN import CycleGAN


class ModelsTest(unittest.TestCase):
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.fill_(0.1)
            # m.bias.fill_(0.0)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(0.1)
            m.bias.data.fill_(0.0)

    def testGen(self):
        hight, width, channels = 128, 128, 3
        out_channels = 3
        num_filters = 64

        image = Variable(torch.ones([1, channels, hight, width]))
        model = GeneratorCNN(channels, out_channels, num_filters)
        gen_image = model.forward(image)
        gen_image_np = gen_image.data.numpy()

        self.assertTrue(gen_image_np.any())
        self.assertEqual(gen_image_np.shape, (1, 3, 128, 128))

    def testDisctiminatorSigmoidTrue(self):
        hight, width, channels = 128, 128, 3
        num_filters = 64
        use_sigmoid = True
        expected_out_shape = (1, 1, 19, 19)

        image = Variable(torch.ones([1, channels, hight, width]))
        model = DiscriminatorCNN(channels, num_filters, use_sigmoid)
        result = model.forward(image)
        result_np = result.data.numpy()

        self.assertTrue(np.logical_and(
            np.greater_equal(result_np, 0),
            np.less_equal(result_np, 1)).all())
        self.assertEqual(result_np.shape, expected_out_shape)

    def testDisctiminatorSigmoidFalse(self):
        hight, width, channels = 128, 128, 3
        num_filters = 64
        use_sigmoid = False
        expected_out_shape = (1, 1, 19, 19)

        image = Variable(torch.ones([1, channels, hight, width]))
        model = DiscriminatorCNN(channels, num_filters, use_sigmoid)
        result = model.forward(image)
        result_np = result.data.numpy()

        self.assertTrue(result_np.any())
        self.assertEqual(result_np.shape, expected_out_shape)

    def testGradients(self):
        hight, width, channels = 128, 128, 3
        out_channels = 3
        num_f = 64
        use_sigmoid = True

        criterionGAN = torch.nn.MSELoss()
        criterionRec = torch.nn.L1Loss()
        image = Variable(torch.ones([1, channels, hight, width]))

        netG = GeneratorCNN(channels, out_channels, num_f)
        netD = DiscriminatorCNN(channels, num_f, use_sigmoid)
        netG.apply(ModelsTest.weights_init)
        netD.apply(ModelsTest.weights_init)

        netG2 = GeneratorCNN(channels, out_channels, num_f)
        netG2.apply(ModelsTest.weights_init)

        D_A_size = netD.forward(image).size()
        label = Variable(torch.Tensor(D_A_size).fill_(0.9))

        fake = netG.forward(image)
        output = netD.forward(fake)
        errGAN = criterionGAN.forward(output, label)

        rec = netG2.forward(fake)
        errRec = criterionRec.forward(rec, image)
        print(errGAN)
        print(errRec)

    def testCycleGAN(self):

        params = edict({'width': 128,
                        'height': 128,
                        'load_size': 142,
                        'epochs': 100,
                        'batch_size': 1,
                        'lr': 2e-4,
                        'beta1': 0.5,
                        'pool_size': 50,
                        'input_nc': 3,
                        'output_nc': 3,
                        'use_lsgan': True,
                        'train': True,
                        'seed': 42,
                        'cuda': False,
                        'num_gpu': 1,
                        'save_interval': 1000,
                        'vis_interval': 500,
                        'log_interval': 50,
                        'num_workers': 2})

        def set_random_seed(seed, cuda):
            random.seed(seed)
            torch.manual_seed(seed)
            if cuda:
                torch.cuda.manual_seed_all(seed)

        set_random_seed(params.seed, params.cuda)

        image = torch.ones([1, params.input_nc, params.height, params.width])
        model = CycleGAN(params=params)
        model.real_A.data.resize_(image.size()).copy_(image)
        model.real_B.data.resize_(image.size()).copy_(image)
        start = time.time()
        model.optimizeParameters()
        end = time.time()
        print(end-start)

    def testBatchNorm(self):
        image = Variable(torch.Tensor(
            [[[
                [0.9418, 0.1969, 0.7203, 0.9023, 0.2595],
                [0.0454, 0.0342, 0.1263, 0.3847, 0.1956],
                [0.4257, 0.8709, 0.3104, 0.9522, 0.1858],
                [0.3646, 0.9174, 0.9921, 0.4153, 0.9868],
                [0.0006, 0.5900, 0.0337, 0.1683, 0.9938]
            ],
                [
                    [0.9523, 0.8345, 0.3192, 0.0336, 0.6997],
                    [0.8247, 0.2527, 0.7152, 0.8945, 0.6273],
                    [0.3698, 0.4293, 0.6364, 0.2152, 0.3920],
                    [0.1698, 0.4251, 0.7469, 0.2056, 0.4006],
                    [0.3587, 0.8192, 0.7600, 0.2096, 0.7360]
                ],
                [
                    [0.8746, 0.7107, 0.5118, 0.0503, 0.0376],
                    [0.6978, 0.6476, 0.4769, 0.0942, 0.9967],
                    [0.9741, 0.7938, 0.3837, 0.6650, 0.5370],
                    [0.8629, 0.6245, 0.9270, 0.3244, 0.0070],
                    [0.5875, 0.0987, 0.6081, 0.7670, 0.8042]
                ]]]
        ))
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 2, 2),
            torch.nn.BatchNorm2d(2))
        model.apply(ModelsTest.weights_init)
        output = model.forward(image)
        print(output)


if __name__ == '__main__':
    unittest.main()
