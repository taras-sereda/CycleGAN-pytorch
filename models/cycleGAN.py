import itertools
import os
from collections import OrderedDict

import torch
import torch.optim as optim
from torch.autograd import Variable

from util.image_pool import ImagePool
from models.base import BaseModel
from models.discriminators import DiscriminatorCNN
from models.generators import GeneratorCNN


# torch SpatialFullConvolution == pytorch ConvTranspose2d
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        # m.bias.fill_(0.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0.0)


class CycleGAN(BaseModel):
    def __init__(self, params):
        super(CycleGAN, self).__init__()
        self.cuda = params.cuda
        self.use_lsgan = params.use_lsgan
        self.use_sigmoid = not params.use_lsgan
        self.backward_type = params.backward_type
        self.save_path = params.save_path
        self.lambda_A = 10
        self.lambda_B = 10
        self.identity = params.identity

        # fake pools
        if params.train:
            self.fakeAPool = ImagePool(params.pool_size)
            self.fakeBPoll = ImagePool(params.pool_size)

        # containers for data and labels
        self.shape_A = (params.batch_size, params.input_nc, params.height, params.width)
        self.shape_B = (params.batch_size, params.output_nc, params.height, params.width)

        self.input_A = torch.Tensor(*self.shape_A)
        self.input_B = torch.Tensor(*self.shape_B)

        self.real_A = Variable(torch.Tensor(*self.shape_A))
        self.real_B = Variable(torch.Tensor(*self.shape_B))
        self.fake_A = Variable(torch.Tensor(*self.shape_A))
        self.fake_B = Variable(torch.Tensor(*self.shape_B))
        self.rec_A = Variable(torch.Tensor(*self.shape_A))
        self.rec_B = Variable(torch.Tensor(*self.shape_B))

        # models
        self.netG_A = GeneratorCNN(name='G_A', in_channels=params.input_nc, out_channels=params.output_nc)
        self.netG_B = GeneratorCNN(name='G_B', in_channels=params.input_nc, out_channels=params.output_nc)

        self.netD_A = DiscriminatorCNN(name='D_A', in_channels=params.input_nc, use_sigmoid=self.use_sigmoid)
        self.netD_B = DiscriminatorCNN(name='D_B', in_channels=params.input_nc, use_sigmoid=self.use_sigmoid)

        # criterions
        self.criterionGAN = torch.nn.MSELoss()
        self.criterionRec = torch.nn.L1Loss()

        # fill of fake and real labels
        D_A_size = self.netD_A.forward(self.real_B).size()
        self.fake_label_A = Variable(torch.zeros(D_A_size))
        self.real_label_A = Variable(torch.Tensor(D_A_size).fill_(0.9))
        D_B_size = self.netD_B.forward(self.real_A).size()
        self.fake_label_B = Variable(torch.zeros(D_B_size))
        self.real_label_B = Variable(torch.Tensor(D_B_size).fill_(0.9))

        # optimizers
        if params.train:
            self.netG_AB_optim = optim.Adam(itertools.chain(self.netG_A.parameters(),
                                                            self.netG_B.parameters()),
                                            lr=params.lr, betas=(params.beta1, 0.999))
            self.netG_A_optim = optim.Adam(self.netG_A.parameters(), lr=params.lr, betas=(params.beta1, 0.999))
            self.netG_B_optim = optim.Adam(self.netG_B.parameters(), lr=params.lr, betas=(params.beta1, 0.999))
            self.netD_A_optim = optim.Adam(self.netD_A.parameters(), lr=params.lr, betas=(params.beta1, 0.999))
            self.netD_B_optim = optim.Adam(self.netD_B.parameters(), lr=params.lr, betas=(params.beta1, 0.999))

        if params.cuda:
            if params.num_gpu == 1:
                # move models on gpu
                self.netG_A = self.netG_A.cuda()
                self.netD_A = self.netD_A.cuda()
                self.netG_B = self.netG_B.cuda()
                self.netD_B = self.netD_B.cuda()

                # move data containers on gpu
                self.input_A = self.input_A.cuda()
                self.input_B = self.input_B.cuda()
                self.real_A = self.real_A.cuda()
                self.real_B = self.real_B.cuda()
                self.fake_A = self.fake_A.cuda()
                self.fake_B = self.fake_B.cuda()
                self.rec_A = self.rec_A.cuda()
                self.rec_B = self.rec_B.cuda()

                # move labels on gpu
                self.fake_label_A = self.fake_label_A.cuda()
                self.real_label_A = self.real_label_A.cuda()
                self.fake_label_B = self.fake_label_B.cuda()
                self.real_label_B = self.real_label_B.cuda()

                # move criterions on gpu
                self.criterionGAN = self.criterionGAN.cuda()
                self.criterionRec = self.criterionRec.cuda()

            else:
                raise NotImplementedError

        # init models parameteres
        self.netG_A.apply(weights_init)
        self.netD_A.apply(weights_init)
        self.netG_B.apply(weights_init)
        self.netD_B.apply(weights_init)

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)

    def test_model(self):
        """test model
        real_A -> fake_B -> rec_A
        real_B -> fake_A -> rec_B"""
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A = self.netG_B(self.fake_B)

        self.real_B = Variable(self.input_B, volatile=True)
        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)

    def backward_basic_D(self, netD, real, fake, real_label, fake_label):

        # real log(D_A(B))
        output = netD.forward(real)
        loss_D_real = self.criterionGAN(output, real_label)

        # fake log(1 - D_A((G_A(A)))
        output2 = netD.forward(fake.detach())
        loss_D_fake = self.criterionGAN(output2, fake_label)
        loss_D = (loss_D_real + loss_D_fake) / 2
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fakeBPoll.query(self.fake_B)

        self.netD_A_optim.zero_grad()
        self.loss_D_A = self.backward_basic_D(self.netD_A, self.real_B, fake_B, self.real_label_A, self.fake_label_A)
        self.netD_A_optim.step()

    def backward_D_B(self):
        fake_A = self.fakeAPool.query(self.fake_A)

        self.netD_B_optim.zero_grad()
        self.loss_D_B = self.backward_basic_D(self.netD_B, self.real_A, fake_A, self.real_label_B, self.fake_label_B)
        self.netD_B_optim.step()

    def backward_basic_G(self, netG, netD, netE, real, real2, real_label, lambda1, lambda2):

        identity = self.identity

        # log(D_A(G_A(A)))
        fake = netG.forward(real)
        output = netD.forward(fake)
        loss_GAN = self.criterionGAN(output, real_label)

        # forward cycle
        rec = netE.forward(fake)
        loss_Rec = self.criterionRec(rec, real) * lambda1

        loss_total = loss_GAN + loss_Rec

        if identity > 0:
            # usefull for photo -> painting
            # for preserving colors composition
            identity2 = netG.forward(real2)
            loss_identity = self.criterionRec(identity2, real2) * lambda1 * identity
            loss_total += loss_identity

        loss_total.backward()

        # backward cycle
        fake2 = netE.forward(real2).detach()
        rec2 = netG.forward(fake2)

        loss_back_cycle = self.criterionRec(rec2, real2) * lambda2
        loss_back_cycle.backward()

        return loss_total, loss_back_cycle, fake.detach(), rec.detach()

    def backward_G_A(self):
        self.netG_A.zero_grad()
        self.netD_A.zero_grad()
        self.netG_B.zero_grad()

        self.loss_G_A, self.loss_cycle_A, self.fake_B, self.rec_A = \
            self.backward_basic_G(self.netG_A, self.netD_A, self.netG_B, self.real_A,
                                  self.real_B, self.real_label_A, self.lambda_A, self.lambda_B)
        self.netG_A_optim.step()

    def backward_G_B(self):
        self.netG_B.zero_grad()
        self.netD_B.zero_grad()
        self.netG_A.zero_grad()

        self.loss_G_B, self.loss_cycle_B, self.fake_A, self.rec_B = \
            self.backward_basic_G(self.netG_B, self.netD_B, self.netG_A, self.real_B,
                                  self.real_A, self.real_label_B, self.lambda_B, self.lambda_A)

        self.netG_B_optim.step()

    def backward_G(self):

        identity = self.identity
        lambda_A = self.lambda_A
        lambda_B = self.lambda_B

        self.netG_AB_optim.zero_grad()

        self.fake_B = self.netG_A(self.real_A)
        output = self.netD_A(self.fake_B)
        self.loss_G_A = self.criterionGAN(output, self.real_label_A)

        self.rec_A = self.netG_B(self.fake_B)
        self.loss_cycle_A = self.criterionRec(self.rec_A, self.real_A) * lambda_A

        self.fake_A = self.netG_B(self.real_B)
        output = self.netD_B(self.fake_A)
        self.loss_G_B = self.criterionGAN(output, self.real_label_B)

        self.rec_B = self.netG_A(self.fake_A)
        self.loss_cycle_B = self.criterionRec(self.rec_B, self.real_B) * lambda_B

        self.loss_G = self.loss_G_A + self.loss_cycle_A + self.loss_G_B + self.loss_cycle_B

        if identity > 0:
            identity_A = self.netG_A(self.real_B)
            loss_identity_A = self.criterionRec(identity_A, self.real_B) * lambda_A * identity

            identity_B = self.netG_B(self.real_A)
            loss_identity_B = self.criterionRec(identity_B, self.real_A) * lambda_B * identity

            self.loss_G += loss_identity_A + loss_identity_B
        self.loss_G.backward()

        self.netG_AB_optim.step()

    def optimize_parameters_fused(self):

        self.forward()

        self.backward_G()
        self.backward_D_A()
        self.backward_D_B()

    def optimize_parameters_separate(self):

        self.forward()

        self.backward_G_A()
        self.backward_D_A()
        self.backward_G_B()
        self.backward_D_B()

    def optimize_parameters(self):

        if self.backward_type == 'fused':
            self.optimize_parameters_fused()
        elif self.backward_type == 'separate':
            self.optimize_parameters_separate()

    def loss_variable2scalar(self, var):
        if self.cuda:
            return var.data.cpu().numpy()[0]
        else:
            return var.data.numpy()[0]

    def get_AB_images_triple(self):
        return torch.cat((self.real_A.data, self.fake_B.data, self.rec_A.data,
                          self.real_B.data, self.fake_A.data, self.rec_B.data))

    def set_inputs(self, input):

        input_A = input['A']
        input_B = input['B']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)

    def get_errors(self):
        loss_G_A = self.loss_G_A.data[0]
        loss_D_A = self.loss_D_A.data[0]
        loss_cycle_A = self.loss_cycle_A.data[0]

        loss_G_B = self.loss_G_B.data[0]
        loss_D_B = self.loss_D_B.data[0]
        loss_cycle_B = self.loss_cycle_B.data[0]

        errors = OrderedDict([('G_A', loss_G_A), ('D_A', loss_D_A), ('cycle_A', loss_cycle_A),
                              ('G_B', loss_G_B), ('D_B', loss_D_B), ('cycle_B', loss_cycle_B)])

        return errors

    def get_errors_string(self):

        errors = self.get_errors()
        errors_str = ''
        for k, v in errors.items():
            errors_str += '{}={:.4f} '.format(k, v)
        return errors_str

    def save_parameters(self, epoch):
        model_file_G_A = os.path.join(self.save_path, 'model_G_A_{}.pth'.format(epoch))
        model_file_G_B = os.path.join(self.save_path, 'model_G_B_{}.pth'.format(epoch))
        model_file_D_A = os.path.join(self.save_path, 'model_D_A_{}.pth'.format(epoch))
        model_file_D_B = os.path.join(self.save_path, 'model_D_B_{}.pth'.format(epoch))
        torch.save(self.netG_A.state_dict(), model_file_G_A)
        torch.save(self.netG_B.state_dict(), model_file_G_B)
        torch.save(self.netD_A.state_dict(), model_file_D_A)
        torch.save(self.netD_B.state_dict(), model_file_D_B)

    def load_parameters(self, epoch):
        print('loading model parameters from epoch {}...'.format(epoch))
        map2device = lambda storage, loc: storage
        model_file_G_A = os.path.join(self.save_path, 'model_G_A_{}.pth'.format(epoch))
        model_file_G_B = os.path.join(self.save_path, 'model_G_B_{}.pth'.format(epoch))
        model_file_D_A = os.path.join(self.save_path, 'model_D_A_{}.pth'.format(epoch))
        model_file_D_B = os.path.join(self.save_path, 'model_D_B_{}.pth'.format(epoch))
        self.netG_A.load_state_dict(torch.load(model_file_G_A, map_location=map2device))
        self.netG_B.load_state_dict(torch.load(model_file_G_B, map_location=map2device))
        self.netD_A.load_state_dict(torch.load(model_file_D_A, map_location=map2device))
        self.netD_B.load_state_dict(torch.load(model_file_D_B, map_location=map2device))
        print('model parameters loaded successfully')

    @staticmethod
    def __print_model_desc(model):
        print('[Model {name} generator architecture]'.format(name=model.name))
        for idx, m in enumerate(model.modules()):
            print(idx, '->', m)

    @staticmethod
    def __get_model_params_num(model):

        get_params = lambda p: p.numel()
        model_params = model.parameters()
        model_params_num = sum(map(get_params, model_params))
        return model_params_num

    def print_model_desription(self):

        net_G_A_params_num = CycleGAN.__get_model_params_num(self.netG_A)
        net_D_A_params_num = CycleGAN.__get_model_params_num(self.netD_A)
        net_G_B_params_num = CycleGAN.__get_model_params_num(self.netG_B)
        net_D_B_params_num = CycleGAN.__get_model_params_num(self.netD_B)
        total_params_num = sum([net_G_A_params_num, net_D_A_params_num,
                                net_G_B_params_num, net_D_B_params_num])
        print('[Model {} params num: {}]'.format(self.netG_A.name, net_G_A_params_num))
        print('[Model {} params num: {}]'.format(self.netD_A.name, net_D_A_params_num))
        print('[Model {} params num: {}]'.format(self.netG_B.name, net_G_B_params_num))
        print('[Model {} params num: {}]'.format(self.netD_B.name, net_D_B_params_num))
        print('[Total CycleGAN params: {}]'.format(total_params_num))
        CycleGAN.__print_model_desc(self.netG_A)
        CycleGAN.__print_model_desc(self.netD_A)
        CycleGAN.__print_model_desc(self.netG_B)
        CycleGAN.__print_model_desc(self.netD_B)
