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

        self.lambda_A = 10
        self.lambda_B = 10

        # fake pools
        if params.train:
            self.fakeAPool = ImagePool(params.pool_size)
            self.fakeBPoll = ImagePool(params.pool_size)

        # containers for data and labels
        if params.train:
            self.real_A = Variable(torch.Tensor(params.batch_size, params.input_nc, params.height, params.width))
            self.real_B = Variable(torch.Tensor(params.batch_size, params.output_nc, params.height, params.width))
            self.fake_A = Variable(torch.Tensor(params.batch_size, params.input_nc, params.height, params.width))
            self.fake_B = Variable(torch.Tensor(params.batch_size, params.output_nc, params.height, params.width))
            self.rec_A = Variable(torch.Tensor(params.batch_size, params.input_nc, params.height, params.width))
            self.rec_B = Variable(torch.Tensor(params.batch_size, params.output_nc, params.height, params.width))

        # models
        self.netG_A = GeneratorCNN(params.input_nc, params.output_nc)
        self.netD_A = DiscriminatorCNN(params.input_nc, use_sigmoid=self.use_sigmoid)
        self.netG_B = GeneratorCNN(params.input_nc, params.output_nc)
        self.netD_B = DiscriminatorCNN(params.input_nc, use_sigmoid=self.use_sigmoid)

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
        self.netG_A_optim = optim.Adam(self.netG_A.parameters(), lr=params.lr, betas=(params.beta1, 0.999))
        self.netD_A_optim = optim.Adam(self.netD_A.parameters(), lr=params.lr, betas=(params.beta1, 0.999))
        self.netG_B_optim = optim.Adam(self.netG_B.parameters(), lr=params.lr, betas=(params.beta1, 0.999))
        self.netD_B_optim = optim.Adam(self.netD_B.parameters(), lr=params.lr, betas=(params.beta1, 0.999))

        if params.cuda:
            if params.num_gpu == 1:
                # move models on gpu
                self.netG_A = self.netG_A.cuda()
                self.netD_A = self.netD_A.cuda()
                self.netG_B = self.netG_B.cuda()
                self.netD_B = self.netD_B.cuda()

                # move data containers on gpu
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

    def forward(self, real_A, real_B):
        raise NotImplementedError
        # gen_A_B = self.netG_A.forward(real_A)
        # output_D_B = self.netD_B.forward(gen_A_B)
        # gen_B_A = self.netG_B.forward(real_B)
        # output_D_A = self.netD_A.forward(gen_B_A)
        # return gen_A_B, output_D_B, gen_B_A, output_D_A

    def fDx_basic(self, netD, real, fake, real_label, fake_label):

        # real log(D_A(B))
        output = netD.forward(real)
        errD_real = self.criterionGAN(output, real_label)
        errD_real.backward()

        # fake log(1 - D_A((G_A(A)))
        output2 = netD.forward(fake)
        errD_fake = self.criterionGAN(output2, fake_label)
        errD_fake.backward()
        errD = (errD_real + errD_fake) / 2
        return errD

    def fDAx(self):
        self.netD_A.zero_grad()
        fake_B = self.fakeBPoll.query(self.fake_B)
        self.errD_A = self.fDx_basic(self.netD_A, self.real_B, fake_B, self.real_label_A, self.fake_label_A)
        self.netD_A_optim.step()

    def fDBx(self):
        self.netD_B.zero_grad()
        fake_A = self.fakeAPool.query(self.fake_A)
        self.errD_B = self.fDx_basic(self.netD_B, self.real_A, fake_A, self.real_label_B, self.fake_label_B)
        self.netD_B_optim.step()

    def fGx_basic(self, netG, netD, netE, real, real2, real_label):

        # # identity not implemented yet
        # # TODO identity
        # GAN Loss : D_A(G_A(A))
        fake = netG.forward(real)
        output = netD.forward(fake)
        errG = self.criterionGAN(output, real_label)
        # errG.backward(retain_variables=True)

        # forward cycle loss
        rec = netE.forward(fake)
        errRec = self.criterionRec(rec, real) * self.lambda_A
        errTotal = errG + errRec
        errTotal.backward()

        # backward cycle loss
        # TODO check if it can be detached .detach()
        # fake2 = netE.forward(real2)
        fake2 = netE.forward(real2).detach()
        rec2 = netG.forward(fake2)
        errAdapt = self.criterionRec(rec2, real2) * self.lambda_B
        errAdapt.backward()

        return errG, errRec, errAdapt, fake.clone().detach(), rec.clone().detach()

    def fGAx(self):
        self.netG_A.zero_grad()
        self.netD_A.zero_grad()
        self.netG_B.zero_grad()
        self.errG_A, self.errRec_A, self.errAdapt_A, self.fake_B, self.rec_A = \
            self.fGx_basic(self.netG_A, self.netD_A, self.netG_B, self.real_A, self.real_B, self.real_label_A)
        self.netG_A_optim.step()

    def fGBx(self):
        self.netG_B.zero_grad()
        self.netD_B.zero_grad()
        self.netG_A.zero_grad()
        self.errG_B, self.errRec_B, self.errAdapt_B, self.fake_A, self.rec_B = \
            self.fGx_basic(self.netG_B, self.netD_B, self.netG_A, self.real_B, self.real_A, self.real_label_B)
        self.netG_B_optim.step()

    def optimizeParameters(self):

        self.fGAx()
        self.fDAx()

        self.fGBx()
        self.fDBx()

    def varError2scalar(self, var):
        if self.cuda:
            return var.data.cpu().numpy()[0]
        else:
            return var.data.numpy()[0]

    def getErrorsDecription(self):

        errG_A, errD_A, errRec_A, errAdapt_A = list(map(lambda x: self.varError2scalar(x),
                                                        [self.errG_A, self.errD_A, self.errRec_A, self.errAdapt_A]))
        errG_B, errD_B, errRec_B, errAdapt_B = list(map(lambda x: self.varError2scalar(x),
                                                        [self.errG_B, self.errD_B, self.errRec_B, self.errAdapt_B]))

        description = ('[A] G:{:.4f} D:{:.4f} Rec:{:.4f} Adapt:{:.4f} ||'
                       ' [B] G:{:.4f} D:{:.4f} Rec:{:.4f} Adapt:{:.4f}').format(errG_A, errD_A, errRec_A, errAdapt_A,
                                                                                errG_B, errD_B, errRec_B, errAdapt_B)
        return description

    def getABImagesTriple(self):
        return torch.cat((self.real_A.data, self.fake_B.data, self.rec_A.data,
                          self.real_B.data, self.fake_A.data, self.rec_B.data))
