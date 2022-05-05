import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from util.metrics import PSNR, roll_2
import pytorch_msssim
from util.util import fft2, ifft2
from util.hfen import hfen


class DLMRIModel(BaseModel):
    def name(self):
        return 'DLMRIModel'

    def gradient_penalty(self, y, x):
        """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
        weight = torch.ones(y.size()).to(self.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=x,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    # Initialize the model
    def initialize(self, opt):
        BaseModel.initialize(self, opt)  # ATTENTION HERE: NEED TO ALTER THE DEFAULT PLAN
        self.netG_I = networks.define_G(opt, opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.which_model_netG_I, opt.norm, not opt.no_dropout, opt.init_type,
                                        opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if self.train_phase == 'generator':
                self.model_names = ['G_I']
                self.loss_names = ['G_I_L1', 'G_I_L2', 'SSIM', 'PSNR']
            else:
                self.model_names = ['G_I', 'D_I']
                self.loss_names = ['G_GAN_I', 'G_I_L1', 'G_I_L2', 'D_GAN_I', 'SSIM', 'PSNR']
        else:  # during test time, only load Gs
            self.model_names = ['G_I']
            self.loss_names = ['SSIM', 'PSNR', 'HFEN']
        if self.isTrain:
            if self.train_phase == 'generator':
                self.visual_names = ['kreal', 'Ireal', 'Ifake', 'Iunder', 'mask', 'Preal', 'IDL']
            else:
                self.visual_names = ['kreal', 'Ireal', 'Ifake', 'Iunder', 'mask', 'Preal', 'IDL']
        else:
            self.visual_names = ['Ireal', 'Ifake', 'Iunder', 'mask', 'IDL']

        self.criterionL1 = torch.nn.L1Loss()
        self.criterionMSE = torch.nn.MSELoss()
        self.ssim_loss = pytorch_msssim.SSIM(val_range=2)
        if self.isTrain:
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(list((self.netG_I.parameters())),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

        if self.isTrain and self.train_phase == 'together':
            self.no_wgan = opt.no_wgan
            self.no_wgan_gp = opt.no_wgan_gp
            if self.no_wgan_gp == False:
                self.disc_step = opt.disc_step
            else:
                self.disc_step = 1
            self.disc_model = opt.disc_model
            use_sigmoid = opt.no_lsgan
            if opt.disc_model == 'pix2pix':
                self.netD_I = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                                opt.which_model_netD_I,
                                                opt.n_layers_D_I, opt.norm, use_sigmoid, opt.init_type,
                                                opt.init_gain,
                                                self.gpu_ids)
            if opt.disc_model == 'traditional':
                self.netD_I = networks.define_D(opt.output_nc, opt.ndf, opt.which_model_netD_I,
                                                opt.n_layers_D_I, opt.norm, use_sigmoid, opt.init_type,
                                                opt.init_gain,
                                                self.gpu_ids)
            self.loss_wgan_gp = opt.loss_wgan_gp
            self.fake_I_pool = ImagePool(opt.pool_size)
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, use_l1=not opt.no_l1gan).to(self.device)
            self.optimizer_D_I = torch.optim.Adam(self.netD_I.parameters(),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_D_I)

    def set_input(self, input):
        self.kreal = input['kreal'].to(self.device)
        self.smap = input['smap'].to(self.device)
        self.mask = input['mask'].to(self.device)
        self.IDL = input['DLRecon'].to(self.device)
        self.AT = networks.OPAT(self.smap)
        self.A = networks.OPA(self.smap)
        self.Iunder = self.AT(self.kreal, self.mask)
        self.Ireal = self.AT(self.kreal, torch.ones_like(self.mask))
        self.scale = torch.mean(torch.abs(self.Iunder))
        self.Preal = torch.atan(self.Ireal[:, 1, :, :] / self.Ireal[:, 0, :, :]).unsqueeze(1)
        self.image_paths = input['path']
        self.ssim_loss = pytorch_msssim.SSIM(val_range=2 * torch.max(torch.abs(self.Ireal)).cpu().float()).to(self.device)

    def backward_G(self):
        # First, G(A) should fake the discriminator
        if self.isTrain and self.train_phase == 'together':
            if self.disc_model == 'pix2pix':
                fake_AB_I = torch.cat((self.Ifake, self.Iunder), 1)
                pred_fake_I = self.netD_I(fake_AB_I)
            if self.disc_model == 'traditional':
                pred_fake_I = self.netD_I(self.Ifake)
            if self.no_wgan == False:
                self.loss_G_GAN_I = -pred_fake_I.mean()
            else:
                self.loss_G_GAN_I = self.criterionGAN(pred_fake_I, True)
        else:
            self.loss_G_GAN_I = 0
        self.loss_G_GAN_I = self.loss_G_GAN_I * self.opt.loss_GAN_I
        self.loss_G_GAN = self.loss_G_GAN_I
        self.loss_G_I_L1 = self.criterionL1(self.Ifake, self.Ireal) * self.opt.loss_content_I_l1
        self.loss_G_I_L2 = self.criterionMSE(self.Ifake, self.Ireal) * self.opt.loss_content_I_l2
        self.loss_G_CON_I = self.loss_G_I_L1 + self.loss_G_I_L2

        self.loss_G = self.loss_G_CON_I + self.loss_G_GAN - self.loss_SSIM * self.opt.loss_ssim
        self.loss_G.backward()

    def backward_D(self):
        if self.disc_model == 'pix2pix':
            fake_AB_I = self.fake_k_pool.query(self.Ifake)
            pred_real_I = self.netD_I(torch.cat((self.Ireal, self.Iunder), 1))
            pred_fake_I = self.netD_I(fake_AB_I.detach())
        if self.disc_model == 'traditional':
            fake_AB_I = self.fake_I_pool.query(self.Ifake)
            pred_real_I = self.netD_I(self.Ireal)
            pred_fake_I = self.netD_I(fake_AB_I.detach())
        if self.no_wgan == False:
            self.loss_D_GAN_fake_I = pred_fake_I.mean()
            self.loss_D_GAN_real_I = -pred_real_I.mean()
        elif self.no_wgan_gp == False:
            self.loss_D_GAN_fake_I = pred_fake_I.mean()
            self.loss_D_GAN_real_I = -pred_real_I.mean()
            alpha = torch.rand(self.Ireal.size(0), 1, 1, 1).to(self.device)
            x_hat = (alpha * self.Ireal.data + (1 - alpha) * self.Ifake.data).requires_grad_(True)
            out_src = self.netD_I(x_hat)
            self.d_loss_gp_I = self.gradient_penalty(out_src, x_hat) * self.loss_wgan_gp
        else:
            self.loss_D_GAN_fake_I = self.criterionGAN(pred_fake_I, False)
            self.loss_D_GAN_real_I = self.criterionGAN(pred_real_I, True)
        self.loss_D_GAN_I = 0.5 * (self.loss_D_GAN_fake_I + self.loss_D_GAN_real_I) * self.opt.loss_GAN_I
        self.loss_D_GAN = self.loss_D_GAN_I * self.opt.beta
        if self.no_wgan_gp == False:
            self.loss_D_GAN = self.loss_D_GAN + self.d_loss_gp_I

        self.loss_D_GAN.backward()

    def forward(self):
        # CG_A = networks.CG_A(self.opt.MODLtol, self.opt.MODLLambda)
        CG = networks.CG.apply

        # Ifake = self.Iunder 	# For S
        Ifake = self.IDL	# For B+S
        for ii in range(self.opt.num_blocks):
            Ifake1 = self.netG_I(Ifake)
            Ifake = CG(Ifake1, self.opt.MODLtol, self.opt.MODLLambda, self.smap, self.mask, self.Iunder)
        self.Ifake = Ifake
	
	# For 'explicit' data-consistency
        # Ifake = self.IDL
        # for ii in range(self.opt.num_blocks):
        #     Ifake1 = self.netG_I(Ifake)
        #     Ifake = CG(Ifake1 + self.IDL, self.opt.MODLtol, self.opt.MODLLambda, self.smap, self.mask,
        #                self.Iunder)
        # self.Ifake = Ifake + self.IDL

        self.loss_PSNR = PSNR(self.Ireal, self.Ifake)
        self.loss_SSIM = self.ssim_loss(self.Ireal, self.Ifake)
        self.loss_HFEN = hfen(self.Ireal, self.Ifake, window_size=11, size_average=True, full=False, device=self.device)
        # if not self.isTrain:
        self.loss_SSIM_DL = self.ssim_loss(self.Ireal, self.IDL)
        self.loss_SSIM_under = self.ssim_loss(self.Ireal, self.Iunder)
        self.loss_PSNR_DL = PSNR(self.Ireal, self.IDL)
        self.loss_PSNR_under = PSNR(self.Ireal, self.Iunder)
        self.loss_HFEN_DL = hfen(self.Ireal, self.IDL, window_size=11, size_average=True, full=False,
                                 device=self.device)
        self.loss_HFEN_under = hfen(self.Ireal, self.Iunder, window_size=11, size_average=True, full=False,
                                    device=self.device)
        print('loss_SSIM_DL', self.loss_SSIM_DL, 'loss_SSIM_under', self.loss_SSIM_under, 'loss_PSNR_DL',
              self.loss_PSNR_DL, 'loss_PSNR_under', self.loss_PSNR_under, 'loss_HFEN_DL', self.loss_HFEN_DL,
              'loss_HFEN_under', self.loss_HFEN_under)

    def optimize_parameters(self):
        if self.isTrain and self.train_phase == 'together':
            self.forward()
            self.set_requires_grad(self.netD_I, True)
            for iter_d in range(self.disc_step):
                self.optimizer_D_I.zero_grad()
                self.backward_D()
                self.optimizer_D_I.step()
            self.set_requires_grad(self.netD_I, False)
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
        else:
            self.forward()
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
