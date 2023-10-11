
import os
import torch
import joblib
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from tools.classify_result import classify_result

def parse_args():
    parser = argparse.ArgumentParser()

    # basic parameters: Train
    parser.add_argument('--output_path', type=str, default=r'.\result\AAU_Net', help='the output path')
    parser.add_argument('--data_path', type=str, default=r'.\data\Simulated_dataset.pkl', help='the output path')
    parser.add_argument("--n_epochs", type=int, default=15, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr_g", type=float, default=0.0005, help="learning rate of generator")
    parser.add_argument("--lr_d", type=float, default=0.0002, help="learning rate of discriminator")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--use_cuda", type=bool, default=True, help="whether to sue cuda")

    # hyper-parameters: Network
    parser.add_argument('--IC', type=list, default=[1, 1], help='input channels of the networks')
    parser.add_argument('--CL', type=list, default=[32, 64, 128, 512], help='channels list of the networks')
    parser.add_argument('--KL', type=list, default=[8, 8, 4, 4], help='kernals list of the networks')
    parser.add_argument('--PL', type=list, default=[2, 2, 1, 1], help='paddings list of the networks')
    parser.add_argument('--SL', type=list, default=[4, 4, 2, 2], help='strides list of the networks')
    parser.add_argument("--latent_dim", type=int, default=10, help="dimensionality of the latent code")
    parser.add_argument('--unfoldings', type=int, default=4, help='unrollings of the networks')
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
    parser.add_argument("--wa", type=float, default=0.001, help="the weight for the adversarial loss")
    parser.add_argument("--wr", type=float, default=0.999, help="the weight for the reconstruction loss")
    parser.add_argument("--ratio", type=float, default=0.333, help="the rough ratio of the anomaly samples")
    opt = parser.parse_args(args=[])

    return opt

class SoftThreshold_1d(nn.Module):
    def __init__(self, channel_num, init_threshold=1e-3):
        super(SoftThreshold_1d, self).__init__()
        # 初始化阈值参数
        self.threshold = nn.Parameter(init_threshold * torch.ones(1, channel_num, 1))

    def forward(self, x):
        # 计算掩码（mask）
        mask1 = (x > self.threshold).float()  # 大于阈值的元素置为1，否则置为0
        mask2 = (x < -self.threshold).float()  # 小于阈值的元素置为1，否则置为0
        # 进行软阈值处理
        out = mask1.float() * (x - self.threshold)  # 大于阈值的元素减去阈值
        out += mask2.float() * (x + self.threshold)  # 小于阈值的元素加上阈值
        return out


class UnrolledAutoEncoder(nn.Module):
    """
    NetG NETWORK
    """
    def __init__(self, opt):
        super(UnrolledAutoEncoder, self).__init__()
        self.opt = opt
        self.opt.Tensor = torch.cuda.FloatTensor if opt.use_cuda else torch.FloatTensor
        self.T = opt.unfoldings
        # Convolutional Filters
        self.W1 = nn.Parameter(torch.randn(opt.CL[0], opt.IC[0], opt.KL[0]), requires_grad=True) # 32*1*8 
        self.strd1 = opt.SL[0]; self.pad1 = opt.PL[0] # 4 2

        self.W2 = nn.Parameter(torch.randn(opt.CL[1], opt.CL[0], opt.KL[1]), requires_grad=True) # 64*32*8 
        self.strd2 = opt.SL[1]; self.pad2 = opt.PL[1] # 4 2

        self.W3 = nn.Parameter(torch.randn(opt.CL[2], opt.CL[1], opt.KL[2]), requires_grad=True) # 128*64*4
        self.strd3 = opt.SL[2]; self.pad3 = opt.PL[2] # 2 1

        self.W4 = nn.Parameter(torch.randn(opt.CL[3], opt.CL[2], opt.KL[3]), requires_grad=True) # 512*128*84
        self.strd4 = opt.SL[3]; self.pad4 = opt.PL[3] # 2 1

        self.c1 = nn.Parameter(torch.ones(1, 1, 1), requires_grad=True) # 1*1*1
        self.c2 = nn.Parameter(torch.ones(1, 1, 1), requires_grad=True)
        self.c3 = nn.Parameter(torch.ones(1, 1, 1), requires_grad=True)
        self.c4 = nn.Parameter(torch.ones(1, 1, 1), requires_grad=True)

        # linear
        self.mu     = nn.Linear(self.opt.CL[3]*16, self.opt.latent_dim) # (8192, 10)
        self.logvar = nn.Linear(self.opt.CL[3]*16, self.opt.latent_dim) # (8192, 10)
        self.linear = nn.Linear(self.opt.latent_dim, self.opt.CL[3]*16) # (10 ,8192)


        # Biases / Thresholds
        self.soft1 = SoftThreshold_1d(opt.CL[0])
        self.soft2 = SoftThreshold_1d(opt.CL[1])
        self.soft3 = SoftThreshold_1d(opt.CL[2])
        self.soft4 = SoftThreshold_1d(opt.CL[3])

        # Initialization
        self.W1.data = .1 / np.sqrt(opt.IC[0] * opt.KL[0]) * self.W1.data
        self.W2.data = .1 / np.sqrt(opt.CL[0] * opt.KL[1]) * self.W2.data
        self.W3.data = .1 / np.sqrt(opt.CL[1] * opt.KL[2]) * self.W3.data
        self.W4.data = .1 / np.sqrt(opt.CL[2] * opt.KL[3]) * self.W4.data

    def forward(self, x, test=False):
        # Encoding
        # x.shape 64*1*1024
        gamma1 = self.soft1(self.c1 * F.conv1d(x,      self.W1, stride=self.strd1, padding=self.pad1)) # self.W1 32*1*8
        """
        torch.nn.functional.conv1d 是 PyTorch 中的一个函数，用于在由多个输入平面组成的输入信号上应用一维卷积。

        该函数的参数和使用方式如下：
        torch.nn.functional.conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor
        
        input：输入的张量，形状为 (minibatch, in_channels, iW)，其中 minibatch 是批次大小，in_channels 是输入通道数，iW 是输入的长度。

        weight：卷积核的张量，形状为 (out_channels, in_channels/groups, kW)，其中 out_channels 是输出通道数，in_channels/groups 是输入通道数除以组数，kW 是卷积核的长度。

        bias（可选）：偏置项的张量，形状为 (out_channels)。默认为 None，表示不使用偏置项。

        stride：卷积核的步幅。可以是一个数字或一个长度为 1 的元组 (sW,)。默认为 1。

        padding：输入的隐式填充。可以是一个字符串（'valid' 或 'same'），一个数字或一个长度为 1 的元组 (padW,)。默认为 0。当 padding='valid' 时表示无填充，padding='same' 时表示填充输入使输出具有与输入相同的形状，但此模式仅支持步幅为 1 的情况。

        dilation：卷积核元素之间的间距。可以是一个数字或一个长度为 1 的元组 (dW,)。默认为 1。

        groups：将输入分成多个组，in_channels 应该可以被组数整除。默认为 1。

        该函数执行一维卷积操作，并返回结果张量。
        """
        gamma2 = self.soft2(self.c2 * F.conv1d(gamma1, self.W2, stride=self.strd2, padding=self.pad2))
        gamma3 = self.soft3(self.c3 * F.conv1d(gamma2, self.W3, stride=self.strd3, padding=self.pad3))
        gamma4 = self.soft4(self.c4 * F.conv1d(gamma3, self.W4, stride=self.strd4, padding=self.pad4))

        for _ in range(self.T):
            # Wi = I - ciD^TiDi 
            # forward computation: gamma(i+1) = soft(gamma^(i+1)-c*DT*(D*gamma^(i+1)-gamma(i)))
            gamma1 = self.soft1((gamma1 - self.c1 * F.conv1d(F.conv_transpose1d(gamma1, self.W1, stride=self.strd1, padding=self.pad1) - x, self.W1, stride=self.strd1, padding=self.pad1)))
            gamma2 = self.soft2((gamma2 - self.c2 * F.conv1d(F.conv_transpose1d(gamma2, self.W2, stride=self.strd2, padding=self.pad2) - gamma1, self.W2, stride=self.strd2, padding=self.pad2)))
            gamma3 = self.soft3((gamma3 - self.c3 * F.conv1d(F.conv_transpose1d(gamma3, self.W3, stride=self.strd3, padding=self.pad3) - gamma2, self.W3, stride=self.strd3, padding=self.pad3)))
            gamma4 = self.soft4((gamma4 - self.c4 * F.conv1d(F.conv_transpose1d(gamma4, self.W4, stride=self.strd4, padding=self.pad4) - gamma3, self.W4, stride=self.strd4, padding=self.pad4)))

        # Calculate the paramater
        mu =     self.mu(gamma4.view(gamma4.shape[0],-1))  # 最后是K+1
        logvar = self.logvar(gamma4.view(gamma4.shape[0],-1))
        z = reparameterization(mu, logvar, self.opt)

        # Decoding
        if test:
            gamma4_hat = self.linear(mu).view_as(gamma4) # 用均值
            z = mu
        else:
            gamma4_hat = self.linear(z).view_as(gamma4)     # 采样
        gamma3_hat = F.conv_transpose1d(gamma4_hat, self.W4, stride=self.strd4, padding=self.pad4)
        gamma2_hat = F.conv_transpose1d(gamma3_hat, self.W3, stride=self.strd3, padding=self.pad3)
        gamma1_hat = F.conv_transpose1d(gamma2_hat, self.W2, stride=self.strd2, padding=self.pad2) # gamma2_hat 64*64*64 W2 64*32*8 -> 64*32*256
        x_hat      = F.conv_transpose1d(gamma1_hat, self.W1, stride=self.strd1, padding=self.pad1)
        return x_hat, z

def reparameterization(mu, logvar, opt):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(opt.Tensor(np.random.normal(0, 1, (mu.size(0), opt.latent_dim))))
    z = sampled_z * std + mu
    return z

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        self.opt = opt
        self.model = nn.Sequential(
            nn.Linear(self.opt.latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        validity = self.model(z)
        return validity

def train_AAUNet(opt):

    # Use binary cross-entropy loss
    adversarial_loss = torch.nn.BCELoss()
    pixelwise_loss = torch.nn.L1Loss() # 输入x和目标y中每个元素之间的平均绝对误差（MAE）

    # Initialize generator and discriminator
    generator     = UnrolledAutoEncoder(opt)
    discriminator = Discriminator(opt)

    if opt.use_cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()
        pixelwise_loss.cuda()

    # Configure data loader
    outf = opt.output_path
    os.makedirs(outf, exist_ok=True) # exist_ok=True参数允许在目录已经存在时不引发异常
    os.makedirs(os.path.join(outf, 'model'), exist_ok=True)
    data_dict = joblib.load(opt.data_path)
    train = TensorDataset(torch.Tensor(data_dict['train_1d']), torch.Tensor(data_dict['train_label']))
    test =  TensorDataset(torch.Tensor(data_dict['test_1d']), torch.Tensor(data_dict['test_label']))
    dataloader = torch.utils.data.DataLoader(dataset=train, batch_size=opt.batch_size, shuffle=True, drop_last=False)
    testloader = torch.utils.data.DataLoader(dataset=test,  batch_size=opt.batch_size, shuffle=True, drop_last=False)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_g, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr_d, betas=(opt.b1, opt.b2))
    
    #  Training
    ACC, TPR, FPR = np.zeros((opt.n_epochs,)), np.zeros((opt.n_epochs,)), np.zeros((opt.n_epochs,)) # 准确率，真正率， 假正率
    opt.Tensor = torch.cuda.FloatTensor if opt.use_cuda else torch.FloatTensor
    with open(os.path.join(outf,"train_log.txt"), "w") as f:
        for epoch in range(opt.n_epochs):

            generator.train()
            discriminator.train()

            for i, (x, _) in enumerate(dataloader):

                # Adversarial ground truths
                valid = Variable(opt.Tensor(x.shape[0], 1).fill_(1.0), requires_grad=False)
                fake = Variable(opt.Tensor(x.shape[0], 1).fill_(0.0), requires_grad=False)

                # Configure input
                x_real = Variable(x.type(opt.Tensor))

                #  Train Generator
                optimizer_G.zero_grad()
                x_hat, x_z = generator(x_real) # x_hat 64*1*1024, x_z 64*10

                # Loss measures generator's ability to fool the discriminator
                g_loss = opt.wa * adversarial_loss(discriminator(x_z), valid) + opt.wr * pixelwise_loss(x_hat, x_real)

                g_loss.backward()
                optimizer_G.step()

                #  Train Discriminator
                if i % opt.n_critic==0:
                    z = Variable(opt.Tensor(np.random.normal(0, 1, (x_hat.shape[0], opt.latent_dim))))
                    optimizer_D.zero_grad()
                    # Measure discriminator's ability to classify real from generated samples
                    real_loss = adversarial_loss(discriminator(z), valid)
                    fake_loss = adversarial_loss(discriminator(x_z.detach()), fake)
                    d_loss = 0.5 * (real_loss + fake_loss)

                    d_loss.backward()
                    optimizer_D.step()
            
            # testing
            generator.eval()
            discriminator.eval()
            TP, TN, FP, FN = 0, 0, 0, 0
            for i, (x, y) in enumerate(testloader):
                x_real = Variable(x.type(opt.Tensor))
                x_hat, x_z = generator(x_real)
                score = discriminator(x_z)
                result = score.cpu().detach().numpy()
                """
                pytorch中有时需要复制一下张量（tensor），如果在迭代过程中就会涉及到梯度的问题。

                我了解的常用的tensor复制方式有两种 .detach（）和.clone（）

                1、.detach（）方法，比如已有tensor a ，b=a.detach（），则相当于给a起了个别名b，两个在内存中实际是一个东西，但是在计算梯度时，梯度从后面传到b不会再往前传了，到这就截止了。当只需要tensor数值，不需要往前的梯度或者故意将梯度在这儿截断时可以用这个方法。

                2、.clone（）方法，与上面正相反，如果c=a.clone（），那么c跟a在内存中是两个tensor，改变其中一个不会对另一个的数值造成影响，但是c拥有与a相同的梯度回传路线，保留了继续回传梯度的功能，但不再共享存储。

                3、 开始都对tensor这一数据类型比较熟悉，看博客有时会看到variable，形象地理解为这是另一种数据格式，tensor属于variable的一部分，tensor只包含数值，除了tensor，variable还有梯度值，梯度是否可计算等等属性（维度更广了），在迭代过程中从variable变量中取出tensor的操作时不可逆的，因为已经丢失了有关梯度的一切信息。

                """
                right_i, TP_i, TN_i, FP_i, FN_i = classify_result(result, y.cpu().detach().numpy(), print_result=False)
                TP+=TP_i.shape[0]; TN+=TN_i.shape[0]; FP+=FP_i.shape[0]; FN+=FN_i.shape[0]
            
            ACC[epoch] = 100*float(TP+TN)/float(len(testloader.dataset))
            TPR[epoch], FPR[epoch] = 100 * float(TP) / float(TP + FN + 0.00001), 100 * float(FP) / float(TN + FP + 0.00001)
            print("Epoch: %d/%d | G loss: %f | D loss: %f | ACC: %f | TPR: %f | FPR: %f" % (epoch+1, opt.n_epochs, g_loss.item(), d_loss.item(), ACC[epoch], TPR[epoch], FPR[epoch]))
            # save models
            torch.save(generator.state_dict(), '%s/model_epo_%03d_GLoss_%.4f_DLoss_%.4f_Generator.pth' % (os.path.join(outf,'model'), epoch+1, g_loss.item(), d_loss.item()))
            torch.save(discriminator.state_dict(), '%s/model_epo_%03d_GLoss_%.4f_DLoss_%.4f_Discrim.pth' % (os.path.join(outf,'model'), epoch+1, g_loss.item(), d_loss.item()))
            f.write("EPOCH = %03d, G_Loss: %.8f, D_Loss: %.8f, ACC: %f, TPR: %f, FPR: %f" %(epoch+1, g_loss.item(), d_loss.item(), ACC[epoch], TPR[epoch], FPR[epoch]))
            # "%03d"表示将一个整数格式化为至少3位宽度的字符串，不足的位数将用零进行填充。
            f.write('\n')
            f.flush()# f.flush() 这一行将文件缓冲区的内容刷新到硬盘，确保写入文件的数据被保存。


    f.close()
    return ACC