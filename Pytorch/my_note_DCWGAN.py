import numpy as np
import torch as t
from torch.autograd import Variable
import torch.optim as torchOpt
import matplotlib.pyplot as plt
import torch.utils.data as utdata




# hyper params
n_G_datain_w = 1  # random input width size for Generator
n_G_datain_h = 1  # random input height size for Generator
n_G_code_len = 100  # random input channel size for Generator
BATCH_SIZE = 64
EPOCH = 5001
D_k_loop = 5
clipval = 0.01
lr = 0.00005




# data preparation
train_images = (np.load("./MNIST/npy-format-data/train_images.npy")-127.5)/127.5
train_labels = np.load("./MNIST/npy-format-data/train_labels.npy")  # not need for training if we don't use the build-in loss function
test_images = (np.load("./MNIST/npy-format-data/test_images.npy")-127.5)/127.5
test_labels = np.load("./MNIST/npy-format-data/test_labels.npy")








# G model definition
class ImageGen(t.nn.Module):
    def __init__(self):
        super(ImageGen, self).__init__()
        self.m = t.nn.Sequential(
            t.nn.ConvTranspose2d(n_G_code_len, 256, (self.k_wh(1, 4), self.k_wh(1, 4)), bias=False),
            t.nn.ReLU(),
            t.nn.ConvTranspose2d(256, 128, (self.k_wh(4, 8), self.k_wh(4, 8)), bias=False),
            t.nn.BatchNorm2d(128),
            t.nn.ReLU(),
            t.nn.ConvTranspose2d(128, 64, (self.k_wh(8, 16), self.k_wh(8, 16)), bias=False),
            t.nn.BatchNorm2d(64),
            t.nn.ReLU(),
            t.nn.ConvTranspose2d(64, 1, (self.k_wh(16, 28), self.k_wh(16, 28)), bias=False),
            t.nn.Tanh()
        )

    def forward(self, input):
        output = self.m(input)
        return output

    def k_wh(self, s_in, s_out):
        return s_out-s_in+1



# D model definition
class ImageDisc(t.nn.Module):
    def __init__(self):
        super(ImageDisc, self).__init__()
        self.conv = t.nn.Sequential(
            t.nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),  # 28x28 -> 14x14
            t.nn.LeakyReLU(negative_slope=0.2, inplace=True),

            t.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),  # 14x14 -> 7x7
            t.nn.BatchNorm2d(128),
            t.nn.LeakyReLU(negative_slope=0.2, inplace=True),

            t.nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=1, bias=False),  # 7x7 -> 4x4
            t.nn.BatchNorm2d(256),
            t.nn.LeakyReLU(negative_slope=0.2, inplace=True),

            t.nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0, bias=False)  # 4x4 -> 1x1
        )

    def forward(self, input):
        output = self.conv(input)
        return output





G = ImageGen().cuda()
D = ImageDisc().cuda()
print(G)
print(D)



def weight_init(m):
    # weight_initialization: important for wgan
    class_name = m.__class__.__name__
    if class_name.find('Conv')!=-1:
        m.weight.data.normal_(0, 0.02)
        print("layer is conv")
    elif class_name.find('Norm')!=-1:
        m.weight.data.normal_(1.0, 0.02)
        print("layer is bn")

D.apply(weight_init)
G.apply(weight_init)





# optimizer
opt_D = torchOpt.RMSprop(D.parameters(), lr=lr)
opt_G = torchOpt.RMSprop(G.parameters(), lr=lr)




# dataset preparation
dataSet = utdata.TensorDataset(data_tensor=t.FloatTensor(train_images), target_tensor=t.FloatTensor(train_labels))
dataLoader = utdata.DataLoader(dataSet, batch_size=BATCH_SIZE, shuffle=True)





for epoch in range(EPOCH):
    for k, [batch_xs, batch_ys] in zip(range(D_k_loop), dataLoader):
        ###################################### improved WGAN ######################################
        G_x_in = Variable(t.randn(BATCH_SIZE, n_G_code_len, n_G_datain_h, n_G_datain_w)).cuda()
        batch_xs_cuda = Variable(batch_xs).view(BATCH_SIZE, 1, 28, 28).cuda()

        # 这里有必要解释一下D是怎么练的.
        # 在原WGAN的论文中, 给出的D的训练算法是 delta_D = grad[mean(D(true)) - mean(D(false))]
        # gd <- gd + delta_D, 这里是进行maximize操作.
        # grad可以进到[]分别运算: grad[mean(D(true)) - mean(D(false))] = grad[mean(D(true))] - grad[mean(D(false))]
        # 所以在true_xs_output做backward时,我们应该前面给正号,即t.ones(true_xs_output.size()), 同理可做fake_xs_output
        # 但是在对D模型作update时,我们要maximize而不是minimize,所以针对optimizer.step()只做minimize的特点,我们在整个式子
        # 前添上一个符号达到minimize的效果. 所以才有 -1 * t.ones(true_xs_output.size()), 对fake_xs同理
        opt_D.zero_grad()
        true_xs_output = t.mean(D(batch_xs_cuda))
        true_xs_output.backward(-1*t.ones(true_xs_output.size()).cuda())
        fake_xs = G(G_x_in).detach()
        fake_xs_output = t.mean(D(fake_xs))
        fake_xs_output.backward(t.ones(fake_xs_output.size()).cuda())
        opt_D.step()

        #################################################
        # weight clipping  for WGAN Lipschitz condition #
        #################################################
        for parm in D.parameters():
            parm.data.clamp_(min=-clipval, max=clipval)

        loss_D = true_xs_output - fake_xs_output
        ###################################### improved WGAN ######################################



    G_x_in = Variable(t.randn(BATCH_SIZE, n_G_code_len, n_G_datain_h, n_G_datain_w)).cuda()
    fake_xs = G(G_x_in)
    fake_xs_output = -t.mean(D(fake_xs))  # 直接在这里加上负号,后面grad时就不加了
    opt_G.zero_grad()
    fake_xs_output.backward(t.ones(fake_xs_output.size()).cuda())   # G 也用线性的loss
    opt_G.step()
    loss_G = fake_xs_output

    print("loss_D = %.6f  " % (loss_D.data.cpu()[0]), "loss_G = %.6f  " % (loss_G.data.cpu()[0]))
    # print("epoch: ", epoch)

    if(epoch%500==0):
        G_x_in = Variable(t.randn(25, n_G_code_len, n_G_datain_h, n_G_datain_w)).cuda()
        imgs_25 = G(G_x_in).view(25, 28, 28)
        imgs_25 = imgs_25.data.cpu().numpy()
        col_size = 5
        grid_imgs = np.vstack(
            [np.hstack([img for img in imgs_25[s:s + col_size]]) for s in range(0, col_size * 5, col_size)])

        plt.imshow(grid_imgs, cmap='gray')
        name = str(loss_G.data.cpu()[0])
        # plt.savefig("./DCGAN_imgs/" + str(epoch) + "_" + name + ".jpg")
        plt.savefig("./DCGAN_imgs/" + str(epoch) + ".jpg")
        print("epoch: ", epoch)



t.save(G, './DCGAN_models/Gen_model.pkl')
t.save(D, './DCGAN_models/Disc_model.pkl')

