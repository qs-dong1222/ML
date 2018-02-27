import numpy as np
import torch as t
from torch.autograd import Variable
import torch.optim as torchOpt
import matplotlib.pyplot as plt
import torch.utils.data as utdata




# hyper params
n_G_datain_w = 1  # random input width size for Generator
n_G_datain_h = 1  # random input height size for Generator
n_G_code_len = 100 # random input channel size for Generator
BATCH_SIZE = 64
EPOCH = 5001
D_k_loop = 5
gamma = 10



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
            t.nn.ConvTranspose2d(n_G_code_len, 256, (self.k_wh(1, 4), self.k_wh(1, 4))),
            t.nn.ReLU(),
            t.nn.ConvTranspose2d(256, 128, (self.k_wh(4, 8), self.k_wh(4, 8))),
            t.nn.BatchNorm2d(128),
            t.nn.ReLU(),
            t.nn.ConvTranspose2d(128, 64, (self.k_wh(8, 16), self.k_wh(8, 16))),
            t.nn.BatchNorm2d(64),
            t.nn.ReLU(),
            t.nn.ConvTranspose2d(64, 1, (self.k_wh(16, 28), self.k_wh(16, 28))),
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
            t.nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # 28x28 -> 14x14
            t.nn.LeakyReLU(negative_slope=0.2, inplace=True),

            t.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 14x14 -> 7x7
            t.nn.BatchNorm2d(128),
            t.nn.LeakyReLU(negative_slope=0.2, inplace=True),

            t.nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=1),  # 7x7 -> 4x4
            t.nn.BatchNorm2d(256),
            t.nn.LeakyReLU(negative_slope=0.2, inplace=True),

            t.nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0),  # 4x4 -> 1x1
        )

    def forward(self, input):
        output = self.conv(input)
        return output





G = ImageGen().cuda()
D = ImageDisc().cuda()
print(G)
print(D)




# G_x_in = Variable(t.randn(BATCH_SIZE, n_G_code_len, n_G_datain_h, n_G_datain_w)).cuda()
# out = G(G_x_in)

# imgs = out.data.cpu().numpy().reshape(BATCH_SIZE, 28, 28)
# print("imgs.shape: ", imgs.shape)
# print("prediction: ", D(out))
# plt.imshow(imgs[0], cmap='gray')
# plt.show()



# optimizer
opt_D = torchOpt.Adam(D.parameters(), lr=0.0001, betas=(0, 0.9))
opt_G = torchOpt.Adam(G.parameters(), lr=0.0001, betas=(0, 0.9))


# dataset preparation
dataSet = utdata.TensorDataset(data_tensor=t.FloatTensor(train_images), target_tensor=t.FloatTensor(train_labels))
dataLoader = utdata.DataLoader(dataSet, batch_size=BATCH_SIZE, shuffle=True)


for epoch in range(EPOCH):
    for k, [batch_xs, batch_ys] in zip(range(D_k_loop), dataLoader):
        ###################################### improved WGAN ######################################
        G_x_in = Variable(t.randn(BATCH_SIZE, n_G_code_len, n_G_datain_h, n_G_datain_w)).cuda()
        batch_xs_cuda = Variable(batch_xs).view(BATCH_SIZE, 1, 28, 28).cuda()
        
        opt_D.zero_grad()
        
        fake_xs = G(G_x_in)
        Wasserstein_D = t.mean(D(batch_xs_cuda)) - t.mean(D(fake_xs))
        Wasserstein_D.backward(-1 * t.ones(Wasserstein_D.size()).cuda())
        
        
        alpha = Variable(t.rand(fake_xs.size())).cuda()
        penalty_xs = (alpha * batch_xs_cuda - (1 - alpha) * fake_xs).detach()
        penalty_xs.requires_grad = True
        penalty_out = D(penalty_xs)
        penalty_xs_grad = t.autograd.grad(outputs=penalty_out, inputs=penalty_xs, 
                                           grad_outputs=t.ones(penalty_out.size()).cuda(), create_graph=True, only_inputs=True)
        penalty = gamma * t.mean(t.pow(t.norm(penalty_xs_grad[0], dim=1) - 1, 2))
        
        penalty.backward(t.ones(penalty.size()).cuda())
        
        # penalty.volatile = False
        loss_D = Wasserstein_D - penalty  # loss_D should be as bigger as possible
        opt_D.step()
        ###################################### improved WGAN ######################################



    G_x_in = Variable(t.randn(BATCH_SIZE, n_G_code_len, n_G_datain_h, n_G_datain_w)).cuda()
    fake_xs = G(G_x_in)
    loss_G = -1 * t.mean(D(fake_xs))  
    opt_G.zero_grad()
    loss_G.backward()
    opt_G.step()

    print("loss_D = %.6f  " % (loss_D.data.cpu()[0]), "loss_G = %.6f  " % (loss_G.data.cpu()[0]))

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
        plt.savefig("./DCGAN_imgs/" + str(epoch) + "_DCWGAN_GP" + ".jpg")
        print("epoch: ", epoch)



t.save(G, './DCGAN_models/Gen_model.pkl')
t.save(D, './DCGAN_models/Disc_model.pkl')

