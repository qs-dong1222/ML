import numpy as np
import torch as t
from torch.autograd import Variable
import torch.optim as torchOpt
import matplotlib.pyplot as plt
import torch.utils.data as utdata




# hyper params
# n_G_datain_w = 4  # random input width size for Generator
# n_G_datain_h = 4  # random input height size for Generator
BATCH_SIZE = 64
EPOCH = 3001
D_k_loop = 1
n_G_code_len = 100




# data preparation
train_images = (np.load("./MNIST/npy-format-data/train_images.npy")-127.5)/127.5
train_labels = np.load("./MNIST/npy-format-data/train_labels.npy")
test_images = (np.load("./MNIST/npy-format-data/test_images.npy")-127.5)/127.5
test_labels = np.load("./MNIST/npy-format-data/test_labels.npy")



idxs = np.where(train_labels==8)
train_images = train_images[idxs]
train_labels = train_labels[idxs]



# G model definition
class ImageGen(t.nn.Module):
    def __init__(self):
        super(ImageGen, self).__init__()
        self.dense = t.nn.Sequential(
            t.nn.Linear(n_G_code_len, 1024*4*4),
            t.nn.BatchNorm1d(1024*4*4)
        )
        self.m = t.nn.Sequential(
            t.nn.ReLU(),
            t.nn.ConvTranspose2d(1024, 512, (self.k_wh(4, 8), self.k_wh(4, 8))),
            t.nn.BatchNorm2d(512),
            t.nn.ReLU(),
            t.nn.ConvTranspose2d(512, 256, (self.k_wh(8, 16), self.k_wh(8, 16))),
            t.nn.BatchNorm2d(256),
            t.nn.ReLU(),
            t.nn.ConvTranspose2d(256, 128, (self.k_wh(16, 22), self.k_wh(16, 22))),
            t.nn.BatchNorm2d(128),
            t.nn.ReLU(),
            t.nn.ConvTranspose2d(128, 1, (self.k_wh(22, 28), self.k_wh(22, 28))),
            t.nn.Tanh()
        )

    def forward(self, input):
        o1 = self.dense(input)
        o1 = o1.view(-1, 1024, 4, 4)
        output = self.m(o1)
        return output

    def k_wh(self, s_in, s_out):
        return s_out-s_in+1





# D model definition
class ImageDisc(t.nn.Module):
    def __init__(self):
        super(ImageDisc, self).__init__()
        self.conv = t.nn.Sequential(
            t.nn.Conv2d(1, 32, (5, 5)),  # 28x28 -> 24x24
            t.nn.LeakyReLU(negative_slope=0.2),

            t.nn.Conv2d(32, 64, (3, 3)),  # 24x24 -> 22x22
            t.nn.BatchNorm2d(64),
            t.nn.LeakyReLU(negative_slope=0.2),
            t.nn.MaxPool2d((2, 2), stride=2),  # 22x22 ->11x11

            t.nn.Conv2d(64, 128, (3, 3)),  # 11x11 -> 9x9
            t.nn.BatchNorm2d(128),
            t.nn.LeakyReLU(negative_slope=0.2),
            t.nn.MaxPool2d((2, 2), stride=2),  # 9x9 -> 4x4

            t.nn.Conv2d(128, 256, (3, 3)),  # 4x4 -> 2x2
            t.nn.BatchNorm2d(256),
            t.nn.LeakyReLU(negative_slope=0.2)  # final output : 256x2x2
        )

        self.fc = t.nn.Sequential(
            t.nn.Linear(256*2*2, 1),
            t.nn.BatchNorm1d(1),
            t.nn.Sigmoid()
        )

    def forward(self, input):
        convout = self.conv(input)
        fc_in = convout.view(-1, 256*2*2)
        output = self.fc(fc_in)
        return output










G = ImageGen().cuda()
D = ImageDisc().cuda()
print(G)
print(D)




# G_x_in = Variable(t.randn(BATCH_SIZE, n_G_code_len)).cuda()
# out = G(G_x_in)
#
# imgs = out.data.cpu().numpy().reshape(BATCH_SIZE, 28, 28)
# print("imgs.shape: ", imgs.shape)
# print("prediction: ", D(out))
# plt.imshow(imgs[0], cmap='gray')
# plt.show()






# optimizer
opt_D = torchOpt.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_G = torchOpt.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))










dataSet = utdata.TensorDataset(data_tensor=t.FloatTensor(train_images), target_tensor=t.FloatTensor(train_labels))
dataLoader = utdata.DataLoader(dataSet, batch_size=BATCH_SIZE, shuffle=True)





for epoch in range(EPOCH):
    for k, [batch_xs, batch_ys] in zip(range(D_k_loop), dataLoader):
        #G_x_in = Variable(t.randn(BATCH_SIZE, 1, n_G_datain_h, n_G_datain_w)).cuda()
        G_x_in = Variable(t.randn(BATCH_SIZE, n_G_code_len)).cuda()
        batch_xs_cuda = Variable(batch_xs).view(BATCH_SIZE, 1, 28, 28).cuda()

        fake_xs = G(G_x_in)
        Dout = D(batch_xs_cuda)
        loss_D = - t.mean(t.log(Dout) + t.log(1 - D(fake_xs)))
        opt_D.zero_grad()
        loss_D.backward(retain_graph=False)
        opt_D.step()

    G_x_in = Variable(t.randn(BATCH_SIZE, n_G_code_len)).cuda()
    fake_xs = G(G_x_in)
    # loss_G = t.mean(t.log(1 - D(fake_xs)))
    loss_G = t.mean(-t.log(D(fake_xs)))  # improved G loss
    opt_G.zero_grad()
    loss_G.backward(retain_graph=False)
    opt_G.step()


    if(epoch%10==0):
        print(loss_D)
    # if(epoch%500==0):
    #     G_x_in = Variable(t.randn(25, n_G_code_len)).cuda()
    #     imgs_25 = G(G_x_in).view(25, 28, 28)
    #     imgs_25 = imgs_25.data.cpu().numpy()
    #     col_size = 5
    #     grid_imgs = np.vstack(
    #         [np.hstack([img for img in imgs_25[s:s + col_size]]) for s in range(0, col_size * 5, col_size)])
    #
    #     plt.imshow(grid_imgs, cmap='gray')
    #     name = str(loss_G.data.cpu()[0])
    #     plt.savefig("./DCGAN_imgs/" + str(epoch) + "_" + name + ".jpg")
    #     print("epoch: ", epoch)



t.save(G, './DCGAN_models/Gen_model.pkl')
t.save(D, './DCGAN_models/Disc_model.pkl')






