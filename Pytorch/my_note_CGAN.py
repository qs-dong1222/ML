import torch as t
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as utdata
import my_ext_api



# hyper param
LR = 0.01
BATCH_SIZE = 100
DECAY = 1.00004
MOMENTUM = 0.5
EPOCH = 50001
K = 1




if __name__ == '__main__':
    # data preparation
    train_images = (np.load("./MNIST/npy-format-data/train_images.npy"))/255
    train_labels = np.load("./MNIST/npy-format-data/train_labels.npy")  # not need for training if we don't use the build-in loss function
    # test_images = (np.load("./MNIST/npy-format-data/test_images.npy")-127.5)/127.5
    # test_labels = np.load("./MNIST/npy-format-data/test_labels.npy")



    # one hot
    labels_one_hot = t.zeros(len(train_labels), 10).scatter_(1, t.LongTensor(train_labels[:, np.newaxis]), 1)
    # np.random.shuffle(train_labels)
    # fake_labels_one_hot = t.zeros(len(train_labels), 10).scatter_(1, t.LongTensor(train_labels[:, np.newaxis]), 1)




    from CGAN_model import *
    G = Gen(z_dim=100, y_dim=10).cuda()
    D = Dis(x_dim=784, y_dim=10).cuda()
    init_model_weight(G, xavier_normal)
    init_model_weight(D, xavier_normal)
    print(G)
    print(D)





    # optimizer
    # opt_D = t.optim.SGD(D.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=DECAY)
    # opt_G = t.optim.SGD(G.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=DECAY)
    opt_D = t.optim.Adam(D.parameters())
    opt_G = t.optim.Adam(G.parameters())



    # dataset preparation
    dataSet = utdata.TensorDataset(data_tensor=t.FloatTensor(train_images), target_tensor=t.FloatTensor(labels_one_hot))
    dataLoader = utdata.DataLoader(dataSet, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


    import matplotlib.pyplot as plt
    w_gp_loss = Wasserstein_GP_Loss().cuda()

    for epoch in range(EPOCH):
        for k, [batch_xs, batch_ys] in zip(range(K), dataLoader):
            noise = noise = Variable(t.rand(BATCH_SIZE, 100)).cuda()
            true_xs, true_ys = Variable(batch_xs).view(-1, 784).cuda(), Variable(batch_ys).cuda()
            fake_xs = G(t.cat((noise, true_ys), dim=1)).detach()
            
            opt_D.zero_grad()
            # train with fakes
            fake_pred = - t.log(  1-D(t.cat((fake_xs, true_ys), dim=1))  ).mean()
            fake_pred.backward(t.ones(fake_pred.size()).cuda())
            # train with trues
            true_pred = - t.log(  D(t.cat((true_xs, true_ys), dim=1)) ).mean()
            true_pred.backward(t.ones(true_pred.size()).cuda())
            opt_D.step()
            # print(-fake_pred-true_pred)
            
            # noise = Variable(t.rand(BATCH_SIZE, 100)).cuda()
            # true_xs, true_ys = Variable(batch_xs).view(-1, 784).cuda(), Variable(batch_ys).cuda()
            # opt_D.zero_grad()
            # fake_xs = G(t.cat((noise, true_ys), dim=1))
            # w_loss = w_gp_loss(D_model=D, fxs=t.cat((fake_xs, true_ys), dim=1), rxs=t.cat((true_xs, true_ys), dim=1))
            # w_loss.backward()
            # opt_D.step()
            # print(w_loss)



        opt_G.zero_grad()
        noise = Variable(t.rand(BATCH_SIZE, 100)).cuda()
        fake_xs = G(t.cat((noise, true_ys), dim=1))
        G_loss = t.log(  1 - D(t.cat((fake_xs, true_ys), dim=1)) ).mean()
        G_loss.backward()
        opt_G.step()

        print("loss_G = %.6f  " % (G_loss.data.cpu()[0]))

        if (epoch % 500 == 0):
            G_x_in = Variable(t.randn(25, 100)).cuda()
            label = [1, 2, 3, 4, 5] * 5
            label = np.array(label)
            label = Variable(t.zeros(len(label), 10).scatter_(1, t.LongTensor(label[:, np.newaxis]), 1)).cuda()

            imgs_25 = G( t.cat((G_x_in, label), dim=1) ).view(25, 28, 28)
            imgs_25 = imgs_25.data.cpu().numpy()
            col_size = 5
            grid_imgs = np.vstack(
                [np.hstack([img for img in imgs_25[s:s + col_size]]) for s in range(0, col_size * 5, col_size)])

            plt.imshow(grid_imgs, cmap='gray')
            plt.savefig("./DCGAN_imgs/CGAN/CGAN_" + str(epoch) + ".jpg")
            print("epoch: ", epoch)




    t.save(G, './DCGAN_models/CGAN_G.pkl')
    t.save(D, './DCGAN_models/CGAN_D.pkl')

