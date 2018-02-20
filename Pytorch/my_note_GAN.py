import numpy as np
import torch as t
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as utdata
import matplotlib.pyplot as plt


# hyper parameters
N_FEATURES = 20
N_TRUE_SAMPLES = 1000
N_RANDOM_IN = 5
EPOCH = 4000
K = 3
N_NEURONS = 128








def s_y(s_x):
    noise = np.random.uniform(0, 0.5, size=N_TRUE_SAMPLES)[:, np.newaxis]
    # s_y = -2*noise + (3.1 * s_x + 1.7 * s_x ** 2 - 11.6 * s_x ** 5)
    s_y = s_x**2 + noise
    return s_y

s_x = np.vstack([np.linspace(-1, 1, N_FEATURES) for i in range(N_TRUE_SAMPLES)])











class Generator(t.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.gen_1 = t.nn.Sequential(
            t.nn.Linear(N_RANDOM_IN, N_NEURONS),
            t.nn.BatchNorm1d(N_NEURONS),
            #t.nn.Tanh(),
            t.nn.ReLU(),
        )
        self.gen_2 = t.nn.Sequential(
            t.nn.Linear(N_NEURONS, N_NEURONS),
            t.nn.BatchNorm1d(N_NEURONS),
            #t.nn.Tanh()
            t.nn.ReLU()
        )
        self.gen_3 = t.nn.Sequential(
            t.nn.Linear(N_NEURONS, N_NEURONS),
            t.nn.ReLU(),
            t.nn.Linear(N_NEURONS, N_FEATURES)
        )

    def forward(self, input):
        G_out_1 = self.gen_1(input)
        G_out_2 = self.gen_2(G_out_1)
        G_out = self.gen_3(G_out_2)
        return G_out







class Discriminator(t.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.dis_1 = t.nn.Sequential(
            t.nn.Linear(N_FEATURES, N_NEURONS),
            #t.nn.BatchNorm1d(N_NEURONS),
            t.nn.Dropout(),
            t.nn.ReLU()
        )
        self.dis_2 = t.nn.Sequential(
            t.nn.Linear(N_NEURONS, N_NEURONS),
            #t.nn.BatchNorm1d(N_NEURONS),
            t.nn.Dropout(),
            t.nn.ReLU()
        )
        self.dis_3 = t.nn.Sequential(
            t.nn.Linear(N_NEURONS, 1),
            t.nn.Sigmoid()
        )

    def forward(self, input):
        D_out_1 = self.dis_1(input)
        D_out_2 = self.dis_2(D_out_1)
        D_out = self.dis_3(D_out_2)
        return D_out_2, D_out






G = Generator().cuda()
D = Discriminator().cuda()



# optimizer
opt_D = t.optim.Adam(D.parameters(), lr=0.001)
opt_G = t.optim.Adam(G.parameters(), lr=0.001)


# trial
loss_new = t.nn.MSELoss().cuda()


plt.ion()
plt.figure(1)

for epoch in range(EPOCH):
    ######  create fake data  ######
    random_in_xs = t.randn(N_TRUE_SAMPLES, N_RANDOM_IN)
    v_random_in_xs = Variable(random_in_xs).cuda()
    ys = s_y(s_x)
    t_xs = t.FloatTensor(ys)
    v_xs = Variable(t_xs).cuda()
    ######  create fake data  ######
    for i in range(K):
    # while(True):
        Gout = G(v_random_in_xs)
        Dout_inter, Dout = D(v_xs)
        loss_D = - t.mean(t.log(Dout) + t.log(1 - D(Gout)[1]))
        # loss = t.mean(t.pow(D(Gout)[0] - Dout_inter, 2))
        opt_D.zero_grad()
        loss_D.backward(retain_graph=True)
        # loss.backward()
        opt_D.step()
        # if(loss_D.data.cpu()[0] - 0.5 <= 0.0001):
        #     break



    ######  create fake data  ######
    random_in_xs = t.randn(N_TRUE_SAMPLES, N_RANDOM_IN)
    v_random_in_xs = Variable(random_in_xs).cuda()
    ######  create fake data  ######
    Gout = G(v_random_in_xs)
    # loss_G = t.mean(t.log(1 - D(G(v_random_in_xs))[1]))
    loss_G = t.mean(-t.log(D(G(v_random_in_xs))[1]))  # improved G loss
    opt_G.zero_grad()
    loss_G.backward(retain_graph=True)
    opt_G.step()



    if (epoch % 10 == 0):
        plt.cla()
        plt.plot(s_x[0], ys[0])
        Gout = G(v_random_in_xs[0:10])
        Gout = Gout.cpu().data.numpy()[0]
        plt.plot(s_x[0], Gout)
        plt.text(0.5, 0.2, "loss G: %.4f" % (loss_G))
        plt.text(0.5, 0.5, "loss D: %.4f" % (loss_D))
        plt.text(0.5, 0.8, "epoch: %d" % (epoch))
        print(epoch)
        plt.pause(0.01)
