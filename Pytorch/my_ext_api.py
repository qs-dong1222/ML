import torch as t
from torch.autograd import Variable
import numpy as np





########################################################################
# Function to calculate the gradient between output and input variable
# ret_val:
#   grad - output gradient calculated w.r.t input variable
# params:
#   variable_in - input torch Variable
#   custom_output_func - customized output function that is defined from a torch loss function class
# example:
#   class Foo(t.nn.Module):
#       def __init__(self):
#           super(Foo, self).__init__()
#           return
#       def forward(self, x):
#           output = x**2 + 3*x
#           return output
#
#   x = Variable(t.FloatTensor([1, 2, 3]), requires_grad=True)
#   z = x/2 + 1
#   y_func = Foo()
#   grad = calcGrad_wrt_varin(z, y_func)
########################################################################
def calcGrad_wrt_varin(variable_in, custom_output_func):
    import torch
    x = variable_in.detach()  # create a new graph brach whose input leaf variables are variable_in
    x.requires_grad = True  # set input leaf variables to be gradient computable
    y = custom_output_func(x)  # calculate the output variable
    y.backward(torch.ones(y.size()))  # calculate the output variable gradient w.r.t input leaf variables
    grad = x.grad
    return grad









#-------------------------------------------------------------------
#      Wasserstein improved loss function (support cuda)
# NOTE: The D model gradient w.r.t intermediate penalty is independently
# computed in this function.
# retval: - wloss: the loss of wasserstein distance which has already
#                  been negatived for optimizer to do grad descent.
# params: - D_model: discriminator model
#         - fxs: fake data set generated from G model
#         - rxs: real data set provided for D
#-------------------------------------------------------------------
class Wasserstein_GP_Loss(t.nn.Module):
    def __init__(self):
        super(Wasserstein_GP_Loss, self).__init__()
        return

    def forward(self, D_model, fxs, rxs, LAMBDA=10):
        D = D_model
        W_loss = - (D(rxs).mean() - D(fxs).mean())  # add negative sign for later optimizer to minimizr
        alpha = Variable(t.rand(fxs.size()))
        if(W_loss.is_cuda):
            alpha = alpha.cuda()
        pxs = (alpha * rxs - (1 - alpha) * fxs).detach()
        pxs.requires_grad = True
        pout = D(pxs)
        one = t.ones(pout.size())
        if (pout.is_cuda):
            one = one.cuda()
        pgrad = t.autograd.grad(outputs=pout, inputs=pxs, grad_outputs=one, create_graph=True, only_inputs=True)[0]
        grad_penalty = ((pgrad.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        grad_penalty.backward()
        return W_loss










#-----------------------------------------------------------------
#               Maxout activation layer
# The maxout activation layer which has a super powerful fitting
# performance. But it has a problem of increasing model arch size.
# params - in_features: number of input features for this layer
#          out_feafures: number of output features for this layer
#          k: number of nuerons to be maxout from for each outputs
# retval - output: the final result
#-----------------------------------------------------------------
class Maxout(t.nn.Module):
    def __init__(self, in_features, out_features, k):
        super(Maxout, self).__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.k = k
        self.m = t.nn.Linear(in_features, k*out_features)

    def forward(self, x):
        if(x.is_cuda):
            self.m = self.m.cuda()
        a = self.m(x).view(-1, self.out_features, self.k)
        output = t.max(a, dim=2)[0]
        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features)  \
            + ', maxout_k=' + str(self.k) + ')'





#-----------------------------------------------------------------
#           init functions to be 'apply'ed on models\
#-----------------------------------------------------------------
def xavier_uniform_init(m, bias=False):
    print("parameters of '" + m.__class__.__name__ + "' has been xavier_uniform initialized")
    if hasattr(m, 'weight'):
        size = m.weight.data.size()
        n_dim = len(m.weight.data.size())
        if(n_dim==1):
            m.weight.data = m.weight.data.view(1, size[0])
        m.weight.data = t.nn.init.xavier_uniform(m.weight.data)
        if (n_dim == 1):
            m.weight.data = m.weight.data.view(size[0])
    if(bias):
        if hasattr(m, 'bias'):
            size = m.bias.data.size()
            n_dim = len(m.bias.data.size())
            if(n_dim==1):
                m.bias.data = m.bias.data.view(1, size[0])
            m.bias.data = t.nn.init.xavier_uniform(m.bias.data)
            if (n_dim == 1):
                m.bias.data = m.bias.data.view(size[0])





def xavier_normal_init(m, bias=False):
    print("parameters of '" + m.__class__.__name__ + "' has been xavier_normal initialized")
    if hasattr(m, 'weight'):
        size = m.weight.data.size()
        n_dim = len(m.weight.data.size())
        if(n_dim==1):
            m.weight.data = m.weight.data.view(1, size[0])
        m.weight.data = t.nn.init.xavier_normal(m.weight.data)
        if (n_dim == 1):
            m.weight.data = m.weight.data.view(size[0])
    if (bias):
        if hasattr(m, 'bias'):
            size = m.bias.data.size()
            n_dim = len(m.bias.data.size())
            if(n_dim==1):
                m.bias.data = m.bias.data.view(1, size[0])
            m.bias.data = t.nn.init.xavier_normal(m.bias.data)
            if (n_dim == 1):
                m.bias.data = m.bias.data.view(size[0])







#-----------------------------------------------------------------
#           init models with given init_function
# params - model: torch Module
#        - init_func: can be any init function above
# example:
#        init_model_weight(any_model, xavier_uniform_init)
#-----------------------------------------------------------------
def init_model_weight(model, init_func):
    model.apply(init_func)


































#----------------------
#   Alias
#----------------------
xavier_uniform = xavier_uniform_init
xavier_normal = xavier_normal_init
