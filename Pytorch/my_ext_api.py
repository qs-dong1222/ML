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