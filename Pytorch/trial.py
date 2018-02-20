import torch as t
from torch.autograd import Variable


# class Model(t.nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.m1 = t.nn.Sequential(
#             t.nn.Linear(10, 15),
#             t.nn.BatchNorm1d(15),
#             t.nn.ReLU(),
#             t.nn.Linear(15, 20),
#             t.nn.BatchNorm1d(20),
#             t.nn.ReLU(),
#             t.nn.Linear(20, 5)
#         )
#
#     def forward(self, input):
#         output = self.m1(input)
#         return output
#
#
# model = Model().cuda()
# print(model)
#
#
#
#
#
# x = Variable(t.randn(1000, 10)).cuda()
#
# o = model(x)
# print(o)






x = t.randn(5)
vx = Variable(x).cuda().view(-1, 5)
print(vx)
