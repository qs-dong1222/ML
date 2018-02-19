# -*- coding: utf-8 -*-
"""
Autograd: automatic differentiation
===================================

Central to all neural networks in PyTorch is the ``autograd`` package.
Let’s first briefly visit this, and we will then go to training our
first neural network.


The ``autograd`` package provides automatic differentiation for all operations
on Tensors. It is a define-by-run framework, which means that your backprop is
defined by how your code is run, and that every single iteration can be
different.

Let us see this in more simple terms with some examples.

Variable
--------

``autograd.Variable`` is the central class of the package. It wraps a
Tensor, and supports nearly all of operations defined on it. Once you
finish your computation you can call ``.backward()`` and have all the
gradients computed automatically.

You can access the raw tensor through the ``.data`` attribute, while the
gradient w.r.t. this variable is accumulated into ``.grad``.

.. figure:: /_static/img/Variable.png
   :alt: Variable

   Variable

There’s one more class which is very important for autograd
implementation - a ``Function``.

``Variable`` and ``Function`` are interconnected and build up an acyclic
graph, that encodes a complete history of computation. Each variable has
a ``.grad_fn`` attribute that references a ``Function`` that has created
the ``Variable`` (except for Variables created by the user - their
``grad_fn is None``).

If you want to compute the derivatives, you can call ``.backward()`` on
a ``Variable``. If ``Variable`` is a scalar (i.e. it holds a one element
data), you don’t need to specify any arguments to ``backward()``,
however if it has more elements, you need to specify a ``grad_output``
argument that is a tensor of matching shape.
"""

import torch
from torch.autograd import Variable

print("###############################################################\n"
      "前言介绍:\n"
      "Pytorch中的variable其实类似tf中的placeholder,但比placeholder要好理解,也更友好.\n"
      "既然variable是一个占位符外包的概念,那么我们就要向这个外包中添加一个内容,这个内容会被包装成一个\n"
      "变量,也就是说,这时,外包中的内容会有变化的性质,这恰恰是我们做grad时想要的.\n"
      "所以,我们可以向这个外包中放入一个tensor,将其包装成一个变量对象.\n"
      "###############################################################\n")



###############################################################
# Create a variable:
x = Variable(torch.ones(2, 2), requires_grad=True)
print("\n\nx = Variable(torch.ones(2, 2), requires_grad=True):\n", x,
      "以torch.ones(2, 2)为内容创建一个variable对象,并使能其梯度计算功能,\n"
      "这样一来,在后续计算中它的grad属性就会被计算并更新")



###############################################################
# Do an operation of variable:
y = x + 2
print("\n\n\ny = x + 2:\n", y,
      "x在上面定义为一个variable,这里新定义的y是包含x的一个表达式,或者说是一个包含x的函数,\n"
      "那么y也自然的会成为一个variable,很像C++中的右值引用的概念,即表达式做变量引用. \n"
      "与直接定义的variable不一样的是,这个由表达式定义的variable不是\"直接嫡传\"的,所以y的grad_fn属性会有值,\n"
      "其表示x+2这个表达式.")




###############################################################
# ``y`` was created as a result of an operation, so it has a ``grad_fn``.
print("\n\n\ny.grad_fn:\n", y.grad_fn,
      "\ny是表达式生成的一个variable,所以他的grad_fn有值,其值是一个表达式函数对象\n")




###############################################################
# Do more operations on y
z = y * y * 3
print("\n\n\nz = y * y * 3:\n", z,
      "这里 * 运算是element-wise运算,并不是矩阵乘法,矩阵乘法是torch.mm(),所以这里我们有3*3*3=27 on each element\n")


import numpy as np
h = torch.FloatTensor(np.arange(9))
h = h.view(3, 3)
print("\n\n\nh = torch.FloatTensor(np.arange(9)),", "h = h.view(3,3):", h)
print("h.mean(): all-element mean", h.mean())
print("h.mean(dim=0): vertical mean", h.mean(dim=0))
print("h.mean(dim=1): horizontal mean", h.mean(dim=1))
print("h.mean(dim=0, keepdim=True): vertical mean without dim reduction", h.mean(dim=0, keepdim=True))



###############################################################
# Gradients
# ---------
# let's backprop now
# ``out.backward()`` is equivalent to doing ``out.backward(torch.Tensor([1.0]))``

print("\n\n\nbuilt a function 'out = z.mean()', which is a function of 'z' and 'z' is a function\n"
      "of 'y', then 'y' is a function of x. So here we have a function chain x->out.")
out = z.mean()
print("out = z.mean():\n", out)

print("\n\n\nhere we do the backpropagation 'out.backward()' on 'out' function. which computes all the grad of those"
      "\nvariables that are related to this 'out' function(here we have function 'x', 'y', 'z' related) and whose"
      "\nattribute 'requires_grad' is set to True or this is a expression variable created from other variables")
out.backward()

###############################################################
# print gradients d(out)/dx
#
print("\nThe whole relation from out to x is: out=(1/4)*z, z=3y^2, y=x+2. so out=(3/4)*(x+2)^2\n"
      "then δout/δx = (3/4)*(x+2)^2. Let's see the attribute 'grad' of x(x.grad) of x w.r.t the \n"
      "latest backpropagation, that is out.backward().")
print("x.grad:\n", x.grad)


###############################################################
# You should have got a matrix of ``4.5``. Let’s call the ``out``
# *Variable* “:math:`o`”.
# We have that :math:`o = \frac{1}{4}\sum_i z_i`,
# :math:`z_i = 3(x_i+2)^2` and :math:`z_i\bigr\rvert_{x_i=1} = 27`.
# Therefore,
# :math:`\frac{\partial o}{\partial x_i} = \frac{3}{2}(x_i+2)`, hence
# :math:`\frac{\partial o}{\partial x_i}\bigr\rvert_{x_i=1} = \frac{9}{2} = 4.5`.

###############################################################
# You can do many crazy things with autograd!


x = torch.randn(3)
x = Variable(x, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

###############################################################
#
gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
y.backward(gradients)

print(x.grad)

###############################################################
# **Read Later:**
#
# Documentation of ``Variable`` and ``Function`` is at
# http://pytorch.org/docs/autograd
