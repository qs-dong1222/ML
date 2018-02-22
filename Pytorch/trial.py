import torch as t
from torch.autograd import Variable
import cv2
import opencvlib



# hyper params
K_SIZE = 15
DE_K_SIZE = 40



class ConvModel(t.nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.m1 = t.nn.Sequential(
            t.nn.Conv2d(1, 1, (K_SIZE, K_SIZE))
        )

    def forward(self, input):
        output = self.m1(input)
        return output





class DeconvModel(t.nn.Module):
    def __init__(self):
        super(DeconvModel, self).__init__()
        self.m = t.nn.Sequential(
            t.nn.ConvTranspose2d(1, 1, (DE_K_SIZE, DE_K_SIZE))
        )

    def forward(self, input):
        output = self.m(input)
        return output










convmodel = ConvModel()
deconvmodel = DeconvModel()






img = cv2.imread("./leona.jpg", 0).reshape(1, 1, 512, 512)

x = Variable(t.FloatTensor(img))
out = convmodel(x)

out_size = ((512 - K_SIZE)//1) + 1
outimg = out.data.numpy().reshape(out_size, out_size)
cv2.imshow("conv img", outimg)
print("conv img size: ", outimg.shape)



deconv_out = deconvmodel(out)
deconvout_size = (out_size-1)*1 + DE_K_SIZE
deoutimg = deconv_out.data.numpy().reshape(deconvout_size, deconvout_size)

cv2.imshow("deconv img", deoutimg)
print("deconv img size: ", deoutimg.shape)








opencvlib.WaitEscToExit()
cv2.destroyAllWindows()







