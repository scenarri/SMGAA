import torch
import numpy as np
import os
import cv2
import scipy.signal.windows as wind
import math
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from models.arch import aconvnets, alexnet, densenet, mobilenetv2, resnet, shufflenetv2, squeezenet, vgg, utils
import random

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def Mkdir(path):
    if os.path.isdir(path):
        pass
    else:
        os.makedirs(path)


"""                                    alpha           L       gamma       faidot
      0        Dihedral                 1            >0       =0           =!0
      1        Trihedral                1            0        =!0          =0
      2        Cylinder                 0.5          >0       =0           =!0
      3        Top Hat                  0.5          0        =!0          =0
      4        Sphere                   0            0        =!0          =0
      5        Edge Broadside           0            >0       =0           =!0
      6        Edge Diffraction         -0.5         >0       =0           =!0
      7        Corner Diffraction       -1           0        =!0          =0

"""
"""theta: [A, x, y, alpha, gamma, L, faidot]
           0  1  2    3      4    5    6       """

#Rx = 0.3047
Ry = 0.3047
fc = 9.6e9
B = 0.591e9
M = 85
N = 85
Mz = 128
Nz = 128
Rx = 0.5*3e8/B

faim = 2*math.asin(0.25*3e8/(fc*Ry))

FAIM = torch.tensor(faim).float().cuda()
PFset = np.asarray([[0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0.5, 0, 0, 0],
                    [0, 0, 0, 0.5, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, -0.5, 0, 0, 0],
                    [0, 0, 0, -1, 0, 0, 0]])

Adjustablevec = np.asarray([[1, 1, 1, 0, 0, 1, 1],
                            [1, 1, 1, 0, 1, 0, 0],
                            [1, 1, 1, 0, 0, 1, 1],
                            [1, 1, 1, 0, 1, 0, 0],
                            [1, 1, 1, 0, 1, 0, 0],
                            [1, 1, 1, 0, 0, 1, 1],
                            [1, 1, 1, 0, 0, 1, 1],
                            [1, 1, 1, 0, 1, 0, 0]])

def adjV(type):
    if type == 0:
        Adjustablevec = np.asarray([1, 1, 1, 0, 1, 0, 0])
    else:
        Adjustablevec = np.asarray([1, 1, 1, 0, 0, 1, 1])
    return Adjustablevec

eps_max = np.array([10, 107, 107, 1.5, 2, 5, 1])
eps_min = np.array([0.001, 20, 20, -1.5, 0, 0, -1])
EPS_max = torch.from_numpy(eps_max).float().unsqueeze(0).cuda()
EPS_min = torch.from_numpy(eps_min).float().unsqueeze(0).cuda()
stepsize = torch.tensor([[[0.05, 0.5, 0.5, 0, 0.01, 0.025, 0.01]]]).cuda()

j1 = torch.complex(torch.tensor(0).float(),torch.tensor(1).float())
px = Rx * ((M - 1) / (Mz - 1))
py = Ry * ((N - 1) / (Nz - 1))
f_step = np.linspace(fc - B / 2, fc + B / 2, num=M, endpoint=True)
fai_step = np.linspace(-faim / 2, faim / 2, num=N, endpoint=True)
F_step = np.expand_dims(f_step,0).repeat(85,0).transpose()
FAI_step = np.expand_dims(fai_step,0).repeat(85,0)
F_step = torch.from_numpy(F_step).cuda()
FAI_step = torch.from_numpy(FAI_step).cuda()
window_x = np.array(wind.taylor(M, sll=35, norm=False)).reshape(1,M)
window_y = np.array(wind.taylor(N, sll=35, norm=False)).reshape(1,N)
window = np.matmul(window_x.T, window_y)
window = torch.tensor(window).float().cuda()
Fxn = (F_step / fc).cuda()
Fyn = (torch.linspace(-fc*torch.sin(FAIM/2), fc*torch.sin(FAIM/2), steps = N)/fc).expand(N,N).cuda()

def ASCimaging(theta):
    delta = torch.complex(torch.zeros([theta.shape[0], 1, M, N]).float(),torch.zeros([theta.shape[0], 1, M, N]).float()).cuda()
    batch = theta.shape[0]
    nb = theta.shape[1]
    # Fxn = F_step * torch.cos(FAI_step) / fc
    # Fyn = F_step * torch.sin(FAI_step) / fc
    for i in range(batch):
        for j in range(nb):
            E1 = theta[i][j][0] * (j1 * torch.sqrt(Fxn ** 2 + Fyn ** 2)) ** theta[i][j][3]
            E2 = torch.exp(((-j1) * 4 * math.pi * fc / 3e8) * (px * theta[i][j][1] * Fxn + py * theta[i][j][2] * Fyn))
            E3 = torch.sinc(0.5 * math.pi * torch.sqrt(Fxn ** 2 + Fyn ** 2) / torch.sin(FAIM / 2) * theta[i][j][5] * (
                    (N - 1) / (Nz - 1)) * torch.sin(torch.atan(Fyn / Fxn) - theta[i][j][6] * faim / 2))
            E4 = torch.exp(-theta[i][j][4] * Fyn)
            delta[i][0] += E1 * E2 * E3 * E4
    deltawindowed = torch.mul(delta, window)
    deltaoaded = F.pad(deltawindowed, (21, 22, 21, 22))
    deltaimg = torch.fft.ifft2(deltaoaded)
    deltaimgabs = torch.abs(deltaimg)
    return deltaimgabs

def generateima(theta, supress=1.):
    batch = theta.shape[0]
    nb = theta.shape[1]
    theta = theta.detach()
    img = torch.zeros([nb,batch,1,128,128])
    for n in range(nb):
        img[n] = ASCimaging(theta[:,n,:].unsqueeze(1))

    theta = theta.detach()
    for B in range(batch):
        for N in range(nb):
            max = torch.max(img[N][B]).item()
            if max > supress:
                scale = max + torch.rand(1).cuda()        #stategy need to be considered
                theta[B, N, 0] = theta[B, N, 0] / scale
            if theta[B, N, 1] > 0:
                theta[B, N, 1] = torch.fmod(theta[B, N, 1], 107)
            else:
                theta[B, N, 1] = torch.fmod(theta[B, N, 1] + 127, 107)
            if theta[B, N, 2] > 0:
                theta[B, N, 2] = torch.fmod(theta[B, N, 2], 107)
            else:
                theta[B, N, 2] = torch.fmod(theta[B, N, 2] + 127, 107)
    theta = torch.clamp(theta,EPS_min,EPS_max)
    theta.requires_grad = True
    deltaimgabs = ASCimaging(theta)
    return deltaimgabs, theta

def check_attack(output, target, last=False):
    flag = 0
    sucess = 0
    conf = torch.softmax(output, 1)
    label = conf.argmax(1)
    minconf = torch.min(conf[:,target])
    indice = conf[:, target].argmin().item()
    if last:
        misclasifiedidx = torch.nonzero(label - target)
        if misclasifiedidx.shape[0] != 0:
           sucess = 1
           targetconfcont = torch.zeros(misclasifiedidx.shape[0])
           for i in range(misclasifiedidx.shape[0]):
               subidx = misclasifiedidx[i]
               sublabel = label[subidx]
               targetconfcont[i] = conf[subidx,sublabel]
           targetidx = targetconfcont.argmax()
           indice = misclasifiedidx[targetidx].item()
           minconf = conf[indice,target]
        else:
           minconf = torch.min(conf[:, target])
           indice = conf[:, target].argmin().item()
    if minconf.item() <= 0.1:
        flag = 1
    return sucess, flag, minconf, indice, conf[:,target]

#def check_attack(output, target):
#    flag = 0
#    sucess = 0
#    conf = torch.softmax(output, 1)
#    label = conf.argmax(1)
    # minconf = torch.min(conf[:,target])
    # indice = conf[:, target].argmin().item()
#    misclasifiedidx = torch.nonzero(label-target)

#    if misclasifiedidx.shape[0] != 0:
#        sucess = 1
#        targetconfcont = torch.zeros(misclasifiedidx.shape[0])
#        for i in range(misclasifiedidx.shape[0]):
#            subidx = misclasifiedidx[i]
#            sublabel = label[subidx]
#            targetconfcont[i] = conf[subidx,sublabel]
#        targetidx = targetconfcont.argmax()
#        indice = misclasifiedidx[targetidx].item()
#        minconf = conf[indice,target]
#    else:
#        minconf = torch.min(conf[:, target])
#        indice = conf[:, target].argmin().item()
#
#    if minconf.item() <= 0.1:
#        flag = 1
#    return sucess, flag, minconf, indice, conf[:,target]


def init_theta(nb_asc, popbatch, seg = None):
    theta = np.zeros([popbatch ,nb_asc, 7])
    theta_idx = np.zeros([popbatch, nb_asc])
    type = np.zeros([popbatch, nb_asc])
    coord = torch.nonzero(seg).cpu().numpy()
    for b in range(popbatch):
        for n in range(nb_asc):
            PF_id = np.random.randint(len(PFset))
            theta_idx[b][n] = PF_id
            if PF_id == 1 or PF_id == 3 or PF_id == 4 or PF_id == 7:
                type[b][n] = 0
            else:
                type[b][n] = 1
            theta[b][n] = PFset[PF_id]
            init_theta = np.zeros_like(theta[b][n])
            Adjustablevec = adjV(type[b][n])
            for nn in range(len(init_theta)):
                #init_theta[nn] = np.random.rand() * (eps_max[nn] - eps_min[nn]) + eps_min[nn]
                if nn != 1 and nn != 2:
                    init_theta[nn] = np.random.rand() * (eps_max[nn] - eps_min[nn])  + eps_min[nn]
                [init_theta[1], init_theta[2]] = coord[np.random.randint(len(coord))] + [21,21]
            theta[b][n] += Adjustablevec * init_theta
            theta[b][n] = np.clip(theta[b][n], eps_min, eps_max)
    return torch.tensor(theta).float().cuda(), theta_idx, type

def crop_center(img,cropx=88,cropy=88):
    y = img.shape[-1]
    x = img.shape[-1]
    startx = x//2 - cropx//2
    starty = y//2 - cropy//2
    return img[:,:,starty:starty+cropy, startx:startx+cropx]


def showimg(data, line=0):
    img = data.detach().cpu().numpy()
    plt.imshow(img[line][0], vmin=0, vmax=1, cmap='gray')
    plt.show()

def load_models(model_name):
    if model_name == 'alex':
        net = alexnet.alexnet()
        net.load_state_dict(torch.load('./models/weights/alexnet.pth'))
    elif model_name == 'vgg':
        net = vgg.vgg11_bn()
        net.load_state_dict(torch.load('./models/weights/vgg16.pth'))
    elif model_name == 'res':
        net = resnet.resnet50()
        net.load_state_dict(torch.load('./models/weights/res50.pth'))
    elif model_name == 'dense':
        net = densenet.densenet121()
        net.load_state_dict(torch.load('./models/weights/densenet121.pth'))
    elif model_name == 'mobile':
        net = mobilenetv2.mobilenet_v2()
        net.load_state_dict(torch.load('./models/weights/mobilenetv2.pth'))
    elif model_name == 'aconv':
        net = aconvnets.AConvNets()
        net.load_state_dict(torch.load('./models/weights/aconvnet.pth'))
    elif model_name == 'shuffle':
        net = shufflenetv2.shufflenet_v2_x1_0()
        net.load_state_dict(torch.load('./models/weights/shufflenetv2.pth'))
    elif model_name == 'squeeze':
        net = squeezenet.squeezenet1_1()
        net.load_state_dict(torch.load('./models/weights/squeezenet.pth'))

    return net.eval().cuda()

class WrapperModel(nn.Module):
    def __init__(self, model, size, resize=True):
        super(WrapperModel, self).__init__()
        self.model = model
        self.resize = resize
        self.size = size
    def forward(self, x):
        if self.resize == True:
            x = self.Resize(x, self.size)
        return self.model(x)

    def Resize(self, img, size):
        img = F.interpolate(img, size=size, mode='bilinear', align_corners=True)
        return img