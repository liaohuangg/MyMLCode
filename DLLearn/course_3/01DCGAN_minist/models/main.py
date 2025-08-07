from data import ReadData
from model import Discriminator, Generator, weights_init
from net import DCGAN
import torch

ngpu=1
ngf=64
ndf=64
nc=1
nz=100
lr=0.003
beta1=0.5
datapath="./data"
batchsize=100

model_save_path="./models/"
figure_save_path="./figures/"

device = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')

dataset=ReadData(datapath)
dataloader=dataset.getdataloader(batch_size=batchsize)

G = Generator(nz,ngf,nc).apply(weights_init)
D = Discriminator(ndf,nc).apply(weights_init)

dcgan=DCGAN(nz, lr,beta1,device, model_save_path,figure_save_path,G, D, dataloader)
dcgan.train(num_epochs=5)
# dcgan.test()



