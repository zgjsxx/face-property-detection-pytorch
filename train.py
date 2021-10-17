
from torch.autograd import Variable
import torch
import data_loader
import torch.nn.functional as F
from torchvision import transforms
import model
import numpy as np

from tqdm import tqdm
from pprint import pprint
import model2

from torch import optim
from torchvision.utils import make_grid
import time
import torch.nn as nn
tfms = transforms.Compose([transforms.Resize((224, 224)),
                           transforms.ToTensor()])

train_dl = data_loader.MultiClassCelebA(data_loader.train_df, r'.\celeba\face', transform = tfms)

valid_dl = data_loader.MultiClassCelebA(data_loader.val_df, r'.\celeba\face', transform = tfms)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def check_cuda():
    _cuda = False
    if torch.cuda.is_available():
        _cuda = True
    return _cuda


is_cuda = check_cuda()
#model_path = r'best_models/model-resnet-151-2.ptn'
model = model2.ResNet50(class_num=2)
model = model.to(device)
#model = torch.load(model_path)

if is_cuda:
    model.cuda()
train_dataloader = torch.utils.data.DataLoader(train_dl, shuffle = False, batch_size = 12, num_workers = 0)
valid_dataloader = torch.utils.data.DataLoader(valid_dl, shuffle = False, batch_size = 12, num_workers = 0)


def pred_acc(original, predicted):
    return torch.round(predicted).eq(original).sum().numpy()/len(original)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.9)

def fit_model(epochs, model, dataloader, phase = 'training', volatile = False):

    pprint("Epoch: {}".format(epochs))

    if phase == 'training':
        model.train()

    if phase == 'validataion':
        model.eval()
        volatile = True

    running_loss = []
    running_acc = []
    b = 0
    for i, data in enumerate(dataloader):

        target = data["label"]
        target = np.array(target, dtype=np.float32)
        target = torch.tensor(target)
        target = np.transpose(target, (1, 0))
        target = target.to(device)
        inputs = data['image'].to(device)

        inputs = Variable(inputs)

        if phase == 'training':
            optimizer.zero_grad()

        ops = model(inputs)

        acc_ = []
        for i, d in enumerate(ops, 0):

            acc = pred_acc(torch.Tensor.cpu(target[i]), torch.Tensor.cpu(d))
            acc_.append(acc)

        loss = criterion(ops, target)
        print("batch_loss: {:.8f}".format(loss.data))

        running_loss.append(loss.item())
        running_acc.append(np.asarray(acc_).mean())
        b += 1

        if phase == 'training':

            loss.backward()

            optimizer.step()

    total_batch_loss = np.asarray(running_loss).mean()
    total_batch_acc = np.asarray(running_acc).mean()


    pprint("{} loss is {} ".format(phase,total_batch_loss))
    pprint("{} accuracy is {} ".format(phase, total_batch_acc))

    return total_batch_loss, total_batch_acc



trn_losses = []; trn_acc = []
val_losses = []; val_acc = []
for i in tqdm(range(1, 20)):
    trn_l, trn_a = fit_model(i, model, train_dataloader)
    val_l, val_a = fit_model(i, model, valid_dataloader, phase = 'validation')
    trn_losses.append(trn_l); trn_acc.append(trn_a)
    val_losses.append(val_l); val_acc.append(val_a)


torch.save(model.state_dict(), "best_models/model-resnet-50-justface-state.ptn")
