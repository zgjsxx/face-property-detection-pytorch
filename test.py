import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import numpy as np
import model2
labels = ['Male','Mouth_Slightly_Open']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model2.ResNet50(class_num=2)
model.to(device)
model_path = r'best_models/model-resnet-50-justface-state.ptn'
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint)
model = model.eval()


def get_tensor(img):
    tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    return tfms(Image.open(img)).unsqueeze(0)


def predict(img, label_lst, model):
    tnsr = get_tensor(img)
    op = model(tnsr.to(device))
    op_b = torch.round(op)
    print(op_b)
    op_b_np = torch.Tensor.cpu(op_b).detach().numpy()

    preds = np.where(op_b_np == 1)[1]

    sigs_op = torch.Tensor.cpu(torch.round((op) * 100)).detach().numpy()[0]

    o_p = np.argsort(torch.Tensor.cpu(op).detach().numpy())[0][::-1]

    #     print("Argsort: {}".format(o_p))
    #     print("Softmax: {}".format(sigs_op))

    #     print(preds)

    label = []
    for i in preds:
        label.append(label_lst[i])

    arg_s = {}
    for i in o_p:
        #         arg_s.append(label_lst[int(i)])
        arg_s[label_lst[int(i)]] = sigs_op[int(i)]

    return label, list(arg_s.items())[:]

print(predict(r'.\test_img\144.jpg', labels, model))
#print(predict(r'.\celeba\img_align_celeba\199950.jpg', labels, model))