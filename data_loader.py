import os
import zipfile
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import numpy
from torch.utils.data import Dataset
labels_path=r".\list_attr_celeba-face.txt"
labels_df = pd.read_csv(labels_path)
#print(labels_df.head())

label_dict = {}
for i in range(1, len(labels_df)):
    label_dict[labels_df['202599'][i].split()[0]] = [x for x in labels_df['202599'][i].split()[1:]]

label_df = pd.DataFrame(label_dict).T
#print(label_df.head())

label_df.columns = (labels_df['202599'][0]).split()
label_df.replace(['-1'], ['0'], inplace = True)

#print(label_df.head())
train_df = label_df.iloc[0:14000,[20,21]]
print(train_df.head())

val_df = label_df.iloc[14000:20000,[20,21]]
print(val_df)

class MultiClassCelebA(Dataset):

    def __init__(self, dataframe, folder_dir, transform = None):

        self.dataframe = dataframe
        self.folder_dir = folder_dir
        self.transform = transform
        self.file_names = dataframe.index


    def __len__(self):
        return len(self.dataframe)


    def __getitem__(self, index):

        image = Image.open(os.path.join(self.folder_dir, self.file_names[index]))
        sample = {'image': image, 'label': self.dataframe.iloc[index].tolist()}
        if self.transform:
            image = self.transform(sample['image'])
            sample = {'image': image, 'label': self.dataframe.iloc[index].tolist()}

        return sample

if __name__=="__main__":
    #a = train_df.iloc[1]
    m = MultiClassCelebA(train_df,r'.\celeba\face')
    index = 13148
    print(m[index])
    img = m[index]["image"]

    img = cv2.cvtColor(numpy.asarray(img),cv2.COLOR_RGB2BGR)
    cv2.imshow("capture", img)
    cv2.waitKey(0)

