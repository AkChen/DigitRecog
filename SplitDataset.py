from torch.utils.data import Dataset
import json
import cv2
import numpy as np

import time




class BioModalDataset(Dataset):

    def __init__(self,file): # 'train' 'val' 'test'
        data_dict = np.load(file)[()]

        self.mnist_data = data_dict['a'] # channel
        self.svhn_data = data_dict['b']

        self.label_list = np.asarray(data_dict['label']).astype(int)

        self.length = len(self.label_list)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        # return (img_data,audio_data,label)

        mnist_data = self.mnist_data[item]
        svgn_data = self.svhn_data[item]
        label = self.label_list[item]

        return mnist_data,svgn_data,label


if __name__ == '__main__':
    #a = librosa.load()


    dataset = BioModalDataset('./data/biomodal/biomodal_train.npy')

    for i in range(len(dataset)):
        print(i)
        time_begin = time.time()
        img,aud,y = dataset[i] # 加载一个的时间 平均一个样本0.08秒 ，所以一个batch 16 数据加载约等于 1s 可以接受。如果dataloader 可以多线程
        time_end = time.time()
        print(time_end-time_begin)
        print(img.shape)
        print(aud.shape)
        print(y)






