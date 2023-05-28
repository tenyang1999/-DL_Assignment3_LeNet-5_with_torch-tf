from torch.utils.data import Dataset
import numpy as np
import os
import cv2
 
class TorchDataset(Dataset):
    def __init__(self, filename, image_dir, repeat=1):

        self.image_label_list = self.read_file(filename)
        self.image_dir = image_dir
        self.len = len(self.image_label_list)
        self.repeat = repeat

    def __getitem__(self, i):
        index = i % self.len
        image_name, label = self.image_label_list[index]
        image_path = os.path.join(self.image_dir, image_name)
        img = cv2.imread(image_path)
        img = self.data_preproccess(img)
        label=np.array(label)
        return img, label
 
    def __len__(self):
        if self.repeat == None:
            data_len = 10000000
        else:
            data_len = len(self.image_label_list) * self.repeat
        return data_len
 
    def read_file(self, filename):
        image_label_list = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                # rstrip：用来去除结尾字符、空白符(包括\n、\r、\t、' '，即：换行、回车、制表符、空格)
                content = line.rstrip().split(' ')
                name = content[0]
                labels = []
                for value in content[1:]:
                    labels.append(int(value))
                image_label_list.append((name, labels))
        return image_label_list
 
    def data_preproccess(self, img):

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img_gray, (32,32))
        return img