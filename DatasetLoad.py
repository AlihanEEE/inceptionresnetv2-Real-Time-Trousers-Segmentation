import os
import cv2
import numpy as np
from torch.utils.data import Dataset as BaseDataset
from glob import glob

# label:color_rgb:parts:actions
# background:0,0,0:: 
# space:140,120,240:: 113
# trapan:250,50,83:: 139


# Don't change this function
def default_augmentation(x, y, aug_count, preprocess_fn, img_size): #aug cont nerede kullanılıyor?
    x_a = preprocess_fn(x, img_size, 1) 
    y_a = preprocess_fn(y, img_size, 0)

    return np.expand_dims(np.array(x_a), 0), np.expand_dims(np.array(y_a), 0)


def default_preprocess(x, img_size, type): #img_size ve type neden kullanılmadı?
    #burada nasıl bir preproccess uygulanıyor image ve mask gelip hiçbir şey yapılmadan geri dönmüş?
    return x


class Dataset(BaseDataset):
    # CLASSES = ['space', 'trapan'] 
    def __init__(
            self,
            images_dir,
            masks_dir,
            img_size,
            classes=None,
            aug_count=1,
            aug_fn=default_augmentation, # burada fonksiyona allies ediyor galiba?
            preprocess_fn=default_preprocess
    ):
        self.images = glob(images_dir + "/*") #glob fonksiyonunda direction için terminalde kod nerede çalışıyorsa ondan sonraki kısımları belirtmek gerekiyor
        self.maskes = glob(masks_dir + "/*")

        self.aug_count = aug_count
        self.aug_fn = aug_fn
        self.preprocess_fn = preprocess_fn
        self.img_size = img_size

    def __getitem__(self, i):
        image = cv2.imread(self.images[i])
        masks = cv2.imread(self.maskes[i], 0) #siyah
        masks_all = []
        # np.unique(masks)

       
        s_mask = np.where(masks == 31, 255.0, 0.0)
        masks_all.append(s_mask)
    
        masks_all = np.array(masks_all)


        aug_x, aug_y = self.aug_fn(image, masks_all, self.aug_count, self.preprocess_fn, self.img_size) 

        return aug_x, aug_y

    def __len__(self):
        return len(self.images)
