import cv2
from glob import glob
import numpy as np


one_hot=np.zeros(255)
uniques= []
stored = []
print(one_hot)
print(one_hot.shape)

for ind , labels in enumerate(glob("png_masks/MASKS/*")):
    image = cv2.imread(labels)
    cv2.imshow("old",image)
    image=np.where(image == 31, 255.0, 0.0 )
    cv2.imshow("new",image)
    cv2.waitKey(0)
    # print(np.unique(image))
    uniq = np.unique(image)
    # uniques.append(uniq)
    
    for index, a  in enumerate (uniq):
        if a == 248:
            label = labels.split('/')[-1]
            # print(label)
            dir ="Ynew/"+label
            # print (dir)
            indexes = dir.split('_')[-1]
            stored.append(indexes)
            label = labels.split('/')[-1]
            dir ="Ynew/"+label
            # cv2.imwrite(dir,image)

        else: continue 
        # print(f'index:{ind}, dir: {dir}\n')
        temp = one_hot[a]
        temp = temp+1
        one_hot[a]= temp
        
    
print(one_hot)


# for i in stored:
#     print(i)
#     temp_img = cv2.imread("png_images/IMAGES/img_" + i)
#     cv2.imwrite("Xnew/IMAGES/img_"+ i,temp_img)
    # cv2.imshow("a",temp_img)

