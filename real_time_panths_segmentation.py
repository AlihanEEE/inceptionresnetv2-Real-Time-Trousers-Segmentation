import os
from numpy import ndarray
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
import Epoch
import torch
from DatasetLoad import Dataset
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from glob import glob
from PreprocessAndAugmentation import augmentation, preprocess
import os, ssl
import time

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
        getattr(ssl, '_create_unverified_context',
                None)): ssl._create_default_https_context = ssl._create_unverified_context

cap = cv2.VideoCapture(0)

def one_channel_to_three(x):
    if x.shape[-1] != 3:
        x = cv2.merge((x, x, x))
    return x


def normalize_label(x):
        x = x.cpu().detach().numpy()
        if x.ndim == 4:
            x = x[0]
        if x.ndim == 3:
            x1 = x[0]
            x1 = one_channel_to_three(x1) * 255.0
        return x1


def resize_img(img):
    return cv2.resize(img, (640, 512))


ENCODER = 'inceptionresnetv2'
ENCODER_WEIGHTS = 'imagenet'
# CLASSES = ['boncuk']
ACTIVATION = 'sigmoid'
DEVICE = 'cuda'
INPUT_SHAPE = (832, 576)
loss = smp.losses.JaccardLoss(mode='binary')
loss.__name__ = 'JaccardLoss'

pipe_model = smp.Unet(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=1,
    activation=ACTIVATION,
)

pipe_model.load_state_dict(torch.load("Models//panth_seg_maskNormalized//best_epoch.pth"))
pipe_model = pipe_model.cuda()
prev_frame_time = 0
new_frame_time = 0

def detectPanths(frame, display = True):

    global new_frame_time
    global prev_frame_time
    new_frame_time = time.time()

    output_shape = frame.shape
    org_img = frame
    org_img = cv2.resize(org_img, (INPUT_SHAPE[0], INPUT_SHAPE[1]))
    img = org_img / 255.0
    x = np.moveaxis(img, -1, 0)
    x = torch.tensor(x)
    x = torch.unsqueeze(x, 0)

    
    pred = pipe_model.predict(x.float().cuda())
    pred = normalize_label(pred)

    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)

    return fps, pred



while True:
    
    ret, frame = cap.read() # video kameramızdan gelen resimleri frame e ve gelip gelmediğini return e aktaracak
    i=0
    avg_age=0
    
    fps ,frame = detectPanths(frame)

    _ , thresh = cv2.threshold(frame, 200, 255, cv2.THRESH_BINARY)
    cv2.putText(thresh,"FPS: " + fps, (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow("Segmented Panths", thresh)


    if cv2.waitKey(1) & 0xFF == ord("q"): break

    

cap.release() # capture ı serbest bırakalım
cv2.destroyAllWindows()

