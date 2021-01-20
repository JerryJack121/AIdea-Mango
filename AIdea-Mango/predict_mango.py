#將原始訓練資料夾內的芒果切割出來
from yolo_mango import YOLO
from PIL import Image
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2

yolo = YOLO()
#原始訓練資料夾
foldpath = r'D:\Dataset\Mango\Train'
#切割後輸出資料夾
savepath = os.path.abspath('D:/Dataset/Mango/Generate_Dataset/train')
img_list = os.listdir(foldpath)
# img_list = [r'D:\Dataset\Mango\Train\00006.jpg']
df = pd.read_csv("D:/DATASET/Mango/Test_UploadSheet.csv", encoding="utf8")
# imglist = np.array(df['image_id'])
for img_name in tqdm(img_list):
    image = Image.open(os.path.join(foldpath, img_name))  #YOLO要的讀檔格式
    r_image, _, totalmangoxywh = yolo.detect_image(image)
    # r_image.show()
    image = cv2.imread(os.path.join(foldpath, img_name))  #切割要的讀檔格式
    #切割原始圖片
    savepath = os.path.join(savepath)
    for mango in totalmangoxywh:
        x, y, w, h = mango
        print(x, y, w, h)
        crop_img = image[y:y + h, x:x + w]
        savepath = os.path.abspath(savepath + '1.jpg')
        cv_img = cv2.imencode('.jpg', crop_img)[1].tofile(savepath)
    # print(totalmangoxywh)

# 輸出結果csv
# result_df.to_csv('D:/Dataset/Mango/result.csv', index=None)
