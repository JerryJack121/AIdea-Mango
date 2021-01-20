import pandas as pd
import numpy as np
import os
import cv2
from tqdm import tqdm

# 讀取資料集標籤檔
df = pd.read_csv(r'D:\dataset\mango\step_3\Test_mangoXYWH.csv',
                 encoding="utf8")
# 圖片路徑
path = r'D:\dataset\mango\step_3\Test'
# 輸出資料夾路徑
savepath = os.path.abspath(r'D:\dataset\mango\step_3\test')
picture_num = 0
# 逐列遍歷dataframe
with tqdm(total=len(df)) as pbar:
    for index, row in df.iterrows():
        data_row = df.loc[index].values[0:5]  #讀入每一行數值
        file_name = data_row[0]  #第0行為檔案圖片名稱
        col = 1
        count = 0
        # print(file_name)
        # 將一顆芒果中多個瑕疵部位分開裁減
        # while not np.isnan(data_row[col]):
        count += 1
        data = np.array(data_row[col:col + 5], dtype=int)
        # label = data_row[col+4]
        col = col + 5
        # 讀取圖檔
        img = cv2.imread(os.path.join(path, file_name))
        # 裁切圖片
        x, y, w, h = data
        # print(x,y,w,h)
        crop_img = img[y:y + h, x:x + w]
        # # 顯示圖片
        # cv2.imshow("crop_img", crop_img)
        # cv2.waitKey(0)
        # 根據類別寫入裁剪後的圖片
        # savepath = os.path.join('D:/DATASET/Mango/Generate_Dataset/test')
        if not os.path.isdir(savepath):
            os.mkdir(savepath)
        saveimg = os.path.join(savepath, file_name[:-4]) + '.jpg'
        cv_img = cv2.imencode('.jpg', crop_img)[1].tofile(saveimg)
        picture_num = picture_num + 1

        # 更新進度條
        pbar.update(1)
        pbar.set_description('Dataset')
        pbar.set_postfix(**{
            'File_name': file_name,
            'Picture_num': picture_num
        })
