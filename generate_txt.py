import pandas as pd
import numpy as np
import os
from tqdm import tqdm

csv_path = 'D:/dataset/Mango/train.csv'
# folder_path = 'D:/dataset/Mango/Train'

save_path = 'D:/dataset/Mango/txt/train_defect.txt'
file_path = 'D:/Dataset/Mango/Train'


# 寫入txt
def data2txt(file, data, label):
    x, y, w, h = data
    box = '%s,%s,%s,%s,%s' % (x, y, x + w, y + h, label)
    file.write(box)

def defect2txt():
    lable_dict = {
        '不良-乳汁吸附': 0,
        '不良-機械傷害': 1,
        '不良-炭疽病': 2,
        '不良-著色不佳': 3,
        '不良-黑斑病': 4,
    }  #Label Encoder
    f = open(save_path, 'w')  #寫入txt檔案位置

    df = pd.read_csv(csv_path, encoding="utf8", header=None)  #讀取metadata
    with tqdm(total=len(df)) as pbar:
        for index, row in df.iterrows():
            data_row = df.loc[index].values[0:-1]  #讀入每一行數值
            file_name = data_row[0]  #第0行為檔案圖片名稱
            f.write(file_path + '/' + file_name)  #兩個框座標中間的空白
            col = 1

            try:
                while not np.isnan(data_row[col]):
                    data = np.array(data_row[col:col + 4], dtype=int)
                    label = data_row[col + 4]
                    label = lable_dict[label]  #Label Encoding
                    f.write(' ')
                    data2txt(f, data, label)
                    col = col + 5  #讀取下一個瑕疵座標
                f.write('\n')  #換行後繼續寫入下一張圖片
            # 讀取最長的那一列會超出csv範圍
            except Exception as e:
                print(file_name, '發生錯誤\n', e)
                break
                f.write('\n')
            # 更新進度條
            pbar.update(1)
            pbar.set_description('defect2txt')
            pbar.set_postfix(**{
                'File_name': file_name,
            })
    f.close()

def mango2txt():
    num = 100
    save_path = r'D:\Dataset\Mango\txt\train_mango.txt'
    csv_path = r'D:\Dataset\Mango\Test_mangoXYWH.csv'
    df = pd.read_csv(csv_path, encoding="utf8")  #讀取metadata
    df = df[:num]
    # count = 0
    f = open(save_path, 'w')  #寫入txt檔案位置
    with tqdm(total=len(df)) as pbar:
        for index, row in df.iterrows():
            # if count == 100:
            #     break
            # count += 1
            data_row = df.loc[index].values[0:5]  #讀入每一行數值
            file_name = data_row[0]  #第0行為檔案圖片名稱
            f.write(file_path + '/' + file_name)  #兩個框座標中間的空白
            col = 1
            data = np.array(data_row[col:col + 4], dtype=int)
            f.write(' ')
            data2txt(f, data, '0')
            f.write('\n')  #換行後繼續寫入下一張圖片

            # 更新進度條
            pbar.update(1)
            pbar.set_description('mango2txt')
            pbar.set_postfix(**{
                'File_name': file_name,
            })
        f.close()

if __name__ == "__main__":
    defect2txt()
    # mango2txt()