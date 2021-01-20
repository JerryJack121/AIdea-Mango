from yolo_mango import YOLO
from PIL import Image
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# 測試資料集
fold_test = r'D:\dataset\Mango\Test'
yolo = YOLO()
D1 = []  #乳汁吸附
D2 = []  #機械傷害
D3 = []  #炭疽病
D4 = []  #著色不佳
D5 = []  #黑斑病

# 讀取測試圖片順序(比賽測試結果要求順序一致)
df = pd.read_csv(r'D:\dataset\Mango\Test_UploadSheet.csv',
                 encoding="utf8")
imglist = np.array(df['image_id'])
for img in tqdm(imglist):
    image = Image.open(os.path.join(fold_test, img))
    r_image, class_list, _ = yolo.detect_image(image)
    # r_image.show()
    # print(class_list)
    D1.append(1) if 'D1' in class_list else D1.append(0)
    D2.append(1) if 'D2' in class_list else D2.append(0)
    D3.append(1) if 'D3' in class_list else D3.append(0)
    D4.append(1) if 'D4' in class_list else D4.append(0)
    D5.append(1) if 'D5' in class_list else D5.append(0)

result_df = pd.DataFrame({
    'image_id': imglist,
    'D1': D1,
    'D2': D2,
    'D3': D3,
    'D4': D4,
    'D5': D5
})
# 輸出結果csv
result_df.to_csv(r'./results/result.csv', index=None)
