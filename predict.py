#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from yolo_mango import YOLO
from PIL import Image

yolo = YOLO()

# while True:
# img = input('Input image filename:')
img = r'D:\dataset\Mango\Train\29196.jpg'
try:
    image = Image.open(img)
except:
    print('Open Error! Try again!')
    # continue
else:
    r_image,_,_ = yolo.detect_image(image)
    r_image.show()
