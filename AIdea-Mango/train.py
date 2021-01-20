#-------------------------------------#
#       对数据集进行训练
#-------------------------------------#
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from comet_ml import Experiment
import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from utils.dataloader_bg import DataLoaderX
from utils.dataloader import yolo_dataset_collate, YoloDataset
from nets.yolo_training import YOLOLoss, Generator
from nets.yolo4 import YoloBody
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')


#---------------------------------------------------#
#   获得类和先验框
#---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen,
                  genval, Epoch, cuda, experiment):
    total_loss = 0
    val_loss = 0
    start_time = time.time()
    print('running epoch: {} / {}'.format((epoch + 1), Epoch))
    net.train()
    with tqdm(total=epoch_size, desc='train', postfix=dict,
              mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = torch.from_numpy(images).to(device,
                                                         non_blocking=True)
                    targets = [torch.from_numpy(ann) for ann in targets]
                else:
                    images = torch.from_numpy(images)
                    targets = [torch.from_numpy(ann) for ann in targets]
            optimizer.zero_grad()
            # with torch.cuda.amp.autocast():
            outputs = net(images)
            losses = []
            for i in range(3):
                # with torch.cuda.amp.autocast():
                loss_item = yolo_losses[i](outputs[i], targets)
                losses.append(loss_item[0])
            loss = sum(losses)

            loss.backward()
            optimizer.step()

            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            

            total_loss += loss
            final_train_loss = total_loss.item() / (iteration + 1)
            waste_time = time.time() - start_time

            lr = get_lr(optimizer)
            
            experiment.log_metric('loss', final_train_loss,epoch=epoch)
            experiment.log_metric('lr', lr, epoch=epoch)

            pbar.set_postfix(**{
                'total_loss': final_train_loss,
                'lr': lr,
                'step/s': waste_time
            })
            pbar.update(1)

            start_time = time.time()
    net.eval()
    with tqdm(total=epoch_size_val, desc='val', postfix=dict,
              mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val = batch[0], batch[1]

            with torch.no_grad():
                if cuda:
                    images_val = torch.from_numpy(images_val).to(
                        device, non_blocking=True)
                    targets_val = [
                        torch.from_numpy(ann) for ann in targets_val
                    ]
                else:
                    images_val = torch.from_numpy(images_val)
                    targets_val = [
                        torch.from_numpy(ann) for ann in targets_val
                    ]
                outputs = net(images_val)
                losses = []
                for i in range(3):
                    loss_item = yolo_losses[i](outputs[i], targets_val)
                    losses.append(loss_item[0])
                loss = sum(losses)
                val_loss += loss
                final_val_loss = val_loss.item() / (iteration + 1)
            pbar.set_postfix(**{'total_loss': final_val_loss})
            experiment.log_metric('val_loss', final_val_loss, epoch=epoch)
            pbar.update(1)
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' %
          (total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))

    print('Saving state, iter:', str(epoch + 1), '\n')
    torch.save(
        model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' %
        ((epoch + 1), total_loss / (epoch_size + 1), val_loss /
         (epoch_size_val + 1)))

    return final_train_loss, final_val_loss, lr


#----------------------------------------------------#
#   检测精度mAP和pr曲线计算参考视频
#   https://www.bilibili.com/video/BV1zE411u7Vw
#----------------------------------------------------#
if __name__ == "__main__":
    #-------------------------------#
    #   输入的shape大小
    #   显存比较小可以使用416x416
    #   显存比较大可以使用608x608
    #-------------------------------#
    input_shape = (416, 416)
    #-------------------------------#
    #   tricks的使用设置
    #-------------------------------#
    Cosine_lr = True
    mosaic = True
    # 用于设定是否使用cuda
    Cuda = True
    smoooth_label = 0
    #-------------------------------#
    #   Dataloder的使用
    #-------------------------------#
    Use_Data_Loader = True

    annotation_train_path = r'D:\Dataset\Mango\txt\train_defect.txt'
    annotation_val_path = r'D:\Dataset\Mango\txt\valid_defect.txt'
    #-------------------------------#
    #   获得先验框和类
    #-------------------------------#
    anchors_path = r'.\model_data\yolo_anchors_defect.txt'
    classes_path = r'.\model_data\defect_classes.txt'
    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)
    num_classes = len(class_names)

    # 创建模型
    model = YoloBody(len(anchors[0]), num_classes)
    #-------------------------------------------#
    #   权值文件的下载请看README
    #-------------------------------------------#
    model_path = r"./pretrained_model/yolo4_voc_weights.pth"
    # 加快模型训练的效率
    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if np.shape(model_dict[k]) == np.shape(v)
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')

    net = model.train()

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.to(device, non_blocking=True)

    # 建立loss函数
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(np.reshape(anchors,[-1,2]),num_classes, \
                                (input_shape[1], input_shape[0]), smoooth_label, Cuda))

    # 訓練樣本
    with open(annotation_train_path) as f:
        lines_train = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines_train)
    np.random.seed(None)
    num_train = len(lines_train)

    # 驗證樣本
    with open(annotation_val_path) as f:
        lines_val = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines_val)
    np.random.seed(None)
    num_val = len(lines_val)

    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if False:
        lr = 1e-3
        Batch_size = 50
        Init_Epoch = 0
        Freeze_Epoch = 20

        optimizer = optim.Adam(net.parameters(), lr, weight_decay=5e-4)
        scaler = torch.cuda.amp.GradScaler()
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                T_max=5,
                                                                eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                     step_size=1,
                                                     gamma=0.95)

        if Use_Data_Loader:
            train_dataset = YoloDataset(lines_train,
                                        (input_shape[0], input_shape[1]),
                                        mosaic=mosaic)
            val_dataset = YoloDataset(lines_val,
                                      (input_shape[0], input_shape[1]),
                                      mosaic=False)
            gen = DataLoaderX(train_dataset,
                              shuffle=True,
                              batch_size=Batch_size,
                              num_workers=0,
                              pin_memory=True,
                              drop_last=True,
                              collate_fn=yolo_dataset_collate)
            gen_val = DataLoaderX(val_dataset,
                                  shuffle=True,
                                  batch_size=Batch_size,
                                  num_workers=0,
                                  pin_memory=True,
                                  drop_last=True,
                                  collate_fn=yolo_dataset_collate)
        else:
            gen = Generator(
                Batch_size, lines[:num_train],
                (input_shape[0], input_shape[1])).generate(mosaic=mosaic)
            gen_val = Generator(
                Batch_size, lines[num_train:],
                (input_shape[0], input_shape[1])).generate(mosaic=False)

        epoch_size = max(1, num_train // Batch_size)
        epoch_size_val = num_val // Batch_size
        #------------------------------------#
        #   冻结一定部分训练
        #------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = False
        train_losses = []
        val_losses = []
        lrs = []

        for epoch in range(Init_Epoch, Freeze_Epoch):
            train_loss, val_loss, lr = fit_one_epoch(net, yolo_losses, epoch,
                                                     epoch_size,
                                                     epoch_size_val, gen,
                                                     gen_val, Freeze_Epoch,
                                                     Cuda)
            lr_scheduler.step()
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            lrs.append(lr)

        # 繪製圖
        plt.figure()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        # plt.xticks(np.arange(0, 10, step=1))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.legend(loc='best')
        plt.savefig('./images/loss.jpg')
        # plt.show()

        plt.figure()
        plt.xlabel('Epochs')
        plt.ylabel('Learning rate')
        # plt.xticks(np.arange(0, 10, step=1))
        plt.plot(lrs, label='Learning Rate')
        plt.legend(loc='best')
        plt.savefig('./images/lr.jpg')
        # plt.show()

    if True:
        lr = 1e-4
        Batch_size = 21
        Freeze_Epoch = 0
        Unfreeze_Epoch = 30

        optimizer = optim.AdamW(net.parameters(), lr, weight_decay=5e-4)
        if Cosine_lr:
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                T_max=5,
                                                                eta_min=1e-5)
        else:
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                     step_size=1,
                                                     gamma=0.95)

        if Use_Data_Loader:
            train_dataset = YoloDataset(lines_train,
                                        (input_shape[0], input_shape[1]),
                                        mosaic=mosaic)
            val_dataset = YoloDataset(lines_val,
                                      (input_shape[0], input_shape[1]),
                                      mosaic=False)
            gen = DataLoaderX(train_dataset,
                              shuffle=True,
                              batch_size=Batch_size,
                              num_workers=0,
                              pin_memory=True,
                              drop_last=True,
                              collate_fn=yolo_dataset_collate)
            gen_val = DataLoaderX(val_dataset,
                                  shuffle=True,
                                  batch_size=Batch_size,
                                  num_workers=0,
                                  pin_memory=True,
                                  drop_last=True,
                                  collate_fn=yolo_dataset_collate)
        else:
            gen = Generator(
                Batch_size, lines[:num_train],
                (input_shape[0], input_shape[1])).generate(mosaic=mosaic)
            gen_val = Generator(
                Batch_size, lines[num_train:],
                (input_shape[0], input_shape[1])).generate(mosaic=False)

        epoch_size = max(1, num_train // Batch_size)
        epoch_size_val = num_val // Batch_size
        #------------------------------------#
        #   解冻后训练
        #------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = True

        train_losses = []
        val_losses = []
        lrs = []
        experiment=Experiment(api_key='Key',
                    project_name="mango-defect",
                    workspace="jerryjack121",
                    disabled=False)
        for epoch in range(Freeze_Epoch, Unfreeze_Epoch):
            train_loss, val_loss, lr = fit_one_epoch(net, yolo_losses, epoch,
                                                     epoch_size,
                                                     epoch_size_val, gen,
                                                     gen_val, Unfreeze_Epoch,
                                                     Cuda, experiment)
            lr_scheduler.step()
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            lrs.append(lr)

        # 繪製圖
        plt.figure()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        # plt.xticks(np.arange(0, 10, step=1))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.legend(loc='best')
        plt.savefig('./images/loss.jpg')
        # plt.show()

        plt.figure()
        plt.xlabel('Epochs')
        plt.ylabel('Learning rate')
        # plt.xticks(np.arange(0, 10, step=1))
        plt.plot(lrs, label='Learning Rate')
        plt.legend(loc='best')
        plt.savefig('./images/lr.jpg')
        # plt.show()