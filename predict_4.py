"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/1/25 21:02
"""
import os
import cv2
import time
import torch
import cvzone
import argparse
import numpy as np
from torchstat import stat
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead

#查看预训练模型的参数量
def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print({'Total': total_num, 'Trainable': trainable_num})


#图像类别
classes=['__background__','person']


def predictImage(img_path,modelSSDLite):
    """
    :param img_path:
    :return:
    """
    # imgTo=Image.open(img)
    imgTo = cv2.imread(img_path)
    imgTo = cv2.resize(imgTo, (320, 320)) / 255
    # 这里需要变换通道(H,W,C)=>(C,H,W)
    # 方式一：
    newImg = np.transpose(imgTo, (2, 0, 1))
    # 转换为tensor类型，这里如果使用torch.tensor(newImg)转换图像类型的话，
    # 后面在输入网络时就会出错：RuntimeError: expected scalar type Double but found Float
    newImg = torch.Tensor(newImg)
    # 扩充维度
    newImg = torch.unsqueeze(input=newImg, dim=0)

    detection = modelSSDLite(newImg)
    print(detection)
    return detection[0]['boxes'], detection[0]['labels'], detection[0]['scores']


# 根据模型返回的结果，将其绘制到图像中
def drawRectangle(boxes, labels, scores, img_path):
    """
    :param boxes: 对应目标的坐标
    :param labels: 对应目标的标签
    :param scores: 对应目标的类别分数
    :return:
    """
    imgRe = cv2.imread(img_path)
    for k in range(len(labels)):
        # 左上角坐标(xleft,yleft)和右下角坐标(xright,yright)
        xleft = int(boxes[k][0])
        yleft = int(boxes[k][1])
        xright = int(boxes[k][2])
        yright = int(boxes[k][3])

        class_id = labels[k].item()

        confidence = scores[k].item()
        # 这里只输出检测是人并且概率值最大的
        if class_id == 1:
            text = classes[class_id] + ': ' + str('{:.4f}'.format(confidence))
            cv2.rectangle(imgRe, (xleft, yleft), (xright, yright), (255, 0, 255), 2)
            cvzone.putTextRect(img=imgRe, text=text, pos=(xleft + 9, yleft - 12),
                               scale=1, thickness=1, colorR=(0, 255, 0))
            break
    cv2.imshow('img', imgRe)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def timeDetect(modelSSDLite):
    """
    :return:
    """
    # 计算开始时间
    start_time = time.time()
    # 计算帧率
    countFPS = 0
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (320, 320))
        newImg = frame / 255

        # 这里需要变换通道(H,W,C)=>(C,H,W)
        # 方式一：
        newImg = np.transpose(newImg, (2, 0, 1))
        # 转换为tensor类型，这里如果使用torch.tensor(newImg)转换图像类型的话，
        # 后面在输入网络时就会出错：RuntimeError: expected scalar type Double but found Float
        newImg = torch.Tensor(newImg)
        newImg = torch.unsqueeze(input=newImg, dim=0)

        # 由模型中BN层属性决定，训练完train样本后，生成的模型model要用来测试样本。
        detection = modelSSDLite(newImg)
        boxes = detection[0]['boxes']
        labels = detection[0]['labels']
        scores = detection[0]['scores']

        for k in range(len(labels)):
            # 左上角坐标(xleft,yleft)和右下角坐标(xright,yright)
            xleft = int(boxes[k][0])
            yleft = int(boxes[k][1])
            xright = int(boxes[k][2])
            yright = int(boxes[k][3])

            class_id = labels[k].item()

            confidence = scores[k].item()
            # 这里只输出检测是人并且概率值最大的
            if class_id == 1:
                text = classes[class_id] + ': ' + str('{:.4f}'.format(confidence))
                cv2.rectangle(frame, (xleft, yleft), (xright, yright), (255, 0, 255), 2)
                cvzone.putTextRect(img=frame, text=text, pos=(xleft + 9, yleft - 12),
                                   scale=1, thickness=1, colorR=(0, 255, 0))
                break

        # 计算结束时间
        end_time = time.time()
        countFPS += 1
        FPS = round(countFPS / (end_time - start_time), 0)
        cv2.putText(img=frame, text='FPS: ' + str(FPS), org=(10, 50),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0, color=(0, 255, 0), thickness=2)
        cv2.imshow('img', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

def infence(modelPath,img_path,video):
    # #必须加这一句：由于我们训练的模型是在GPU，但是我们测试的时候是在CPU上，所以需要设置一下设备
    # model=torch.load(modelPath,map_location=lambda storage, loc: storage)
    modelSSDLite=torch.load(modelPath,map_location=torch.device('cpu'))

    # #加载在COCO数据集上训练的预训练模型
    # modelSSDLite=ssdlite320_mobilenet_v3_large(pretrained=False,progress=True)
    # # replace the classifier with a new one, that has
    # # 将分类器替换为具有用户定义的 num_classes的新分类器
    # # 获取分类器的输入参数的数量
    # # c_in_features=[modelSSDLite.head.classification_head.module_list[i].in_channels for i in range(len(modelSSDLite.head.classification_head.module_list))]
    # c_in_features=[]
    # norm_Layers=[]
    # for i in range(len(modelSSDLite.head.classification_head.module_list)):
    #     in_channels_1=modelSSDLite.head.classification_head.module_list[i][0][0].in_channels
    #     normLayer=modelSSDLite.head.classification_head.module_list[i][0][1]
    #     c_in_features.append(in_channels_1)
    #     norm_Layers.append(normLayer)
    #
    # num_anchors=modelSSDLite.anchor_generator.num_anchors_per_location()
    # # # 用新的头部替换预先训练好的头部
    # modelSSDLite.head.classification_head=SSDLiteClassificationHead(in_channels=c_in_features,num_anchors=num_anchors,
    #                                                                     num_classes=len(classes),norm_layer=torch.nn.BatchNorm2d)

    modelSSDLite.eval()
    print(modelSSDLite)

    if video==1:
        boxes,labels,scores=predictImage(img_path=img_path,modelSSDLite=modelSSDLite)
        drawRectangle(boxes=boxes, labels=labels, scores=scores, img_path=img_path)
    else:
        timeDetect(modelSSDLite=modelSSDLite)

if __name__ == '__main__':
    parser=argparse.ArgumentParser(add_help=True)
    parser.add_argument('--weight',default='../models/myModelBestssd320_1_1.pth',type=str)
    parser.add_argument('--img_path',default='../data/ssdLite320_mobilenet_v3_large/test_img/2206.png',type=str)
    parser.add_argument('--video',default=1,type=int)

    args=parser.parse_args()
    infence(modelPath=args.weight,img_path=args.img_path,video=args.video)
    pass