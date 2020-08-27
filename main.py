# efficientdet bug
# faster rcnn bug
# phone camera bug line 83
import sys
import cv2 as cv
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QFileDialog, QMainWindow
from gui import Ui_MainWindow
import time
import colorsys
from PIL import Image, ImageFont, ImageDraw
import winsound

from yolov3.yolo import YOLO
from yolov4.yolo import YOLO4

import mask.mrcnn.model as modellib
from mask.mdetect import InferenceConfig

from deeplab.nets.deeplab import Deeplabv3
from deeplab.predict import find_bbox, letterbox_image
from deeplab.classification import BModel

from faster.frcnn import FRCNN

from det.efficientdet import EfficientDet

# Load Models
img = Image.open('fire_666.jpg')
# deeplabv3
NCLASSES = 3
HEIGHT = 448
WIDTH = 448
model_deeplab = Deeplabv3(classes=NCLASSES,input_shape=(HEIGHT,WIDTH,3))
model_deeplab.load_weights("deeplab/deeplab.h5")
model_lenet = BModel()
model_lenet.load('deeplab/lenet.h5')
im = img.resize((448, 448))
im, nw, nh = letterbox_image(im, [448, 448])
im = np.array(im)
im = im/255
im = im.reshape(-1, 448, 448, 3)
pr = model_deeplab.predict(im)[0]

# Faster RCNN
f = FRCNN()
im, text = f.detect_image(img)

# EfficientDet
e = EfficientDet()
im, text = e.detect_image(img)

# YOLO v3
y = YOLO()
im, text = y.detect_image(img)

# YOLO v4
y4 = YOLO4()
im, text = y4.detect_image(img)

# Mask_RCNN
config = InferenceConfig()
model_mask = modellib.MaskRCNN(mode="inference", model_dir="mask/logs", config=config)
model_mask.load_weights("mask/weights.h5", by_name=True)
im = np.array(img)
results = model_mask.detect([im], verbose=1)


class gui(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.camera = cv.VideoCapture(0)
        self.is_camera_opened = False  # 摄像头有没有打开标记

    def Camera(self):
        if self.comboBox.currentText() == 'Local Camera':
            camera = cv.VideoCapture(0)
        elif self.comboBox.currentText() == 'Phone Camera':
            camera = cv.VideoCapture('rtsp://admin:admin@10.23.111.66:8080/h264_ulaw.sdp')
        while('Camera' in self.comboBox.currentText()):
            ref, frame = camera.read()
            frame = cv.resize(frame, (448, 448), interpolation=cv.INTER_CUBIC)
            rows, cols, channels = frame.shape
            bytesPerLine = channels * cols
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            QImg = QImage(frame.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
            self.input.setPixmap(QPixmap.fromImage(QImg).scaled(
                self.input.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            cv.waitKey(30) & 0xff
        camera.release()

    def Read(self):
        filename,  _ = QFileDialog.getOpenFileName(self, 'Choose Image Profile')
        self.file = filename
        if filename:
            captured = cv.imread(str(filename))
            # OpenCV图像以BGR通道存储，显示时需要从BGR转到RGB
            captured = cv.cvtColor(captured, cv.COLOR_BGR2RGB)
            captured = cv.resize(captured, (448, 448), interpolation=cv.INTER_CUBIC)
            rows, cols, channels = captured.shape
            bytesPerLine = channels * cols
            QImg = QImage(captured.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
            self.input.setPixmap(QPixmap.fromImage(QImg).scaled(
                self.input.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def Video(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Choose Video Profile')
        self.video = filename
        video = cv.VideoCapture(filename)
        ref, frame = video.read()
        while(ref and self.comboBox.currentText() == 'Video'):
            frame = cv.resize(frame, (448, 448), interpolation=cv.INTER_CUBIC)
            rows, cols, channels = frame.shape
            bytesPerLine = channels * cols
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            QImg = QImage(frame.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
            self.input.setPixmap(QPixmap.fromImage(QImg).scaled(
                self.input.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            cv.waitKey(30) & 0xff
            ref, frame = video.read()
        video.release()
    def Text_yolo(self):
        self.lineEdit.setText('YOLO_V3 Detection:')
    def Text_yolo4(self):
        self.lineEdit.setText('YOLO_V4 Detection:')
    def Text_frcnn(self):
        self.lineEdit.setText('Faster RCNN Detection:')
    def Text_deeplab(self):
        self.lineEdit.setText('DeepLab_V3+ Detection:')
    def Text_mask(self):
        self.lineEdit.setText('Mask RCNN Detection:')
    def Text_det(self):
        self.lineEdit.setText('EfficientDet Detection:')

    def FRCNN(self):
        duration = 50  # millisecond
        freq = 1000  # Hz
        if self.comboBox.currentText() == 'Image':
            im = Image.open(self.file)
            time_start = time.time()
            im,text = f.detect_image(im)
            t = time.time()-time_start
            im = cv.cvtColor(np.array(im), cv.COLOR_RGB2BGR)
            im = cv.resize(im, (448, 448), interpolation=cv.INTER_CUBIC)
            im = cv.cvtColor(im,cv.COLOR_BGR2RGB)
            rows, cols, channels = im.shape
            bytesPerLine = channels * cols
            QImg = QImage(im.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
            self.output.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.output.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            text = 'Time used:'+str(round(t,4))+'s \n'+text
            self.textEdit.setPlainText(text)
            if 'Fire' in text or 'Smoke' in text:
                winsound.Beep(freq, duration)

        else:
            if self.comboBox.currentText() == 'Video':
                camera = cv.VideoCapture(self.video)
            elif self.comboBox.currentText() == 'Local Camera':
                camera = cv.VideoCapture(0)
            else:
                camera = cv.VideoCapture('rtsp://admin:admin@100.77.206.71:8080/h264_ulaw.sdp')
            ref, frame = camera.read()
            while(ref and self.comboBox.currentText() != 'Image'):
                rows, cols, channels = frame.shape
                bytesPerLine = channels * cols
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                QImg = QImage(frame.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
                self.input.setPixmap(QPixmap.fromImage(QImg).scaled(
                    self.input.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                im = Image.fromarray(np.uint8(frame))
                im, text = f.detect_image(im)
                im = cv.cvtColor(np.array(im), cv.COLOR_RGB2BGR)
                im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
                rows, cols, channels = im.shape
                bytesPerLine = channels * cols
                QImg = QImage(im.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
                self.output.setPixmap(QPixmap.fromImage(QImg).scaled(
                    self.output.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.textEdit.setPlainText(text)
                cv.waitKey(100) & 0xff
                ref, frame = camera.read()
            camera.release()

    def DeeplabV3(self):
        if self.comboBox.currentText() == 'Image':
            WIDTH,HEIGHT = 448,448
            image = Image.open(self.file)
            img = image.resize((448,448))
            im = img
            time_start = time.time()
            img,nw,nh = letterbox_image(img,[448,448])
            img = np.array(img)
            img = img/255
            img = img.reshape(-1,448,448,3)
            pr = model_deeplab.predict(img)[0]
            pr = pr.reshape((448, 448,3)).argmax(axis=-1)
            pr = Image.fromarray(np.uint8(pr))
            pr = pr.resize((448,448))
            pr = pr.crop(((WIDTH-nw)//2, (HEIGHT-nh)//2,(WIDTH-nw)//2+nw,(HEIGHT-nh)//2+nh))
            pr = np.array(pr)
            mask1 = np.zeros((nh, nw))  # fire
            mask2 = np.zeros((nh, nw))  #smoke

            if 2 in pr:
                mask1 = (( pr[:,: ] == 1 )*255).astype('uint8')
                mask2 = (( pr[:,: ] == 2 )*255).astype('uint8')
            else:
                if 'fire' in self.file:
                    mask1 = (( pr[:,: ] == 1 )*255).astype('uint8')
                elif 'smoke' in self.file:
                    mask2 = (( pr[:,: ] == 1 )*255).astype('uint8')
                else:
                    prediction = model_lenet.predict(image)
                    if prediction == 0:
                        mask1 = (( pr[:,: ] == 1 )*255).astype('uint8')
                    else:
                        mask2 = (( pr[:,: ] == 1 )*255).astype('uint8')
            class_names = ['BG', 'fire','smoke']
            hsv_tuples = [(x / len(class_names), 1., 1.)
                          for x in range(len(class_names))]
            colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            colors = list(
                map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),colors))

            np.random.seed(10101)  # Fixed seed for consistent colors across runs.
            np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
            np.random.seed(None)  # Reset seed to default.
            font = ImageFont.truetype(font='arial.ttf',size=np.floor(3e-2 * im.size[1] + 0.5).astype('int32'))
            thickness = (im.size[0] + im.size[1]) // 300

            bboxs1 = find_bbox(mask1)
            bboxs1 = bboxs1[bboxs1[:,-1]>2000]
            bboxs2 = find_bbox(mask2)
            bboxs2 = bboxs2[bboxs2[:,-1]>5000]
            t = time.time() - time_start

            num_box = len(bboxs1)+len(bboxs2)
            if  num_box > 1:
                text = 'Found {} boxes for {} \n'.format(num_box, 'image')
            else:
                text = 'Found {} box for {} \n'.format(num_box, 'image')
            if len(bboxs1) >= 1:
                label = 'fire'

                for i in range(len(bboxs1)):
                    left, top, width,height = bboxs1[i,0:4]
                    right = left + width
                    bottom = top + height
                    top = max(0, np.floor(top + 0.5).astype('int32'))
                    left = max(0, np.floor(left + 0.5).astype('int32'))
                    bottom = min(im.size[1], np.floor(bottom + 0.5).astype('int32'))
                    right = min(im.size[0], np.floor(right + 0.5).astype('int32'))
                    draw = ImageDraw.Draw(im)
                    label_size = draw.textsize(label, font)

                    if top - label_size[1] >= 0:
                        text_origin = np.array([left, top - label_size[1]])
                    else:
                        text_origin = np.array([left, top + 1])

                    for j in range(thickness):
                        draw.rectangle(
                            [left + j, top + j, right - j, bottom - j],
                            outline=colors[1])
                        draw.rectangle(
                            [tuple(text_origin), tuple(text_origin + label_size)],
                            fill=colors[1])
                    draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                    del draw

            if len(bboxs2) >= 1:
                label = 'smoke'

                for i in range(len(bboxs2)):
                    left, top, width,height = bboxs2[i,0:4]
                    right = left + width
                    bottom = top + height
                    top = max(0, np.floor(top + 0.5).astype('int32'))
                    left = max(0, np.floor(left + 0.5).astype('int32'))
                    bottom = min(im.size[1], np.floor(bottom + 0.5).astype('int32'))
                    right = min(im.size[0], np.floor(right + 0.5).astype('int32'))
                    draw = ImageDraw.Draw(im)
                    label_size = draw.textsize(label, font)

                    if top - label_size[1] >= 0:
                        text_origin = np.array([left, top - label_size[1]])
                    else:
                        text_origin = np.array([left, top + 1])

                    for j in range(thickness):
                        draw.rectangle(
                            [left + j, top + j, right - j, bottom - j],
                            outline=colors[2])
                        draw.rectangle(
                            [tuple(text_origin), tuple(text_origin + label_size)],
                            fill=colors[2])
                    draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                    del draw

            masked_image = cv.cvtColor(np.array(im),cv.COLOR_RGB2BGR)
            masked_image = cv.resize(masked_image, (448, 448), interpolation=cv.INTER_CUBIC)
            masked_image = cv.cvtColor(masked_image,cv.COLOR_BGR2RGB)
            rows, cols, channels = masked_image.shape
            bytesPerLine = channels * cols
            QImg = QImage(masked_image.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
            self.output.setPixmap(QPixmap.fromImage(QImg).scaled(
                self.output.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            if len(bboxs1) > 0 and len(bboxs2) > 0:
                a = 'Fire & Smoke\n'
            elif len(bboxs1) > 0:
                a = 'Fire\n'
            elif len(bboxs2) > 0:
                a = 'Smoke\n'
            else:
                a = 'Normal\n'
            text = 'Classification:'+ a + text
            text = 'Time used:'+str(round(t,4))+'s \n'+text
            self.textEdit.setPlainText(text)
        else:
            if self.comboBox.currentText() == 'Video':
                camera = cv.VideoCapture(self.video)
            elif self.comboBox.currentText() == 'Local Camera':
                camera = cv.VideoCapture(0)
            else:
                camera = cv.VideoCapture('rtsp://admin:admin@100.77.206.71:8080/h264_ulaw.sdp')
            ref, frame = camera.read()
            fps = 0
            while(ref and self.comboBox.currentText() != 'Image'):
                frame = cv.resize(frame, (448, 448), interpolation=cv.INTER_CUBIC)
                t1 = time.time()
                rows, cols, channels = frame.shape
                bytesPerLine = channels * cols
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                QImg = QImage(frame.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
                self.input.setPixmap(QPixmap.fromImage(QImg).scaled(
                    self.input.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                image = Image.fromarray(np.uint8(frame))
                WIDTH, HEIGHT = 448, 448
                img = image.resize((448, 448))
                im = img

                img, nw, nh = letterbox_image(img, [448, 448])
                img = np.array(img)
                img = img / 255
                img = img.reshape(-1, 448, 448, 3)
                pr = model_deeplab.predict(img)[0]
                pr = pr.reshape((448, 448, 3)).argmax(axis=-1)

                pr = Image.fromarray(np.uint8(pr))
                pr = pr.resize((448, 448))
                pr = pr.crop(((WIDTH - nw) // 2, (HEIGHT - nh) // 2, (WIDTH - nw) // 2 + nw, (HEIGHT - nh) // 2 + nh))
                pr = np.array(pr)
                mask1 = np.zeros((nh, nw))  # fire
                mask2 = np.zeros((nh, nw))  # smoke

                if 2 in pr:
                    mask1 = ((pr[:, :] == 1) * 255).astype('uint8')
                    mask2 = ((pr[:, :] == 2) * 255).astype('uint8')
                else:
                    prediction = model_lenet.predict(image)
                    if prediction == 0:
                        mask1 = ((pr[:, :] == 1) * 255).astype('uint8')
                    else:
                        mask2 = ((pr[:, :] == 1) * 255).astype('uint8')
                class_names = ['BG', 'fire', 'smoke']
                hsv_tuples = [(x / len(class_names), 1., 1.)
                              for x in range(len(class_names))]
                colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
                colors = list(
                    map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

                np.random.seed(10101)  # Fixed seed for consistent colors across runs.
                np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
                np.random.seed(None)  # Reset seed to default.
                font = ImageFont.truetype(font='arial.ttf', size=np.floor(3e-2 * im.size[1] + 0.5).astype('int32'))
                thickness = (im.size[0] + im.size[1]) // 300

                bboxs1 = find_bbox(mask1)
                bboxs1 = bboxs1[bboxs1[:, -1] > 2000]
                bboxs2 = find_bbox(mask2)
                bboxs2 = bboxs2[bboxs2[:, -1] > 10000]

                num_box = len(bboxs1) + len(bboxs2)
                if num_box > 1:
                    text = 'Found {} boxes for {} \n'.format(num_box, 'image')
                else:
                    text = 'Found {} box for {} \n'.format(num_box, 'image')
                if len(bboxs1) >= 1:
                    label = 'fire'

                    for i in range(len(bboxs1)):
                        left, top, width, height = bboxs1[i, 0:4]
                        right = left + width
                        bottom = top + height
                        top = max(0, np.floor(top + 0.5).astype('int32'))
                        left = max(0, np.floor(left + 0.5).astype('int32'))
                        bottom = min(im.size[1], np.floor(bottom + 0.5).astype('int32'))
                        right = min(im.size[0], np.floor(right + 0.5).astype('int32'))
                        draw = ImageDraw.Draw(im)
                        label_size = draw.textsize(label, font)

                        if top - label_size[1] >= 0:
                            text_origin = np.array([left, top - label_size[1]])
                        else:
                            text_origin = np.array([left, top + 1])

                        for j in range(thickness):
                            draw.rectangle(
                                [left + j, top + j, right - j, bottom - j],
                                outline=colors[1])
                            draw.rectangle(
                                [tuple(text_origin), tuple(text_origin + label_size)],
                                fill=colors[1])
                        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                        del draw

                if len(bboxs2) >= 1:
                    label = 'smoke'

                    for i in range(len(bboxs2)):
                        left, top, width, height = bboxs2[i, 0:4]
                        right = left + width
                        bottom = top + height
                        top = max(0, np.floor(top + 0.5).astype('int32'))
                        left = max(0, np.floor(left + 0.5).astype('int32'))
                        bottom = min(im.size[1], np.floor(bottom + 0.5).astype('int32'))
                        right = min(im.size[0], np.floor(right + 0.5).astype('int32'))
                        draw = ImageDraw.Draw(im)
                        label_size = draw.textsize(label, font)

                        if top - label_size[1] >= 0:
                            text_origin = np.array([left, top - label_size[1]])
                        else:
                            text_origin = np.array([left, top + 1])

                        for j in range(thickness):
                            draw.rectangle(
                                [left + j, top + j, right - j, bottom - j],
                                outline=colors[2])
                            draw.rectangle(
                                [tuple(text_origin), tuple(text_origin + label_size)],
                                fill=colors[2])
                        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                        del draw

                fps = (fps + (1. / (time.time() - t1))) / 2
                masked_image = cv.cvtColor(np.array(im), cv.COLOR_RGB2BGR)
                if len(bboxs1)>0 and len(bboxs2)>0:
                    a = 'Fire & Smoke\n'
                    masked_image = cv.putText(masked_image, "Fire & Smoke, fps= %.2f" % (fps), (0, 40), cv.FONT_HERSHEY_SIMPLEX, 1,(0, 0, 255), 2)
                elif len(bboxs1)>0:
                    a = 'Fire\n'
                    masked_image = cv.putText(masked_image, "Fire, fps= %.2f" % (fps), (0, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif len(bboxs2)>0:
                    a = 'Smoke\n'
                    masked_image = cv.putText(masked_image, "Smoke, fps= %.2f" % (fps), (0, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0,128, 255), 2)
                else:
                    a = 'Normal\n'
                    masked_image = cv.putText(masked_image, "Normal, fps= %.2f" % (fps), (0, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                masked_image = cv.cvtColor(masked_image, cv.COLOR_BGR2RGB)
                masked_image = cv.resize(masked_image, (448, 448), interpolation=cv.INTER_CUBIC)
                rows, cols, channels = masked_image.shape
                bytesPerLine = channels * cols
                QImg = QImage(masked_image.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
                self.output.setPixmap(QPixmap.fromImage(QImg).scaled(
                    self.output.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

                text = 'Classification:' + a + text
                self.lineEdit.setText('DeepLab_V3+ Detection:')
                self.textEdit.setPlainText(text)
                cv.waitKey(30) & 0xff
                ref, frame = camera.read()
            camera.release()

    def YOLOV3(self):
        if self.comboBox.currentText() == 'Image':
            im = Image.open(self.file)
            time_start = time.time()
            im,text = y.detect_image(im)
            t = time.time() - time_start
            im = cv.cvtColor(np.array(im), cv.COLOR_RGB2BGR)
            im = cv.resize(im, (448, 448), interpolation=cv.INTER_CUBIC)
            im = cv.cvtColor(im,cv.COLOR_BGR2RGB)
            rows, cols, channels = im.shape
            bytesPerLine = channels * cols
            QImg = QImage(im.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
            self.output.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.output.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            text = 'Time used:'+str(round(t,4))+'s \n'+text

            self.textEdit.setPlainText(text)
        else:
            if self.comboBox.currentText() == 'Video':
                camera = cv.VideoCapture(self.video)
            elif self.comboBox.currentText() == 'Local Camera':
                camera = cv.VideoCapture(0)
            else:
                camera = cv.VideoCapture('rtsp://admin:admin@100.77.206.71:8080/h264_ulaw.sdp')
            ref, frame = camera.read()
            fps = 0
            while(ref and self.comboBox.currentText() != 'Image'):
                frame = cv.resize(frame, (448, 448), interpolation=cv.INTER_CUBIC)
                t1 = time.time()
                rows, cols, channels = frame.shape
                bytesPerLine = channels * cols
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                QImg = QImage(frame.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
                self.input.setPixmap(QPixmap.fromImage(QImg).scaled(
                    self.input.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                im = Image.fromarray(np.uint8(frame))
                im, text = y.detect_image(im)
                im = cv.cvtColor(np.array(im), cv.COLOR_RGB2BGR)
                im = cv.resize(im, (448, 448), interpolation=cv.INTER_CUBIC)
                fps = (fps + (1. / (time.time() - t1))) / 2
                if 'Fire & Smoke' in text:
                    im = cv.putText(im, "Fire & Smoke, fps= %.2f" % (fps), (0, 40),
                                              cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif 'Fire' in text:
                    im = cv.putText(im, "Fire, fps= %.2f" % (fps), (0, 40), cv.FONT_HERSHEY_SIMPLEX,
                                              1, (0, 0, 255), 2)
                elif 'Smoke' in text:
                    im = cv.putText(im, "Smoke, fps= %.2f" % (fps), (0, 40),
                                              cv.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 255), 2)
                else:
                    im = cv.putText(im, "Normal, fps= %.2f" % (fps), (0, 40),
                                              cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                im = cv.cvtColor(im,cv.COLOR_BGR2RGB)
                rows, cols, channels = im.shape
                bytesPerLine = channels * cols
                QImg = QImage(im.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
                self.output.setPixmap(QPixmap.fromImage(QImg).scaled(
                self.output.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.textEdit.setPlainText(text)
                self.lineEdit.setText('YOLO_V3 Detection:')
                cv.waitKey(30) & 0xff
                ref, frame = camera.read()
            camera.release()

    def YOLOV4(self):
        if self.comboBox.currentText() == 'Image':
            im = Image.open(self.file)
            time_start = time.time()
            im,text = y4.detect_image(im)
            t = time.time() - time_start
            im = cv.cvtColor(np.array(im), cv.COLOR_RGB2BGR)
            im = cv.resize(im, (448, 448), interpolation=cv.INTER_CUBIC)
            im = cv.cvtColor(im,cv.COLOR_BGR2RGB)
            rows, cols, channels = im.shape
            bytesPerLine = channels * cols
            QImg = QImage(im.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
            self.output.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.output.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            text = 'Time used:'+str(round(t,4))+'s \n'+text

            self.textEdit.setPlainText(text)
        else:
            if self.comboBox.currentText() == 'Video':
                camera = cv.VideoCapture(self.video)
            elif self.comboBox.currentText() == 'Local Camera':
                camera = cv.VideoCapture(0)
            else:
                camera = cv.VideoCapture('rtsp://admin:admin@100.77.206.71:8080/h264_ulaw.sdp')
            ref, frame = camera.read()
            fps = 0
            while(ref and self.comboBox.currentText() != 'Image'):
                frame = cv.resize(frame, (448, 448), interpolation=cv.INTER_CUBIC)
                t1 = time.time()
                rows, cols, channels = frame.shape
                bytesPerLine = channels * cols
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                QImg = QImage(frame.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
                self.input.setPixmap(QPixmap.fromImage(QImg).scaled(
                    self.input.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                im = Image.fromarray(np.uint8(frame))
                im, text = y4.detect_image(im)
                im = cv.cvtColor(np.array(im), cv.COLOR_RGB2BGR)
                im = cv.resize(im, (448, 448), interpolation=cv.INTER_CUBIC)
                fps = (fps + (1. / (time.time() - t1))) / 2
                if 'Fire & Smoke' in text:
                    im = cv.putText(im, "Fire & Smoke, fps= %.2f" % (fps), (0, 40),
                                              cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif 'Fire' in text:
                    im = cv.putText(im, "Fire, fps= %.2f" % (fps), (0, 40), cv.FONT_HERSHEY_SIMPLEX,
                                              1, (0, 0, 255), 2)
                elif 'Smoke' in text:
                    im = cv.putText(im, "Smoke, fps= %.2f" % (fps), (0, 40),
                                              cv.FONT_HERSHEY_SIMPLEX, 1, (0,128, 255), 2)
                else:
                    im = cv.putText(im, "Normal, fps= %.2f" % (fps), (0, 40),
                                              cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                im = cv.cvtColor(im,cv.COLOR_BGR2RGB)
                rows, cols, channels = im.shape
                bytesPerLine = channels * cols
                QImg = QImage(im.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
                self.output.setPixmap(QPixmap.fromImage(QImg).scaled(
                self.output.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.textEdit.setPlainText(text)
                self.lineEdit.setText('YOLO_V4 Detection:')
                cv.waitKey(30) & 0xff
                ref, frame = camera.read()
            camera.release()

    def EfficientDet(self):
        if self.comboBox.currentText() == 'Image':
            im = Image.open(self.file)
            time_start = time.time()
            im,text = e.detect_image(im)
            t = time.time() - time_start
            im = cv.cvtColor(np.array(im), cv.COLOR_RGB2BGR)
            im = cv.resize(im, (448, 448), interpolation=cv.INTER_CUBIC)
            im = cv.cvtColor(im,cv.COLOR_BGR2RGB)
            rows, cols, channels = im.shape
            bytesPerLine = channels * cols
            QImg = QImage(im.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
            self.output.setPixmap(QPixmap.fromImage(QImg).scaled(
            self.output.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            text = 'Time used:'+str(round(t,4))+'s \n'+text
            self.textEdit.setPlainText(text)
        else:
            if self.comboBox.currentText() == 'Video':
                camera = cv.VideoCapture(self.video)
            elif self.comboBox.currentText() == 'Local Camera':
                camera = cv.VideoCapture(0)
            else:
                camera = cv.VideoCapture('rtsp://admin:admin@100.77.206.71:8080/h264_ulaw.sdp')
            ref, frame = camera.read()
            fps = 0
            duration = 50  # millisecond
            freq = 1000  # Hz

            while(ref and self.comboBox.currentText() != 'Image'):
                frame = cv.resize(frame, (448, 448), interpolation=cv.INTER_CUBIC)
                t1 = time.time()
                rows, cols, channels = frame.shape
                bytesPerLine = channels * cols
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                QImg = QImage(frame.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
                self.input.setPixmap(QPixmap.fromImage(QImg).scaled(
                    self.input.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                im = Image.fromarray(np.uint8(frame))
                im, text = e.detect_image(im)
                im = cv.cvtColor(np.array(im), cv.COLOR_RGB2BGR)
                im = cv.resize(im, (448, 448), interpolation=cv.INTER_CUBIC)
                fps = (fps + (1. / (time.time() - t1))) / 2
                if 'Fire & Smoke' in text:
                    im = cv.putText(im, "Fire & Smoke, fps= %.2f" % (fps), (0, 40),
                                              cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    winsound.Beep(freq, duration)
                elif 'Fire' in text:
                    im = cv.putText(im, "Fire, fps= %.2f" % (fps), (0, 40), cv.FONT_HERSHEY_SIMPLEX,
                                              1, (0, 0, 255), 2)
                    winsound.Beep(freq, duration)
                elif 'Smoke' in text:
                    im = cv.putText(im, "Smoke, fps= %.2f" % (fps), (0, 40),
                                              cv.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 255), 2)
                    winsound.Beep(freq, duration)
                else:
                    im = cv.putText(im, "Normal, fps= %.2f" % (fps), (0, 40),
                                              cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                im = cv.cvtColor(im,cv.COLOR_BGR2RGB)
                rows, cols, channels = im.shape
                bytesPerLine = channels * cols
                QImg = QImage(im.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
                self.output.setPixmap(QPixmap.fromImage(QImg).scaled(
                    self.output.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.textEdit.setPlainText(text)
                self.lineEdit.setText('EfficientDet Detection:')
                cv.waitKey(30) & 0xff
                ref, frame = camera.read()
            camera.release()

    def Mask_RCNN(self):
        class_names = ['BG', 'fire', 'smoke']
        if self.comboBox.currentText() == 'Image':
            image = Image.open(self.file)
            im = np.array(image)
            time_start = time.time()
            results = model_mask.detect([im], verbose=1)
            t = time.time() - time_start
            r = results[0]
            hsv_tuples = [(x / len(class_names), 1., 1.)
                          for x in range(len(class_names))]
            colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            colors = list(
                map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),colors))

            np.random.seed(10101)  # Fixed seed for consistent colors across runs.
            np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
            np.random.seed(None)  # Reset seed to default.

            font = ImageFont.truetype(font='arial.ttf',size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
            thickness = (image.size[0] + image.size[1]) // 300

            for i in range(len(r['rois'])):
                class_id = r['class_ids'][i]
                predicted_class = class_names[class_id]
                box = r['rois'][i]
                score = r['scores'][i]

                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)

                top, left, bottom, right = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                for j in range(thickness):
                    draw.rectangle(
                        [left + j, top + j, right - j, bottom - j],
                        outline=colors[class_id])
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=colors[class_id])
                draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                del draw

            if len(r['rois']) > 1:
                text = 'Found {} boxes for {} \n'.format(len(r['rois']), 'image')
            else:
                text = 'Found {} box for {} \n'.format(len(r['rois']), 'image')
            for i in range(len(r['rois'])):
                class_id = r['class_ids'][i]
                score = r['scores'][i] if r['scores'] is not None else None
                label = class_names[class_id]
                text += 'Box'+str(i+1)+':'+label +'   score='+str(round(score,3))+'\n'

            if 'fire' in text and 'smoke' in text:
                a = 'Fire & Smoke\n'
            elif 'fire' in text:
                a = 'Fire\n'
            elif 'smoke' in text:
                a = 'Smoke\n'
            else:
                a = 'Normal\n'
            text = 'Classification:' + a + text

            masked_image = cv.cvtColor(np.array(image),cv.COLOR_RGB2BGR)
            masked_image = cv.resize(masked_image, (448, 448), interpolation=cv.INTER_CUBIC)
            masked_image = cv.cvtColor(masked_image,cv.COLOR_BGR2RGB)
            rows, cols, channels = masked_image.shape
            bytesPerLine = channels * cols
            QImg = QImage(masked_image.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
            self.output.setPixmap(QPixmap.fromImage(QImg).scaled(
                self.output.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            text = 'Time used:'+str(round(t,4))+'s \n'+text
            self.textEdit.setPlainText(text)
        else:
            if self.comboBox.currentText() == 'Video':
                camera = cv.VideoCapture(self.video)
            elif self.comboBox.currentText() == 'Local Camera':
                camera = cv.VideoCapture(0)
            else:
                camera = cv.VideoCapture('rtsp://admin:admin@100.77.206.71:8080/h264_ulaw.sdp')
            ref, frame = camera.read()
            fps = 0
            while(ref and self.comboBox.currentText() != 'Image'):
                frame = cv.resize(frame, (448, 448), interpolation=cv.INTER_CUBIC)
                t1 = time.time()
                rows, cols, channels = frame.shape
                bytesPerLine = channels * cols
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                QImg = QImage(frame.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
                self.input.setPixmap(QPixmap.fromImage(QImg).scaled(
                    self.input.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                image = Image.fromarray(np.uint8(frame))
                im = np.array(image)
                results = model_mask.detect([im], verbose=1)
                r = results[0]
                hsv_tuples = [(x / len(class_names), 1., 1.)
                              for x in range(len(class_names))]
                colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
                colors = list(
                    map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

                np.random.seed(10101)  # Fixed seed for consistent colors across runs.
                np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
                np.random.seed(None)  # Reset seed to default.

                font = ImageFont.truetype(font='arial.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
                thickness = (image.size[0] + image.size[1]) // 300

                for i in range(len(r['rois'])):
                    class_id = r['class_ids'][i]
                    predicted_class = class_names[class_id]
                    box = r['rois'][i]
                    score = r['scores'][i]

                    label = '{} {:.2f}'.format(predicted_class, score)
                    draw = ImageDraw.Draw(image)
                    label_size = draw.textsize(label, font)

                    top, left, bottom, right = box
                    top = max(0, np.floor(top + 0.5).astype('int32'))
                    left = max(0, np.floor(left + 0.5).astype('int32'))
                    bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                    right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                    print(label, (left, top), (right, bottom))

                    if top - label_size[1] >= 0:
                        text_origin = np.array([left, top - label_size[1]])
                    else:
                        text_origin = np.array([left, top + 1])

                    for j in range(thickness):
                        draw.rectangle(
                            [left + j, top + j, right - j, bottom - j],
                            outline=colors[class_id])
                    draw.rectangle(
                        [tuple(text_origin), tuple(text_origin + label_size)],
                        fill=colors[class_id])
                    draw.text(text_origin, label, fill=(0, 0, 0), font=font)
                    del draw

                if len(r['rois']) > 1:
                    text = 'Found {} boxes for {} \n'.format(len(r['rois']), 'image')
                else:
                    text = 'Found {} box for {} \n'.format(len(r['rois']), 'image')
                for i in range(len(r['rois'])):
                    class_id = r['class_ids'][i]
                    score = r['scores'][i] if r['scores'] is not None else None
                    label = class_names[class_id]
                    text += 'Box' + str(i + 1) + ':' + label + '   score=' + str(round(score, 3)) + '\n'

                if 'fire' in text and 'smoke' in text:
                    a = 'Fire & Smoke\n'
                elif 'fire' in text:
                    a = 'Fire\n'
                elif 'smoke' in text:
                    a = 'Smoke\n'
                else:
                    a = 'Normal\n'
                text = 'Classification:' + a + text

                masked_image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
                masked_image = cv.resize(masked_image, (448, 448), interpolation=cv.INTER_CUBIC)

                fps = (fps + (1. / (time.time() - t1))) / 2
                if 'Fire & Smoke' in text:
                    masked_image = cv.putText(masked_image, "Fire & Smoke, fps= %.2f" % (fps), (0, 40),
                                              cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif 'Fire' in text:
                    masked_image = cv.putText(masked_image, "Fire, fps= %.2f" % (fps), (0, 40), cv.FONT_HERSHEY_SIMPLEX,
                                              1, (0, 0, 255), 2)
                elif 'Smoke' in text:
                    masked_image = cv.putText(masked_image, "Smoke, fps= %.2f" % (fps), (0, 40),
                                              cv.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 255), 2)
                else:
                    masked_image = cv.putText(masked_image, "Normal, fps= %.2f" % (fps), (0, 40),
                                              cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                masked_image = cv.cvtColor(masked_image, cv.COLOR_BGR2RGB)
                rows, cols, channels = masked_image.shape
                bytesPerLine = channels * cols
                QImg = QImage(masked_image.data, cols, rows, bytesPerLine, QImage.Format_RGB888)
                self.output.setPixmap(QPixmap.fromImage(QImg).scaled(
                    self.output.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.textEdit.setPlainText(text)
                self.lineEdit.setText('Mask RCNN Detection:')

                cv.waitKey(30) & 0xff
                ref, frame = camera.read()
            camera.release()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = gui()
    window.show()
    sys.exit(app.exec_())