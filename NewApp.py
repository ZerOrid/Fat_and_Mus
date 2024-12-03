import shutil
from pyexpat import features

from PyQt5.QtWidgets import (QMainWindow, QMessageBox, QFileDialog, QApplication)
from PyQt5.QtGui import QPixmap
import numpy as np
import os
import nibabel as nib
from  UI import Ui_MainWindow
import SimpleITK as itk
import sys
from qimage2ndarray import array2qimage
import torch
from Processing import *
from PyQt5 import QtCore, QtGui, QtWidgets
import math
from vtk.vtkCommonDataModel import vtkPiecewiseFunction
from vtk.vtkIOImage import vtkMetaImageReader, vtkNIFTIImageReader
from vtk.vtkRenderingCore import (
    vtkColorTransferFunction,
    vtkRenderer,
    vtkVolume,
    vtkVolumeProperty
)
from vtk.vtkRenderingVolume import vtkGPUVolumeRayCastMapper
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from radiomics import featureextractor
import radiomics
import SimpleITK as sitk

gpu_id = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NewWindow(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(NewWindow, self).__init__()
        self.setupUi(self)
        self.img = None
        self.showmask = None
        self.prinimg = None
        self.printlivermask = None
        self.mask = None
        self.space = None
        self.fmap_block = list()
        self.grad_block = list()
        self.numprint = None
        self.fatmus_mask = None
        self.face_flage = 0

        self.view1.setMouseTracking(True)
        self.view2.setMouseTracking(True)
        self.view3.setMouseTracking(True)
        self.view1.installEventFilter(self)
        self.view2.installEventFilter(self)
        self.view3.installEventFilter(self)

        self.leng_img = -100
        self.width_img = -100
        self.high_img = -100
        self.right_press_flag = False
        self.left_press_flag = False
        self.face_w = 272
        self.face_h = 272
        self.file_name = None
        self.file_path = None

        self.spine_mask_path = os.path.join('data', 'ori', 'spine')
        self.fatmus_mask_path = os.path.join('data', 'ori', 'fatmus')
        self.img_path = os.path.join('data', 'ori', 'img')
        self.crop_spine_mask_path = os.path.join('data', 'crop', 'spine')
        self.crop_fatmus_mask_path = os.path.join('data', 'crop', 'fatmus')
        self.crop_img_path = os.path.join('data', 'crop', 'img')
        os.makedirs(self.spine_mask_path, exist_ok=True)
        os.makedirs(self.fatmus_mask_path, exist_ok=True)
        os.makedirs(self.img_path, exist_ok=True)
        os.makedirs(self.crop_spine_mask_path, exist_ok=True)
        os.makedirs(self.crop_fatmus_mask_path, exist_ok=True)
        os.makedirs(self.crop_img_path, exist_ok=True)
        os.makedirs(os.path.join('data', 'nnunet_in'), exist_ok=True)
        os.makedirs(os.path.join('data', 'nnunet_out'), exist_ok=True)

        self.spine_mask_file = ''
        self.fatmus_mask_file = ''
        self.img_file = ''
        self.crop_spine_mask_file = ''
        self.crop_fatmus_mask_file = ''
        self.crop_img_file = ''

        self.volumes = {}
        self.volume_path = ''
        self.volume_old = None
        self.afpg2 = None
        self.afpg1 = None
        self.am_mean = None
        self.am_25var = None
        self.vf_25var = None
        self.scale_ratio = 1

        self.scene1.mouseDoubleClickEvent = lambda event: self.pointselect(event, 1)
        self.scene2.mouseDoubleClickEvent = lambda event: self.pointselect(event, 2)
        self.scene3.mouseDoubleClickEvent = lambda event: self.pointselect(event, 3)

        self.pen = QtGui.QPen(QtCore.Qt.green)
        self.pen2 = QtGui.QPen(QtCore.Qt.red, 4)
        self.pen3 = QtGui.QPen(QtCore.Qt.red)

        self.x_line1 = QtWidgets.QGraphicsLineItem()
        self.x_line2 = QtWidgets.QGraphicsLineItem()
        self.x_line1.setPen(self.pen)
        self.x_line2.setPen(self.pen)
        self.y_line1 = QtWidgets.QGraphicsLineItem()
        self.y_line2 = QtWidgets.QGraphicsLineItem()
        self.y_line1.setPen(self.pen)
        self.y_line2.setPen(self.pen)
        self.z_line1 = QtWidgets.QGraphicsLineItem()
        self.z_line2 = QtWidgets.QGraphicsLineItem()
        self.z_line1.setPen(self.pen)
        self.z_line2.setPen(self.pen)

        self.x_point1 = QtWidgets.QGraphicsEllipseItem()
        self.x_point2 = QtWidgets.QGraphicsEllipseItem()
        self.x_point1.setPen(self.pen2)
        self.x_point2.setPen(self.pen2)
        self.y_point1 = QtWidgets.QGraphicsEllipseItem()
        self.y_point2 = QtWidgets.QGraphicsEllipseItem()
        self.y_point1.setPen(self.pen2)
        self.y_point2.setPen(self.pen2)
        self.z_point1 = QtWidgets.QGraphicsEllipseItem()
        self.z_point2 = QtWidgets.QGraphicsEllipseItem()
        self.z_point1.setPen(self.pen2)
        self.z_point2.setPen(self.pen2)

        self.x_point_flag = 1
        self.y_point_flag = 1
        self.z_point_flag = 1
        self.x_point2line = QtWidgets.QGraphicsLineItem()
        self.x_point2line.setPen(self.pen3)
        self.y_point2line = QtWidgets.QGraphicsLineItem()
        self.y_point2line.setPen(self.pen3)
        self.z_point2line = QtWidgets.QGraphicsLineItem()
        self.z_point2line.setPen(self.pen3)

        self.x_x = None
        self.x_y = None
        self.y_x = None
        self.y_y = None
        self.z_x = None
        self.z_y = None
        self.pixmapItem1 = None
        self.pixmapItem2 = None
        self.pixmapItem3 = None

        self.desired_features_MUS = {
            'wavelet-LHL_glszm_GrayLevelVariance': 'mus_lhl_glszm_gray_level_variance',
            'wavelet-LLL_glcm_ClusterTendency': 'mus_lll_glcm_cluster_tendency',
            'wavelet-LHL_firstorder_Entropy': 'mus_lhl_firstorder_entropy',
            'wavelet-LHL_glcm_JointEntropy': 'mus_lhl_glcm_joint_entropy',
            'wavelet-HHL_firstorder_10Percentile': 'mus_hhl_firstorder_10percentile',
        }

        self.desired_features_FAT = {
            'wavelet-LHH_glrlm_GrayLevelVariance': 'fat_lhh_glrlm_gray_level_variance',
            'wavelet-HLH_firstorder_Mean': 'fat_hlh_firstorder_mean',
        }

        self.features = []

    def pointselect(self, event, view_type):
        if self.prinimg is None:
            return

        if event.button() == Qt.LeftButton:
        # 处理左键点击，更新切片显示和绘制截面线
            self.handle_left_click(event, view_type)
        elif event.button() == Qt.RightButton:
        # 处理右键点击，测量距离
            self.handle_right_click(event, view_type)

    def handle_left_click(self, event, view_type):
        # 获取鼠标位置
        pos_x = event.scenePos().x()
        pos_y = event.scenePos().y()

        if view_type == 1:
            # 视图1的坐标计算
            self.leng_img = self.leng_img  # 保持不变
            self.width_img = int(round((pos_y / self.face_h) * self.width_max, 0))
            self.high_img = int(round((pos_x / self.face_w) * self.high_max, 0))
            self.x_x = pos_x
            self.x_y = pos_y
            self.y_x = pos_x
            self.y_y = int(round(self.face_h * (self.leng_img / self.leng_max), 0))
            self.z_x = int(round(self.face_w * (pos_y / self.face_h), 0))
            self.z_y = self.y_y
        elif view_type == 2:
            # 视图2的坐标计算
            self.leng_img = int(round((pos_y / self.face_h) * self.leng_max, 0))
            self.width_img = self.width_img  # 保持不变
            self.high_img = int(round((pos_x / self.face_w) * self.high_max, 0))
            self.x_x = pos_x
            self.x_y = int(round(self.face_h * (self.width_img / self.width_max), 0))
            self.y_x = pos_x
            self.y_y = pos_y
            self.z_x = int(round(self.face_w * (self.width_img / self.width_max), 0))
            self.z_y = pos_y
        elif view_type == 3:
            # 视图3的坐标计算
            self.leng_img = int(round((pos_y / self.face_h) * self.leng_max, 0))
            self.width_img = int(round((pos_x / self.face_w) * self.width_max, 0))
            self.high_img = self.high_img  # 保持不变
            self.x_x = int(round(self.face_w * (self.high_img / self.high_max), 0))
            self.x_y = int(round(self.face_h * (pos_x / self.face_w), 0))
            self.y_x = self.x_x
            self.y_y = pos_y
            self.z_x = pos_x
            self.z_y = pos_y

        # 更新显示和绘制截面线
        self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
        self.draw_line(self.x_x, self.x_y, self.y_x, self.y_y, self.z_x, self.z_y)

    def handle_right_click(self, event, view_type):
        pos_x = event.scenePos().x()
        pos_y = event.scenePos().y()

        if view_type == 1:
            point_flag = 'x_point_flag'
            point1 = self.x_point1
            point2 = self.x_point2
            point2line = self.x_point2line
            scene = self.scene1
            distance_attr = 'x_distance'
            space_x = self.space[0]
            space_y = self.space[1]
            max_x = self.high_max
            max_y = self.width_max
            face_w = self.face_w
            face_h = self.face_h
            draw_point_method = self.draw_point_x
        elif view_type == 2:
            point_flag = 'y_point_flag'
            point1 = self.y_point1
            point2 = self.y_point2
            point2line = self.y_point2line
            scene = self.scene2
            distance_attr = 'y_distance'
            space_x = self.high_max
            space_y = self.leng_max
            max_x = self.high_max
            max_y = self.leng_max
            face_w = self.face_w
            face_h = self.face_h
            draw_point_method = self.draw_point_y
        elif view_type == 3:
            point_flag = 'z_point_flag'
            point1 = self.z_point1
            point2 = self.z_point2
            point2line = self.z_point2line
            scene = self.scene3
            distance_attr = 'z_distance'
            space_x = self.width_max
            space_y = self.leng_max
            max_x = self.width_max
            max_y = self.leng_max
            face_w = self.face_w
            face_h = self.face_h
            draw_point_method = self.draw_point_z

        flag = getattr(self, point_flag)

        if flag == 1:
            # 绘制第一个点
            draw_point_method(point1, pos_x, pos_y)
            setattr(self, f'point_{view_type}_1_x', pos_x)
            setattr(self, f'point_{view_type}_1_y', pos_y)
            # 移除之前的线条和点
            self.remove_previous_items(view_type)
            setattr(self, point_flag, 2)
        elif flag == 2:
            # 绘制第二个点并计算距离
            draw_point_method(point2, pos_x, pos_y)
            setattr(self, f'point_{view_type}_2_x', pos_x)
            setattr(self, f'point_{view_type}_2_y', pos_y)
            point1_x = getattr(self, f'point_{view_type}_1_x')
            point1_y = getattr(self, f'point_{view_type}_1_y')
            self.drawline(scene, point2line, point1_x, point1_y, pos_x, pos_y)
            # 计算距离
            distance_x = abs(point1_x - pos_x) / face_w * max_x * space_x
            distance_y = abs(point1_y - pos_y) / face_h * max_y * space_y
            distance = math.sqrt(distance_x ** 2 + distance_y ** 2)
            setattr(self, distance_attr, distance)
            self.distance.setText(f"{distance:>.8f}")
            setattr(self, point_flag, 1)

    def remove_previous_items(self, view_type):
        # 移除之前的线条和点
        self.scene1.removeItem(self.x_point2line)
        self.scene2.removeItem(self.y_point2line)
        self.scene3.removeItem(self.z_point2line)
        self.scene1.removeItem(self.x_point1)
        self.scene1.removeItem(self.x_point2)
        self.scene2.removeItem(self.y_point1)
        self.scene2.removeItem(self.y_point2)
        self.scene3.removeItem(self.z_point1)
        self.scene3.removeItem(self.z_point2)
    
    def remove_previous_items(self, view_type):
        # 移除之前的线条和点
        self.scene1.removeItem(self.x_point2line)
        self.scene2.removeItem(self.y_point2line)
        self.scene3.removeItem(self.z_point2line)
        self.scene1.removeItem(self.x_point1)
        self.scene1.removeItem(self.x_point2)
        self.scene2.removeItem(self.y_point1)
        self.scene2.removeItem(self.y_point2)
        self.scene3.removeItem(self.z_point1)
        self.scene3.removeItem(self.z_point2)


    def draw_point_x(self, item, x, y):
        self.scene1.removeItem(item)
        self.scene1.removeItem(item)
        item.setRect(x - 2, y - 2, 4, 4)
        self.scene1.addItem(item)
        self.scene1.addItem(item)

    def drawline(self, scene, item, x1, y1, x2, y2):
        item.setLine(QtCore.QLineF(QtCore.QPointF(x1, y1),
                                   QtCore.QPointF(x2, y2)))
        scene.addItem(item)

    def draw_point_y(self, item, x, y):
        self.scene2.removeItem(item)
        self.scene2.removeItem(item)
        item.setRect(x - 2, y - 2, 4, 4)
        self.scene2.addItem(item)
        self.scene2.addItem(item)

    def draw_point_z(self, item, x, y):
        self.scene3.removeItem(item)
        self.scene3.removeItem(item)
        item.setRect(x - 2, y - 2, 4, 4)
        self.scene3.addItem(item)
        self.scene3.addItem(item)

    def draw_line(self, x_x, x_y, y_x, y_y, z_x, z_y):
        self.scene1.removeItem(self.x_line1)
        self.scene1.removeItem(self.x_line2)
        self.scene2.removeItem(self.y_line1)
        self.scene2.removeItem(self.y_line2)
        self.scene3.removeItem(self.z_line1)
        self.scene3.removeItem(self.z_line2)
        self.x_line1.setLine(QtCore.QLineF(QtCore.QPointF(int(x_x), 0),
                                           QtCore.QPointF(int(x_x), self.scene1.height())))
        self.x_line2.setLine(QtCore.QLineF(QtCore.QPointF(0, int(x_y)),
                                           QtCore.QPointF(self.scene1.width(), int(x_y))))
        self.y_line1.setLine(QtCore.QLineF(QtCore.QPointF(int(y_x), 0),
                                           QtCore.QPointF(int(y_x), self.scene2.height())))
        self.y_line2.setLine(QtCore.QLineF(QtCore.QPointF(0, int(y_y)),
                                           QtCore.QPointF(self.scene2.width(), int(y_y))))
        self.z_line1.setLine(QtCore.QLineF(QtCore.QPointF(int(z_x), 0),
                                           QtCore.QPointF(int(z_x), self.scene3.height())))
        self.z_line2.setLine(QtCore.QLineF(QtCore.QPointF(0, int(z_y)),
                                           QtCore.QPointF(self.scene3.width(), int(z_y))))
        self.scene1.addItem(self.x_line1)
        self.scene1.addItem(self.x_line2)
        self.scene2.addItem(self.y_line1)
        self.scene2.addItem(self.y_line2)
        self.scene3.addItem(self.z_line1)
        self.scene3.addItem(self.z_line2)

    def cleardistancef(self):
        self.scene1.removeItem(self.x_point2line)
        self.scene2.removeItem(self.y_point2line)
        self.scene3.removeItem(self.z_point2line)
        self.scene1.removeItem(self.x_point1)
        self.scene1.removeItem(self.x_point2)
        self.scene2.removeItem(self.y_point1)
        self.scene2.removeItem(self.y_point2)
        self.scene3.removeItem(self.z_point1)
        self.scene3.removeItem(self.z_point2)
        self.distance.clear()
    
    def clearallf(self):
        self.spine_mask = None
        self.fatmus_mask = None
        self.crop_spine_mask = None
        self.crop_fatmus_mask = None
        self.img = None
        self.crop_img = None
        self.printlivermask = None
        self.space = None
        self.numprint = None
        self.showmask = None
        self.prinimg = None
        self.mask = None
        self.fmap_block = list()
        self.grad_block = list()
        self.numprint = None
        self.fatmus_mask = None
        self.face_flage = 0

        self.plotresult.clear()
        self.INR_line.clear()
        self.RBC_line.clear()
        self.Sodium_line.clear()
        self.Albumin_line.clear()
        self.Creatinine_line.clear()
        self.Height_line.clear()
        self.VF_Median_line.clear()
        self.AM_Median_line.clear()
        self.AM_Normal_line.clear()
        self.BM_1_4CT_line.clear()
        self.VF_1_4CT_line.clear()
        self.BM_Median_line.clear()
        self.SF_Median_line.clear()
        self.am_mean = None
        self.am_am_first_quartile = None
        self.vf_am_first_quartile = None

    def __setDragEnabled(self, isEnabled: bool):
        """ 设置拖拽是否启动 """
        self.view1.setDragMode(self.view1.ScrollHandDrag if isEnabled else self.view1.NoDrag)
        self.view2.setDragMode(self.view2.ScrollHandDrag if isEnabled else self.view2.NoDrag)
        self.view3.setDragMode(self.view3.ScrollHandDrag if isEnabled else self.view3.NoDrag)
    
    def __isEnableDrag(self, pixmap):
        """ 根据图片的尺寸决定是否启动拖拽功能 """
        if self.prinimg is not None:
            v = pixmap.width() > self.face_w
            h = pixmap.height() > self.face_h
            return v or h
    def showpic_xyz(self, x, y, z, w_size, h_size):
        if self.prinimg is not None:
            if self.prinimg.ndim == 3:
                image_axi = array2qimage(np.expand_dims(self.prinimg[x, ...], axis=-1), normalize=True)
            elif self.prinimg.ndim == 4:
                image_axi = array2qimage(self.prinimg[x, ...])

            pixmap_axi = QPixmap.fromImage(image_axi).scaled(w_size, h_size)
            self.pixmapItem1 = QtWidgets.QGraphicsPixmapItem(pixmap_axi)
            self.scene1.addItem(self.pixmapItem1)
            self.view1.setSceneRect(QtCore.QRectF(pixmap_axi.rect()))
            self.view1.setScene(self.scene1)

            if self.prinimg.ndim == 3:
                image_cor = array2qimage(np.expand_dims(self.prinimg[:, y, ...], axis=-1), normalize=True)
            elif self.prinimg.ndim == 4:
                image_cor = array2qimage(self.prinimg[:, y, ...])
            pixmap_cor = QPixmap.fromImage(image_cor).scaled(w_size, h_size)
            self.pixmapItem2 = QtWidgets.QGraphicsPixmapItem(pixmap_cor)
            self.scene2.addItem(self.pixmapItem2)
            self.view2.setSceneRect(QtCore.QRectF(pixmap_cor.rect()))
            self.view2.setScene(self.scene2)

            if self.prinimg.ndim == 3:
                image_sag = array2qimage(np.expand_dims(self.prinimg[:, :, z, ...], axis=-1), normalize=True)
            elif self.prinimg.ndim == 4:
                image_sag = array2qimage(self.prinimg[:, :, z, ...])
            pixmap_sag = QPixmap.fromImage(image_sag).scaled(w_size, h_size)
            self.pixmapItem3 = QtWidgets.QGraphicsPixmapItem(pixmap_sag)
            self.scene3.addItem(self.pixmapItem3)
            self.view3.setSceneRect(QtCore.QRectF(pixmap_sag.rect()))
            self.view3.setScene(self.scene3)

            self.__setDragEnabled(self.__isEnableDrag(pixmap_axi))

    def showpic(self):
        fname = QFileDialog().getOpenFileName(self, caption='Load CT image',
                                              directory='data',
                                              filter="Image(*.nii *.nii.gz)")
        self.file_path = fname[0]
        self.parent_file = os.path.abspath(os.path.join(self.file_path, ".."))
        if self.prinimg is not None:
            self.clearallf()
        if len(fname[1]) != 0:
            self.statusbar.showMessage("Loading the image...")
            self.file_name = fname[0].split('/')[-1]
            self.img_file = os.path.join(self.img_path, self.file_name)
            self.fatmus_mask_file = os.path.join(self.fatmus_mask_path, self.file_name)
            self.spine_mask_file = os.path.join(self.spine_mask_path, self.file_name)
            if not os.path.exists(self.img_file):
                shutil.copy(self.file_path, self.img_file)
            print(self.file_name)
            img = itk.ReadImage(self.img_file)
            self.space = img.GetSpacing()
            # img = itk.GetArrayFromImage(img)
            self.img = itk.GetArrayFromImage(img)
            img = np.clip(self.img, -17.0, 201.0)
            img = np.flip(img, axis=0)
            self.prinimg = (img - 99.40078) / 39.392952
            self.ori_prinimg = self.prinimg
            self.leng_max, self.width_max, self.high_max = self.img.shape
            self.face_w = 272
            self.face_h = 272

            self.leng_img = int(self.leng_max / 2)
            self.width_img = int(self.width_max / 2)
            self.high_img = int(self.high_max / 2)
            self.showpic_xyz(int(self.leng_max / 2), int(self.width_max / 2), int(self.high_max / 2), self.face_w,
                             self.face_h)
            self.x_x = self.face_w // 2
            self.x_y = self.face_h // 2
            self.y_x = self.face_w // 2
            self.y_y = self.face_h // 2
            self.z_x = self.face_w // 2
            self.z_y = self.face_h // 2
            self.draw_line(self.x_x, self.x_y, self.y_x, self.y_y, self.z_x, self.z_y)

            # self.coord.setText(f"Path: {self.file_path}")

            reader = vtkNIFTIImageReader()
            reader.SetFileName(self.img_file)
            reader.Update()

            volumeMapper = vtkGPUVolumeRayCastMapper()
            volumeMapper.SetInputData(reader.GetOutput())

            volumeProperty = vtkVolumeProperty()
            volumeProperty.SetInterpolationTypeToLinear()  # 设置体绘制的属性设置，决定体绘制的渲染效果
            volumeProperty.ShadeOn()  # 打开或者关闭阴影
            volumeProperty.SetAmbient(0.4)
            volumeProperty.SetDiffuse(0.6)  # 漫反射
            volumeProperty.SetSpecular(0.2)  # 镜面反射
            # 设置不透明度
            compositeOpacity = vtkPiecewiseFunction()
            compositeOpacity.AddPoint(70, 0.00)
            compositeOpacity.AddPoint(90, 0.4)
            compositeOpacity.AddPoint(180, 0.6)
            volumeProperty.SetScalarOpacity(compositeOpacity)  # 设置不透明度
            # 设置梯度不透明属性
            volumeGradientOpacity = vtkPiecewiseFunction()
            volumeGradientOpacity.AddPoint(10, 0.0)
            volumeGradientOpacity.AddPoint(90, 0.5)
            volumeGradientOpacity.AddPoint(100, 1.0)

            # volumeProperty.SetGradientOpacity(volumeGradientOpacity)  # 设置梯度不透明度效果对比
            # 设置颜色属性
            color = vtkColorTransferFunction()
            color.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
            color.AddRGBPoint(64.0, 1.0, 0.52, 0.3)
            color.AddRGBPoint(109.0, 1.0, 1.0, 1.0)
            color.AddRGBPoint(220.0, 0.2, 0.2, 0.2)
            volumeProperty.SetColor(color)

            volume = vtkVolume()  # 和vtkActor作用一致
            volume.SetMapper(volumeMapper)
            volume.SetProperty(volumeProperty)

            if self.volume_old is not None:
                self.ren.RemoveViewProp(self.volume_old)
            self.ren.AddViewProp(volume)
            self.volume_old = volume
            camera = self.ren.GetActiveCamera()
            c = volume.GetCenter()

            camera.SetViewUp(0, 0, 1)
            camera.SetViewAngle(60)
            camera.SetPosition(c[0] - 300, c[1] - 600, c[2])
            camera.SetFocalPoint(c[0], c[1] - 200, c[2])
            camera.Azimuth(40.0)
            camera.Elevation(10.0)
            # self.render_window.Render()
            self.iren.Initialize()

            view_size = self.view1.size()
            view_w = view_size.width()
            view_h = view_size.height()
            self.vtkWidget.resize(view_w, view_h)
            self.statusbar.showMessage("Loaded the image.")
    def show_fatmus_mask(self):

        if self.img is None:
            self.statusbar.showMessage("Please load a image, first.")
        else:
            fname = QFileDialog.getOpenFileName(self, caption='Load fat and muscle mask', directory='data',
                                                filter="Image(*.nii *.nii.gz)")
            if len(fname[1]) != 0:
                self.showmask = read_nii(fname[0])
                if self.showmask.shape == self.img.shape:
                    self.mask_np = read_nii(fname[0])
                    self.img_np = read_nii(self.img_file)
                    self.new_img = self.mask_np * self.img_np
                    ori2new_nii(fname[0], self.new_img, os.path.join('./data', 'new_fatmus.nii.gz'))
                    if self.prinimg is not None:
                        self.printmask_fatmus(0.5)
                        # self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                        self.draw_line(self.x_x, self.x_y, self.y_x, self.y_y, self.z_x, self.z_y)
                        reader = vtkNIFTIImageReader()
                        # reader.SetFileName(os.path.join('./data', 'new_fatmus.nii.gz'))
                        reader.SetFileName(fname[0])
                        reader.Update()
                        volumeMapper = vtkGPUVolumeRayCastMapper()
                        volumeMapper.SetInputData(reader.GetOutput())

                        volumeProperty = vtkVolumeProperty()
                        volumeProperty.SetInterpolationTypeToLinear()  # 设置体绘制的属性设置，决定体绘制的渲染效果
                        volumeProperty.ShadeOn()  # 打开或者关闭阴影
                        volumeProperty.SetAmbient(0.4)
                        volumeProperty.SetDiffuse(0.6)  # 漫反射
                        volumeProperty.SetSpecular(0.2)  # 镜面反射
                        # 设置不透明度
                        compositeOpacity = vtkPiecewiseFunction()
                        compositeOpacity.AddPoint(0, 0)
                        compositeOpacity.AddPoint(1, 0.2)
                        compositeOpacity.AddPoint(2, 0.2)
                        compositeOpacity.AddPoint(3, 0.2)
                        compositeOpacity.AddPoint(4, 0.2)
                        volumeProperty.SetScalarOpacity(compositeOpacity)  # 设置不透明度

                        # 设置梯度不透明属性
                        volumeGradientOpacity = vtkPiecewiseFunction()
                        volumeGradientOpacity.AddPoint(0, 0.0)
                        volumeGradientOpacity.AddPoint(1, 0.5)
                        volumeGradientOpacity.AddPoint(5, 1.0)

                        # volumeProperty.SetGradientOpacity(volumeGradientOpacity)  # 设置梯度不透明度效果对比
                        # 设置颜色属性
                        color = vtkColorTransferFunction()
                        color.AddRGBPoint(0, 0, 0, 0)
                        color.AddRGBPoint(1, 0.8, 0.8, 0.8)
                        color.AddRGBPoint(2, 0.4, 0.4, 0.4)
                        color.AddRGBPoint(3, 0.8, 0.52, 0.3)
                        color.AddRGBPoint(4, 0.8, 0.8, 0.3)
                        volumeProperty.SetColor(color)

                        volume = vtkVolume()  # 和vtkActor作用一致
                        volume.SetMapper(volumeMapper)
                        volume.SetProperty(volumeProperty)
                        if self.volume_old is not None:
                            self.ren.RemoveViewProp(self.volume_old)
                        self.ren.AddViewProp(volume)
                        self.volume_old = volume
                        camera = self.ren.GetActiveCamera()
                        c = volume.GetCenter()
                        camera.SetViewUp(0, 0, 1)
                        camera.SetViewAngle(60)
                        camera.SetPosition(c[0] - 300, c[1] - 600, c[2])
                        camera.SetFocalPoint(c[0], c[1] - 200, c[2])
                        camera.Azimuth(30.0)
                        camera.Elevation(30.0)
                        self.iren.Initialize()
                        os.remove(os.path.join('./data', 'new_fatmus.nii.gz'))
                        self.statusbar.showMessage("The mask of fat and muscle has been displayed.")
                else:
                    # self.coord.setText('请加载与图对应的分割结果')
                    self.statusbar.showMessage("Please load a correspronding mask of fat and muscle.")
    def show_spine_mask(self):

        if self.img is None:
            self.statusbar.showMessage("Please load a image, first.")
        else:
            fname = QFileDialog.getOpenFileName(self, caption='Load spine mask', directory='data',
                                                filter="Image(*.nii *.nii.gz)")
            if len(fname[1]) != 0:
                spine_mask = itk.ReadImage(fname[0])
                self.showmask = itk.GetArrayFromImage(spine_mask)
                if self.showmask.shape == self.img.shape:
                    self.mask_np = read_nii(fname[0])
                    self.img_np = read_nii(self.img_file)
                    self.new_img = self.mask_np * self.img_np
                    ori2new_nii(fname[0], self.new_img, os.path.join('./data', 'new_spine.nii.gz'))

                    if self.prinimg is not None:
                        self.printmask_spine(0.5)
                        self.draw_line(self.x_x, self.x_y, self.y_x, self.y_y, self.z_x, self.z_y)
                        reader = vtkNIFTIImageReader()
                        reader.SetFileName(os.path.join('./data', 'new_spine.nii.gz'))
                        reader.Update()
                        volumeMapper = vtkGPUVolumeRayCastMapper()
                        volumeMapper.SetInputData(reader.GetOutput())

                        volumeProperty = vtkVolumeProperty()
                        volumeProperty.SetInterpolationTypeToLinear()  # 设置体绘制的属性设置，决定体绘制的渲染效果
                        volumeProperty.ShadeOn()  # 打开或者关闭阴影
                        volumeProperty.SetAmbient(0.4)
                        volumeProperty.SetDiffuse(0.6)  # 漫反射
                        volumeProperty.SetSpecular(0.2)  # 镜面反射
                        # 设置不透明度
                        compositeOpacity = vtkPiecewiseFunction()
                        compositeOpacity.AddPoint(0, 0.00)
                        compositeOpacity.AddPoint(19, 1)
                        volumeProperty.SetScalarOpacity(compositeOpacity)  # 设置不透明度

                        # 设置颜色属性
                        color = vtkColorTransferFunction()
                        color.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
                        color.AddRGBPoint(19.0, 1.0, 1.0, 1)
                        color.AddRGBPoint(20.0, 0.8, 0.8, 0.8)
                        color.AddRGBPoint(21.0, 0.6, 0.6, 0.6)
                        color.AddRGBPoint(22.0, 0.4, 0.4, 0.4)
                        volumeProperty.SetColor(color)

                        volume = vtkVolume()  # 和vtkActor作用一致
                        volume.SetMapper(volumeMapper)
                        volume.SetProperty(volumeProperty)
                        if self.volume_old is not None:
                            self.ren.RemoveViewProp(self.volume_old)
                        self.ren.AddViewProp(volume)
                        self.volume_old = volume
                        camera = self.ren.GetActiveCamera()
                        c = volume.GetCenter()
                        camera.SetViewUp(0, 0, 1)
                        camera.SetViewAngle(60)
                        camera.SetPosition(c[0] - 300, c[1] - 600, c[2])
                        camera.SetFocalPoint(c[0], c[1] - 200, c[2])
                        camera.Azimuth(30.0)
                        camera.Elevation(30.0)
                        self.iren.Initialize()
                        os.remove(os.path.join('./data', 'new_spine.nii.gz'))
                        self.statusbar.showMessage("The mask of spine has been displayed.")
                else:

                    self.statusbar.showMessage("Please load a correspronding mask of spine.")  

    def cal_fatmus_vol(self):
        excel_3d_data = []
        data_3d_name = [
    'SF-volume', 'VF-volume', 'AM-volume', 'BM-volume',  # 4
    'SF-HU_mean', 'SF-HU_median', 'SF-HU_25per', 'SF-HU_75per', 'SF-HU_gap', 'SF-HU_25perVar', 'SF-HU_75perVar',  # 11
    'VF-HU_mean', 'VF-HU_median', 'VF-HU_25per', 'VF-HU_75per', 'VF-HU_gap', 'VF-HU_25perVar', 'VF-HU_75perVar',  # 18
    'AM-HU_mean', 'AM-HU_median', 'AM-HU_25per', 'AM-HU_75per', 'AM-HU_gap', 'AM-HU_25perVar', 'AM-HU_75perVar',  # 25
    'BM-HU_mean', 'BM-HU_median', 'BM-HU_25per', 'BM-HU_75per', 'BM-HU_gap', 'BM-HU_25perVar', 'BM-HU_75perVar',  # 32
    'Normalized_AM_Volume']
        self.statusbar.showMessage("Calculating the 3D data of fat and muscle.")
        self.show_message_compute_volume()
        self.height = self.Height_line.text()

        if os.path.exists(self.crop_fatmus_mask_file) == False and self.crop_fatmus_mask is None:
            self.fatmus_seg()
            self.spine_seg()
        elif os.path.exists(self.crop_fatmus_mask_file):
            crop_fatmus_mask = itk.ReadImage(self.crop_fatmus_mask_file)
            self.crop_fatmus_mask = itk.GetArrayFromImage(crop_fatmus_mask)
        if os.path.exists(self.crop_img_file) == True:
            crop_img = itk.ReadImage(self.crop_img_file)
            self.crop_img = itk.GetArrayFromImage(crop_img)
        space = itk.ReadImage(self.crop_img_file).GetSpacing()
        #开始计算每种组织的体积
        for l in range(1, 5):
            label_mask = self.crop_fatmus_mask.copy()
            label_mask[label_mask != l] = 0
            label_mask[label_mask == l] = 1
            vol = np.sum(label_mask) * space[0] * space[1] * space[2] / 1000
            excel_3d_data.append(vol)
        excel_3d_data = np.around(excel_3d_data, decimals=4)
        excel_3d_data = excel_3d_data.tolist()
        image_flat = self.crop_img.flatten()
        mask_flat = self.crop_fatmus_mask.flatten()
        # 计算每种组织的统计指标
        for l in range(1, 5):
            mask_flat_one = np.zeros(mask_flat.shape)
            mask_flat_one[mask_flat == l] = 1
            mask_indices = np.nonzero(mask_flat_one)
            mask_voxel_values = image_flat[mask_indices]
            mean_value = np.mean(mask_voxel_values)  # 平均值
            median_value = np.percentile(mask_voxel_values,  50)  # 中位数
            percentile_25 = np.percentile(mask_voxel_values, 25)  # 25百分位数（第一个四分位数）
            percentile_75 = np.percentile(mask_voxel_values, 75)  # 75百分位数
            gap = percentile_75 - percentile_25  # 75和25百分位数的差值（间距）
            var_25 = percentile_25 / gap if gap != 0 else 0  # 25百分位数变异系数
            var_75 = percentile_75 / gap if gap != 0 else 0  # 75百分位数变异系数
             # 将计算结果添加到数据列表
            excel_3d_data.extend([mean_value, median_value, percentile_25, percentile_75, gap, var_25, var_75])

        excel_3d_data = np.around(excel_3d_data, decimals=4)

        print(excel_3d_data)
        self.vf_median = excel_3d_data[12]  # 中位数CT值 - 内脏脂肪组织
        self.am_median = excel_3d_data[19]  # 中位数CT值 - 腹部肌肉
        self.bm_median = excel_3d_data[26]  # 中位数CT值 - 背部肌肉
        self.sf_median = excel_3d_data[5]   # 中位数CT值 - 皮下脂肪组织
        self.bm_25th_percentile = excel_3d_data[27]  # 第一四分位CT值 - 背部肌肉
        self.vf_25th_percentile = excel_3d_data[13]  # 第一四分位CT值 - 内脏脂肪
        self.am_volume_normalized = excel_3d_data[2] # 归一化体积

        self.AM_Normal_line.setText(str(round(self.am_volume_normalized, 4)))   # 展示归一化腹部肌肉体积

        # 展示背部肌肉第一个四分位数CT值
        self.BM_1_4CT_line.setText(str(excel_3d_data[27]))   # 背部肌肉的第一个四分位数CT值

        # 展示内脏脂肪第一个四分位数CT值
        self.VF_1_4CT_line.setText(str(excel_3d_data[13]))   # 内脏脂肪的第一个四分位数CT值

        # 展示背部肌肉中位CT值
        self.BM_Median_line.setText(str(excel_3d_data[26]))   # 背部肌肉的中位数CT值

        # 展示皮下脂肪中位CT值
        self.SF_Median_line.setText(str(excel_3d_data[5]))   # 皮下脂肪的中位数CT值

        self.AM_Median_line.setText(str(excel_3d_data[19]))

        self.VF_Median_line.setText(str(excel_3d_data[12]))

        # 禁用不需要进一步编辑的文本框，以便用户不能更改这些计算得到的结果
        self.VF_Median_line.setEnabled(False)  # 内脏脂肪中位数CT值
        self.AM_Median_line.setEnabled(False)  # 腹部肌肉中位数CT值
        self.AM_Normal_line.setEnabled(False)  # 腹部肌肉归一化体积
        self.BM_1_4CT_line.setEnabled(False)  # 背部肌肉的第一个四分位数CT值
        self.VF_1_4CT_line.setEnabled(False)  # 内脏脂肪的第一个四分位数CT值
        self.BM_Median_line.setEnabled(False)  # 背部肌肉的中位数CT值
        self.SF_Median_line.setEnabled(False)  # 皮下脂肪的中位数CT值
        
        if self.crop_img is None or self.crop_fatmus_mask is None:
            self.statusbar.showMessage("Cropped image or mask not loaded.")
            return

        features = {}
        self.extract_radiomics_features(features)
        self.statusbar.showMessage("Calculated the 3D data of fat and muscle.")
        return excel_3d_data, data_3d_name

    def extract_radiomics_features(self, features):

    # 创建特征提取器
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.disableAllFeatures()
        extractor.enableFeatureClassByName('firstorder')
        extractor.enableFeatureClassByName('glcm')
        extractor.enableFeatureClassByName('glszm')
        extractor.enableFeatureClassByName('glrlm')
        extractor.enableImageTypeByName('Wavelet')

        desired_features_MUS = self.desired_features_MUS
        desired_features_FAT = self.desired_features_FAT
        # 提取肌肉区域的放射组学特征
        muscle_label = 3  # 腹部肌肉的标签
        muscle_mask_array = (self.crop_fatmus_mask == muscle_label).astype(np.uint8)
        muscle_mask_sitk = sitk.GetImageFromArray(muscle_mask_array)
        muscle_image_sitk = sitk.GetImageFromArray(self.crop_img)
        muscle_mask_sitk.CopyInformation(muscle_image_sitk)

        muscle_features = extractor.execute(muscle_image_sitk, muscle_mask_sitk)
        for feature_name, variable_name in desired_features_MUS.items():
            value = muscle_features.get('wavelet' + feature_name.split('wavelet')[1])
            if value is not None:
                print("value=",value)
                self.features = np.append(self.features,value)
                #setattr(self, variable_name, value)
                #

        # 提取脂肪区域的放射组学特征
        fat_label = 2  # 内脏脂肪的标签
        fat_mask_array = (self.crop_fatmus_mask == fat_label).astype(np.uint8)
        fat_mask_sitk = sitk.GetImageFromArray(fat_mask_array)
        fat_image_sitk = sitk.GetImageFromArray(self.crop_img)
        fat_mask_sitk.CopyInformation(fat_image_sitk)

        fat_features = extractor.execute(fat_image_sitk, fat_mask_sitk)
        for feature_name, variable_name in desired_features_FAT.items():
            value = fat_features.get('wavelet' + feature_name.split('wavelet')[1])
            if value is not None:
                print("value=",value)
                self.features = np.append(self.features, value)
                #setattr(self, variable_name, value)
                #self.features[variable_name] = value

    def show_mask(self):
        fname = QFileDialog.getOpenFileName(self, caption='Load mask', directory='data',
                                            filter="Image(*.nii *.nii.gz)")
        if len(fname[1]) != 0 and self.img is not None:
            img = itk.ReadImage(fname[0])
            self.showmask = itk.GetArrayFromImage(img)
            if self.showmask.shape == self.img.shape:
                self.affine = nib.load(self.img_path).affine
                self.mask_np = nib.load(fname[0]).get_fdata()
                self.img_np = nib.load(self.img_path).get_fdata()
                self.new_img = self.mask_np * self.img_np
                self.new_img = nib.Nifti1Image(self.new_img, affine=self.affine)
                nib.save(self.new_img, os.path.join('data', 'new_img.nii.gz'))
                if self.prinimg is not None:
                    self.printmask_fatmus(0.5)
                    self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                    self.draw_line(self.x_x, self.x_y, self.y_x, self.y_y, self.z_x, self.z_y)
                    reader = vtkNIFTIImageReader()
                    reader.SetFileName(os.path.join('data', 'new_img.nii.gz'))
                    reader.Update()
                    volumeMapper = vtkGPUVolumeRayCastMapper()
                    volumeMapper.SetInputData(reader.GetOutput())

                    volumeProperty = vtkVolumeProperty()
                    volumeProperty.SetInterpolationTypeToLinear()  # 设置体绘制的属性设置，决定体绘制的渲染效果
                    volumeProperty.ShadeOn()  # 打开或者关闭阴影
                    volumeProperty.SetAmbient(0.4)
                    volumeProperty.SetDiffuse(0.6)  # 漫反射
                    volumeProperty.SetSpecular(0.2)  # 镜面反射
                    # 设置不透明度
                    compositeOpacity = vtkPiecewiseFunction()
                    compositeOpacity.AddPoint(70, 0.00)
                    compositeOpacity.AddPoint(90, 0.4)
                    compositeOpacity.AddPoint(180, 0.6)
                    volumeProperty.SetScalarOpacity(compositeOpacity)  # 设置不透明度

                    # 设置梯度不透明属性
                    volumeGradientOpacity = vtkPiecewiseFunction()
                    volumeGradientOpacity.AddPoint(10, 0.0)
                    volumeGradientOpacity.AddPoint(90, 0.5)
                    volumeGradientOpacity.AddPoint(100, 1.0)

                    # volumeProperty.SetGradientOpacity(volumeGradientOpacity)  # 设置梯度不透明度效果对比
                    # 设置颜色属性
                    color = vtkColorTransferFunction()
                    color.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
                    color.AddRGBPoint(64.0, 1.0, 0.52, 0.3)
                    color.AddRGBPoint(109.0, 1.0, 1.0, 1.0)
                    color.AddRGBPoint(220.0, 0.2, 0.2, 0.2)
                    volumeProperty.SetColor(color)

                    volume = vtkVolume()  # 和vtkActor作用一致
                    volume.SetMapper(volumeMapper)
                    volume.SetProperty(volumeProperty)
                    if self.volume_old is not None:
                        self.ren.RemoveViewProp(self.volume_old)
                    self.ren.AddViewProp(volume)
                    self.volume_old = volume
                    # self.volume_path = fname[0]
                    camera = self.ren.GetActiveCamera()
                    c = volume.GetCenter()

                    camera.SetViewUp(0, 0, 1)
                    camera.SetViewAngle(60)
                    camera.SetPosition(c[0] - 300, c[1] - 600, c[2])
                    camera.SetFocalPoint(c[0], c[1] - 200, c[2])
                    # camera.SetPosition(c[0], c[1] - 800, c[2] - 200)
                    # camera.SetFocalPoint(c[0], c[1], c[2])
                    camera.Azimuth(30.0)
                    camera.Elevation(30.0)
                    self.iren.Initialize()
                    os.remove(os.path.join('data', 'new_img.nii.gz'))
                    self.statusbar.showMessage("The segmentation result has been displayed")
            else:
                # self.coord.setText('请加载与图对应的分割结果')
                self.statusbar.showMessage("Please load a correspronding segmentation result")
        elif self.img is None:
            # self.coord.setText('请先导入图像')
            self.statusbar.showMessage("Please load a image first")

    def clinicf_read(self):
        self.statusbar.showMessage("Please input data as required.")
    
    def showmask_path(self, path):
        self.showmask = read_nii(path)
        if self.showmask.shape == self.img.shape:
            self.mask_np = read_nii(path)
            self.img_np = read_nii(self.img_file)
            self.new_img = self.mask_np * self.img_np
            ori2new_nii(path, self.new_img, os.path.join('data', 'new_fatmus.nii.gz'))

            if self.prinimg is not None:
                self.printmask_fatmus(0.5)
                self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                self.draw_line(self.x_x, self.x_y, self.y_x, self.y_y, self.z_x, self.z_y)
                reader = vtkNIFTIImageReader()
                reader.SetFileName(os.path.join('data', 'new_fatmus.nii.gz'))
                reader.Update()

                volumeMapper = vtkGPUVolumeRayCastMapper()
                volumeMapper.SetInputData(reader.GetOutput())
                volumeProperty = vtkVolumeProperty()
                volumeProperty.SetInterpolationTypeToLinear()  # 设置体绘制的属性设置，决定体绘制的渲染效果
                volumeProperty.ShadeOn()  # 打开或者关闭阴影
                volumeProperty.SetAmbient(0.4)
                volumeProperty.SetDiffuse(0.6)  # 漫反射
                volumeProperty.SetSpecular(0.2)  # 镜面反射
                # 设置不透明度
                compositeOpacity = vtkPiecewiseFunction()
                compositeOpacity.AddPoint(70, 0.00)
                compositeOpacity.AddPoint(90, 0.4)
                compositeOpacity.AddPoint(180, 0.6)
                volumeProperty.SetScalarOpacity(compositeOpacity)  # 设置不透明度

                # 设置梯度不透明属性
                volumeGradientOpacity = vtkPiecewiseFunction()
                volumeGradientOpacity.AddPoint(10, 0.0)
                volumeGradientOpacity.AddPoint(90, 0.5)
                volumeGradientOpacity.AddPoint(100, 1.0)

                # volumeProperty.SetGradientOpacity(volumeGradientOpacity)  # 设置梯度不透明度效果对比
                # 设置颜色属性
                color = vtkColorTransferFunction()
                color.AddRGBPoint(0.0, 0.0, 0.0, 0.0)
                color.AddRGBPoint(64.0, 1.0, 0.52, 0.3)
                color.AddRGBPoint(109.0, 1.0, 1.0, 1.0)
                color.AddRGBPoint(220.0, 0.2, 0.2, 0.2)
                volumeProperty.SetColor(color)

                volume = vtkVolume()  # 和vtkActor作用一致
                volume.SetMapper(volumeMapper)
                volume.SetProperty(volumeProperty)
                if self.volume_old is not None:
                    self.ren.RemoveViewProp(self.volume_old)
                self.ren.AddViewProp(volume)
                self.volume_old = volume
                # self.volume_path = fname[0]
                camera = self.ren.GetActiveCamera()
                c = volume.GetCenter()
                camera.SetViewUp(0, 0, 1)
                # camera.SetPosition(c[0], c[1] - 800, c[2] - 200)
                # camera.SetFocalPoint(c[0], c[1], c[2])
                camera.SetViewAngle(60)
                camera.SetPosition(c[0] - 300, c[1] - 600, c[2])
                camera.SetFocalPoint(c[0], c[1] - 200, c[2])
                camera.Azimuth(30.0)
                camera.Elevation(30.0)
                self.iren.Initialize()
                os.remove(os.path.join('data', 'new_fatmus.nii.gz'))
                self.cal_fatmus_vol()

    def printmask_fatmus(self, alpha):
        new_prinimg = []
        # 设置每个标记类别的颜色
        color_map = {
            1: [255, 0, 0],  # 标记0的颜色为红色
            2: [0, 255, 0],  # 标记1的颜色为绿色
            3: [0, 0, 255],  # 标记2的颜色为蓝色
            4: [255, 255, 0],  # 标记3的颜色为黄色
            0: [0, 0, 0]  # 标记4的颜色为紫色
        }
        for i in range(self.leng_max):
            imgone = self.ori_prinimg[i, ...]
            maskone = self.showmask[(self.leng_max - 1) - i, ...]
            imgone = self.normalize(imgone)
            imgone = np.repeat(np.expand_dims(imgone, axis=-1), 3, axis=-1)
            maskone_ = np.repeat(np.expand_dims(maskone, axis=-1), 3, axis=-1)
            # 为每个标记类别设置颜色
            mask_colored = np.zeros_like(maskone_)
            for k in range(3):
                for label in color_map:
                    mask_one_k = maskone_[:, :, k]
                    mask_colored_k = mask_colored[:, :, k]
                    mask_colored_k[mask_one_k == label] = color_map[label][k]
                    mask_colored[:, :, k] = mask_colored_k
            # 将颜色叠加到图像上
            cam_img = alpha * mask_colored + 1 * imgone
            new_prinimg.append(cam_img[None, :, :, :])
        new_prinimg = np.concatenate(new_prinimg, axis=0)
        self.prinimg = new_prinimg
        return new_prinimg

    def printmask_spine(self, alpha):
        new_prinimg = []
        # 设置每个标记类别的颜色
        color_map = {
            19: [255, 0, 0],  # 标记0的颜色为红色
            20: [0, 255, 0],  # 标记1的颜色为绿色
            21: [0, 0, 255],  # 标记2的颜色为蓝色
            22: [255, 255, 0],  # 标记3的颜色为黄色
            0: [0, 0, 0]  # 标记4的颜色为紫色
        }
        for i in range(self.leng_max):
            imgone = self.ori_prinimg[i, ...]
            maskone = self.showmask[(self.leng_max - 1) - i, ...]
            imgone = self.normalize(imgone)
            imgone = np.repeat(np.expand_dims(imgone, axis=-1), 3, axis=-1)
            maskone_ = np.repeat(np.expand_dims(maskone, axis=-1), 3, axis=-1)
            # 为每个标记类别设置颜色
            mask_colored = np.zeros_like(maskone_)
            for k in range(3):
                for label in color_map:
                    mask_one_k = maskone_[:, :, k]
                    mask_colored_k = mask_colored[:, :, k]
                    mask_colored_k[mask_one_k == label] = color_map[label][k]
                    mask_colored[:, :, k] = mask_colored_k
            # 将颜色叠加到图像上
            cam_img = alpha * mask_colored + 1 * imgone
            new_prinimg.append(cam_img[None, :, :, :])
        new_prinimg = np.concatenate(new_prinimg, axis=0)
        self.prinimg = new_prinimg
        return new_prinimg
    
    def is_number(self, str):
        try:
            if str == 'NaN':
                return False
            float(str)
            return True
        except ValueError:
            return False
    
    def show_message_predict(self):
        QMessageBox.information(self, "Note", "The prediction is about to be made and will take longer if no "
                                              "segmentation results are entered and automatic segmentation is not performed",
                                QMessageBox.Yes)

    def show_message_compute_volume(self):
        QMessageBox.information(self, "Note", "If the mask is not imported in advance and there are no related "
                                              "files in the directory, the segmentation will be performed automatically, "
                                              "which will take a longer time",
                                QMessageBox.Yes)

    def show_message_incompelete_data(self):
        QMessageBox.information(self, "Note", " Incomplete input data, please input data as required.", QMessageBox.Yes)

    def show_message_fatmus(self):
        QMessageBox.information(self, "Note", "Fat and muscle will be extracted. This will spread a few minutes.",
                                QMessageBox.Yes)

    def show_message_spine(self):
        QMessageBox.information(self, "Note", "Spine will be extracted. This will spread a few minutes.",
                                QMessageBox.Yes)

    def predictf(self):
        values = []
        input_clinic = True

        self.Sodium = self.Sodium_line.text()
        self.RBC = self.RBC_line.text()
        self.INR = self.INR_line.text()
        self.Albumin = self.Albumin_line.text()
        self.Creatinine = self.Creatinine_line.text()
        self.Height = self.Height_line.text()
        values = [self.Sodium,self.RBC,self.INR,self.Albumin, self.Creatinine,self.Height]
        for nn in values:
            if nn is None:
                input_clinic = False
        if input_clinic:
            print(values)
            self.Sodium = float(self.Sodium_line.text()) if self.Sodium_line.text() else None
            self.RBC = float(self.RBC_line.text()) if self.RBC_line.text() else None
            self.INR = float(self.INR_line.text()) if self.INR_line.text() else None
            self.Albumin = float(self.Albumin_line.text()) if self.Albumin_line.text() else None
            self.Creatinine = float(self.Creatinine_line.text()) if self.Creatinine_line.text() else None
            self.Height = float(self.Height_line.text())if self.Height_line.text() else None

            beta_Sodium = -0.1209   
            beta_RBC = -0.2428
            beta_INR = 1.221
            beta_Creatinine = 0.001403
            beta_Albumin = 0.07301
            beta_vf_median = 0.017
            beta_am_median = -0.009113
            beta_am_volume = -0.4701
            beta_bm_25 = 0.1702
            beta_vf_25 = -0.4675
            beta_bm_median = -0.02085
            beta_sf_median = -0.02164
            beta_mus_lhl_glszm_var = -1.153
            beta_mus_lll_glcm_cluster = -1.376
            beta_mus_lhl_entropy = 2.845
            beta_mus_lhl_glcm_joint = 0.2725
            beta_mus_hhl_10perc = -0.82
            beta_fat_lhh_glrlm_var = -0.00198
            beta_fat_hlh_mean = -0.2725

    # 将所有变量代入到计算公式中
            score = (
                0.04586 +
                beta_Sodium * self.Sodium +
                beta_RBC * self.RBC +
                beta_INR * self.INR +
                beta_Albumin * self.Albumin +
                beta_Creatinine * self.Creatinine +
                beta_vf_median * self.vf_median +  # 中位数CT值-内脏脂肪组织
                beta_am_median * self.am_median +  # 中位数CT值-腹部肌肉
                beta_am_volume * (self.am_volume_normalized / self.Height) +  # 腹部肌肉归一化体积
                beta_bm_25 * self.bm_25th_percentile +  # 第一四分位CT值-背部肌肉
                beta_vf_25 * self.vf_25th_percentile +  # 第一四分位CT值-内脏脂肪
                beta_bm_median * self.bm_median +  # 中位数CT值-背部肌肉
                beta_sf_median * self.sf_median +  # 中位数CT值-皮下脂肪组织
                beta_mus_lhl_glszm_var * self.features[0]+  # LHL波段-灰度方差
                beta_mus_lll_glcm_cluster * self.features[1] +  # LLL波段-聚类倾向
                beta_mus_lhl_entropy * self.features[2]+ # LHL波段-熵
                beta_mus_lhl_glcm_joint * self.features[3]+  # LHL波段-联合熵
                beta_mus_hhl_10perc * self.features[4] +  # HHL波段-10百分位
                beta_fat_lhh_glrlm_var * self.features[5]+  # LHH波段-灰度方差
                beta_fat_hlh_mean * self.features[6]  # HLH波段-均值
                )
            print(features)

            print(score)
            score = -score
    # 计算概率
            pred = 1/(1+math.exp(score))
            if pred > 0.311:
                self.plotresult.setTextColor(QtCore.Qt.red)
            elif pred <= 0.311:
                self.plotresult.setTextColor(QtCore.Qt.black)
            self.plotresult.setText("{0:0.5f}".format(pred))
            self.statusbar.showMessage("The prediction is finished.")
        else:
            # self.coord.setText(f"输入不完整,请按要求导入数据")
            self.show_message_incompelete_data()
            self.statusbar.showMessage("Incomplete input data, please input data as required.")


    def spine_seg(self):
        if self.img is not None:
            self.spine_mask_file = os.path.join(self.spine_mask_path, self.file_name)
            if not os.path.exists(self.spine_mask_file):
                nnunet_spine_nii_path = os.path.join('data', 'nnunet_in', 'spine_0_0000.nii.gz')
                nnunet_spine_mask_path = os.path.join('data', 'nnunet_out', 'spine_0.nii.gz')
                if os.path.exists(nnunet_spine_nii_path):
                    os.remove(nnunet_spine_nii_path)
                shutil.copy(self.img_file, nnunet_spine_nii_path)
                self.statusbar.showMessage("Start to segment spine, please wait a moment")
                self.show_message_spine()
                os.system("nnUNet_predict -i ./data/nnunet_in "
                          "-o ./data/nnunet_out -t 057 -f 0  -m 3d_fullres -tr nnUNetTrainerV2 -chk model_best")
                shutil.copy(nnunet_spine_mask_path, self.spine_mask_file)
                os.remove(nnunet_spine_nii_path)
                os.remove(nnunet_spine_mask_path)
                os.remove(os.path.join('data', 'nnunet_out', 'plans.pkl'))
                self.statusbar.showMessage("Spine segmentation has been done")
            else:
                self.statusbar.showMessage("Spine segmentation has been existed.")

            if os.path.exists(self.fatmus_mask_file):
                self.spine_crop_img()
            if os.path.exists(self.crop_fatmus_mask_file):
                self.showmask_path(self.crop_fatmus_mask_file)
        else:
            # self.coord.setText(f"请先导入图像")
            self.statusbar.showMessage('Please load an image first')

    def fatmus_seg(self):
        if self.img is not None:
            self.fatmus_mask_file = os.path.join(self.fatmus_mask_path, self.file_name)
            if not os.path.exists(self.fatmus_mask_file):
                nnunet_fatmus_nii_path = os.path.join('data', 'nnunet_in', 'fatmus_0_0000.nii.gz')
                nnunet_fatmus_mask_path = os.path.join('data', 'nnunet_out', 'fatmus_0.nii.gz')
                if os.path.exists(nnunet_fatmus_nii_path):
                    os.remove(nnunet_fatmus_nii_path)

                shutil.copy(self.img_file, nnunet_fatmus_nii_path)
                self.statusbar.showMessage("Start to segment spine, please wait a moment.")
                self.show_message_fatmus()
                os.system("nnUNet_predict -i data/nnunet_in "
                          "-o data/nnunet_out -t 006 -f 1 -m 2d -tr nnUNetTrainerV2 -chk model_best")
                shutil.copy(nnunet_fatmus_mask_path, self.fatmus_mask_file)
                os.remove(nnunet_fatmus_nii_path)
                os.remove(nnunet_fatmus_mask_path)
                os.remove(os.path.join('data', 'nnunet_out', 'plans.pkl'))
                fatmus_mask = itk.ReadImage(self.fatmus_mask_file)
                self.fatmus_mask = itk.GetArrayFromImage(fatmus_mask)
                self.statusbar.showMessage("Fat and muscle segmentation has been done.")
            else:
                self.statusbar.showMessage("Fat and muscle segmentation has been done.")

            if os.path.exists(self.spine_mask_file):
                self.spine_crop_img()
            if os.path.exists(self.crop_fatmus_mask_file):
                self.showmask_path(self.crop_fatmus_mask_file)
        else:
            # self.coord.setText(f"请先导入图像")
            self.statusbar.showMessage('Please load an image first.')
    
    def spine_crop_img(self):
        begin_slice_muscle = 1000
        end_slice_muscle = 0
        begin_slice_fat = 1000
        end_slice_fat = 0
        l3 = 22
        t12 = 19
        l2 = 21
        if self.img is not None:
            self.crop_spine_mask_file = os.path.join(self.crop_spine_mask_path, self.file_name)
            self.crop_fatmus_mask_file = os.path.join(self.crop_fatmus_mask_path, self.file_name)
            self.crop_img_file = os.path.join(self.crop_img_path, self.file_name)
            if not os.path.exists(self.crop_spine_mask_file) or not os.path.exists(self.crop_fatmus_mask_file):

                spine_mask_data = read_nii(self.spine_mask_file)
                origin_nii_data = read_nii(self.img_file)
                fatmus_mask_data = read_nii(self.fatmus_mask_file)

                # 初始化新数组
                new_img_muscle = np.zeros(origin_nii_data.shape)
                new_fatmus_mask_muscle = np.zeros(fatmus_mask_data.shape)
                new_img_fat = np.zeros(origin_nii_data.shape)
                new_fatmus_mask_fat = np.zeros(fatmus_mask_data.shape)

                # 处理肌肉部分（T12 到 L3）
                if t12 in spine_mask_data and l3 in spine_mask_data:
                    spine_mask_data_muscle = spine_mask_data.copy()
                    spine_mask_data_muscle = remove_archs_spine_mask(spine_mask_data_muscle)
                    spine_mask_data_muscle[spine_mask_data_muscle < t12] = 0
                    spine_mask_data_muscle[spine_mask_data_muscle > l3] = 0
                    spine_mask_data_copy_muscle = spine_mask_data_muscle.copy()
                    spine_mask_data_copy_muscle[spine_mask_data_copy_muscle != 0] = 1
                    for d in range(spine_mask_data_copy_muscle.shape[0]):
                        if 1 in spine_mask_data_copy_muscle[d, :, :]:
                            if d < begin_slice_muscle:
                                begin_slice_muscle = d
                            if d > end_slice_muscle:
                                end_slice_muscle = d
                    if end_slice_muscle > begin_slice_muscle:
                        new_fatmus_mask_muscle[begin_slice_muscle:end_slice_muscle + 1] = fatmus_mask_data[begin_slice_muscle:end_slice_muscle + 1]
                        new_img_muscle[begin_slice_muscle:end_slice_muscle + 1] = origin_nii_data[begin_slice_muscle:end_slice_muscle + 1]
                    else:
                        new_fatmus_mask_muscle[end_slice_muscle:begin_slice_muscle + 1] = fatmus_mask_data[end_slice_muscle:begin_slice_muscle + 1]
                        new_img_muscle[end_slice_muscle:begin_slice_muscle + 1] = origin_nii_data[end_slice_muscle:begin_slice_muscle + 1]
                        a = begin_slice_muscle
                        begin_slice_muscle = end_slice_muscle
                        end_slice_muscle = a
                else:
                    self.statusbar.showMessage(
                        "This image does not contain T12 or L3, which cannot calculate the muscle volume correctly. Please change another image.")
                    return

                # 处理脂肪部分（L2 到 L3）
                if l2 in spine_mask_data and l3 in spine_mask_data:
                    spine_mask_data_fat = spine_mask_data.copy()
                    spine_mask_data_fat = remove_archs_spine_mask(spine_mask_data_fat)
                    spine_mask_data_fat[spine_mask_data_fat < l2] = 0
                    spine_mask_data_fat[spine_mask_data_fat > l3] = 0
                    spine_mask_data_copy_fat = spine_mask_data_fat.copy()
                    spine_mask_data_copy_fat[spine_mask_data_copy_fat != 0] = 1
                    for d in range(spine_mask_data_copy_fat.shape[0]):
                        if 1 in spine_mask_data_copy_fat[d, :, :]:
                            if d < begin_slice_fat:
                                begin_slice_fat = d
                            if d > end_slice_fat:
                                end_slice_fat = d
                    if end_slice_fat > begin_slice_fat:
                        new_fatmus_mask_fat[begin_slice_fat:end_slice_fat + 1] = fatmus_mask_data[begin_slice_fat:end_slice_fat + 1]
                        new_img_fat[begin_slice_fat:end_slice_fat + 1] = origin_nii_data[begin_slice_fat:end_slice_fat + 1]
                    else:
                        new_fatmus_mask_fat[end_slice_fat:begin_slice_fat + 1] = fatmus_mask_data[end_slice_fat:begin_slice_fat + 1]
                        new_img_fat[end_slice_fat:begin_slice_fat + 1] = origin_nii_data[end_slice_fat:begin_slice_fat + 1]
                        a = begin_slice_fat
                        begin_slice_fat = end_slice_fat
                        end_slice_fat = a
                else:
                    self.statusbar.showMessage(
                        "This image does not contain L2 or L3, which cannot calculate the fat volume correctly. Please change another image.")
                    return

                # 合并肌肉和脂肪的裁剪结果
                new_fatmus_mask = np.maximum(new_fatmus_mask_muscle, new_fatmus_mask_fat)
                new_img = np.maximum(new_img_muscle, new_img_fat)

                # 保存裁剪后的结果
                ori2new_nii(self.fatmus_mask_file, new_fatmus_mask, self.crop_fatmus_mask_file)
                ori2new_nii(self.img_file, new_img, self.crop_img_file)
        else:
            self.statusbar.showMessage('Please load an image first.')
    # 定义获取梯度的函数
    def backward_hook(self, module, grad_in, grad_out):
        self.grad_block.append(grad_out[0].detach())

    # 定义获取特征图的函数
    def farward_hook(self, module, input, output):
        self.fmap_block.append(output)

    def normalize(self, x):
        maa = x.max()
        mii = x.min()
        x = (x - mii) * 255 / (maa - mii)
        return x
    
    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            if self.face_flage == 1 and self.leng_img != -100:
                self.leng_img = self.leng_img + 1
                if self.leng_img >= self.leng_max:
                    self.leng_img = self.leng_max - 1
                self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                self.y_y = int(round(self.face_h * (self.leng_img / self.leng_max), 0))
                self.z_y = int(round(self.face_h * (self.leng_img / self.leng_max), 0))
                self.draw_line(self.x_x, self.x_y, self.y_x, self.y_y, self.z_x, self.z_y)
            elif self.face_flage == 2 and self.leng_img != -100:
                self.width_img = self.width_img + 1
                if self.width_img >= self.width_max:
                    self.width_img = self.width_max - 1
                self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                self.x_y = int(round(self.face_h * (self.width_img / self.width_max), 0))
                self.z_x = int(round(self.face_w * (self.width_img / self.width_max), 0))
                self.draw_line(self.x_x, self.x_y, self.y_x, self.y_y, self.z_x, self.z_y)
            elif self.face_flage == 3 and self.leng_img != -100:
                self.high_img = self.high_img + 1
                if self.high_img >= self.high_max:
                    self.high_img = self.high_max - 1
                self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                self.x_x = int(round(self.face_w * (self.high_img / self.high_max), 0))
                self.y_x = int(round(self.face_w * (self.high_img / self.high_max), 0))
                self.draw_line(self.x_x, self.x_y, self.y_x, self.y_y, self.z_x, self.z_y)


        elif event.angleDelta().y() < 0:
            if self.face_flage == 1 and self.leng_img != -100:
                self.leng_img = self.leng_img - 1
                if self.leng_img < 0:
                    self.leng_img = 0
                self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                self.y_y = int(round(self.face_h * (self.leng_img / self.leng_max), 0))
                self.z_y = int(round(self.face_h * (self.leng_img / self.leng_max), 0))
                self.draw_line(self.x_x, self.x_y, self.y_x, self.y_y, self.z_x, self.z_y)

            elif self.face_flage == 2 and self.leng_img != -100:
                self.width_img = self.width_img - 1
                if self.width_img < 0:
                    self.width_img = 0
                self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                self.x_y = int(round(self.face_h * (self.width_img / self.width_max), 0))
                self.z_x = int(round(self.face_w * (self.width_img / self.width_max), 0))
                self.draw_line(self.x_x, self.x_y, self.y_x, self.y_y, self.z_x, self.z_y)



            elif self.face_flage == 3 and self.leng_img != -100:
                self.high_img = self.high_img - 1
                if self.high_img < 0:
                    self.high_img = 0
                self.showpic_xyz(self.leng_img, self.width_img, self.high_img, self.face_w, self.face_h)
                self.x_x = int(round(self.face_w * (self.high_img / self.high_max), 0))
                self.y_x = int(round(self.face_w * (self.high_img / self.high_max), 0))
                self.draw_line(self.x_x, self.x_y, self.y_x, self.y_y, self.z_x, self.z_y)

if __name__ == '__main__':
    from qt_material import apply_stylesheet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme='dark_yellow.xml')

    mw = NewWindow()
    mw.show()
    sys.exit(app.exec_())

