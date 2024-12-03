from PyQt5.QtCore import Qt, QSize
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication
# import PyQt5_stylesheets
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtk.vtkCommonColor import vtkNamedColors
from vtk.vtkRenderingCore import (
    vtkColorTransferFunction,
    vtkRenderer,
    vtkRenderWindow,
    vtkVolume,
    vtkVolumeProperty
)

def set_table_data(table_widget, row, column, data):
    # 创建表格�?
    item = QtWidgets.QTableWidgetItem(data)
    # 设置表格项的对齐方式
    item.setTextAlignment(0x0002)  # 居中对齐
    # 将表格项添加到表格部件的指定位置
    table_widget.setItem(row, column, item)

class Ui_MainWindow(object):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.actionsegmente_fat_and_muscle = None
        self.aspect_ratio = 1051 / 645 #窗口的高宽比
    
    def resizeEvent(self, event): #这是一个事件处理方法，用于处理窗口大小调整事件。当用户调整窗口大小时，这个方法会被自动调用。
        print("resizeEvent triggered")
        current_size = self.size()
        new_width = current_size.width()
        new_height = int(new_width / self.aspect_ratio)
        self.resize(new_width, new_height)
        view_size = self.view1.size()
        view_w = view_size.width()
        view_h = view_size.height()
        self.vtkWidget.resize(view_w, view_h)

    def __initWidget(self):
        self.view1.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view1.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view2.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view3.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view3.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # 以鼠标所在位置为锚点进行缩放
        self.view1.setTransformationAnchor(self.view1.AnchorUnderMouse)
        self.view2.setTransformationAnchor(self.view2.AnchorUnderMouse)
    
    def menu_ui(self,MainWindow):
        #创建菜单栏
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(15)
        font.setBold(True)
        self.menubar.setFont(font)

        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuSegmentation = QtWidgets.QMenu(self.menubar)
        self.menuSegmentation.setObjectName("menuSegmentation")
        self.menu_display_mask = QtWidgets.QMenu(self.menubar)
        self.menu_display_mask.setObjectName("menu_display_mask")
        self.menuclinic_data = QtWidgets.QMenu(self.menubar)
        self.menuclinic_data.setObjectName("menuclinic_data")

        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        # self.statusbar.setStyleSheet('background-color: black; color: #FFA500; font-weight: bold;')
        self.statusbar.setFont(font)

        self.actionopen_file = QtWidgets.QAction(MainWindow)
        self.actionopen_file.setObjectName("actionopen_file")
        self.actionsegmente_spine = QtWidgets.QAction(MainWindow)
        self.actionsegmente_spine.setObjectName("actionsegmente_spine")
        self.actionsegmente_fat_and_muscle = QtWidgets.QAction(MainWindow)
        self.actionsegmente_fat_and_muscle.setObjectName("actionsegmente_fat_and_muscle")
        self.action_display_FM = QtWidgets.QAction(MainWindow)
        self.action_display_FM.setObjectName("action_display_FM")
        self.action_display_S = QtWidgets.QAction(MainWindow)
        self.action_display_S.setObjectName("action_display_S")
        self.action_predict = QtWidgets.QAction(MainWindow)
        self.action_predict.setObjectName("action_predict")

        self.menuFile.addAction(self.actionopen_file)
        self.menuSegmentation.addAction(self.actionsegmente_spine)
        self.menuSegmentation.addAction(self.actionsegmente_fat_and_muscle)
        self.menu_display_mask.addAction(self.action_display_S)
        self.menu_display_mask.addAction(self.action_display_FM)
        self.menuclinic_data.addAction(self.action_predict)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuSegmentation.menuAction())
        self.menubar.addAction(self.menu_display_mask.menuAction())
        self.menubar.addAction(self.menuclinic_data.menuAction())

        #连接动作的触发信号到槽函数
        self.actionopen_file.triggered.connect(MainWindow.showpic)
        self.actionsegmente_spine.triggered.connect(MainWindow.spine_seg)
        self.actionsegmente_fat_and_muscle.triggered.connect(MainWindow.fatmus_seg)
        self.action_display_FM.triggered.connect(MainWindow.show_fatmus_mask)
        self.action_display_S.triggered.connect(MainWindow.show_spine_mask)
        self.action_predict.triggered.connect(MainWindow.predictf)

    def setupUi(self,MainWindow):
        print('begin-------------------------------------')
        MainWindow.setGeometry(100, 100, 1082, 663)
        MainWindow.setWindowTitle("Predict")
        MainWindow.setAutoFillBackground(True)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
         # self.centralwidget.setGeometry(100, 100, 1200, 840)
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menu_ui(MainWindow)
        allLayout = QtWidgets.QHBoxLayout()

        viewwidget = QtWidgets.QWidget()
        viewLayout = QtWidgets.QGridLayout()

        self.scene1 = QtWidgets.QGraphicsScene()
        self.view1 = QtWidgets.QGraphicsView()
        self.scene2 = QtWidgets.QGraphicsScene()
        self.view2 = QtWidgets.QGraphicsView()
        self.scene3 = QtWidgets.QGraphicsScene()
        self.view3 = QtWidgets.QGraphicsView()
        self._color_background = QtGui.QColor('#000000')
        self.scene1.setBackgroundBrush(self._color_background)
        self.scene2.setBackgroundBrush(self._color_background)
        self.scene3.setBackgroundBrush(self._color_background)
        self.view1.setScene(self.scene1)
        self.view2.setScene(self.scene2)
        self.view3.setScene(self.scene3)

        self.view1.setStyleSheet(
            'border-width: 1px;border-style: solid;border-color: rgb(255, 255, 255, 255);background-color: rgb(0, 0, 0, 0);')
        self.view2.setStyleSheet(
            'border-width: 1px;border-style: solid;border-color: rgb(255, 255, 255, 255);background-color: rgb(0, 0, 0, 0);')
        self.view3.setStyleSheet(
            'border-width: 1px;border-style: solid;border-color: rgb(255, 255, 255, 255);background-color: rgb(0, 0, 0, 0);')
        self.__initWidget()

        self.pic_box_vface = QtWidgets.QGroupBox(self.centralwidget)
        self.pic_box_vface.setStyleSheet(
            'border-width: 0px;border-style: solid;border-color: rgb(255, 255, 255);background-color: rgba(0, 0, 0, 0) ;')

        self.ren = vtkRenderer()
        self.vtkWidget = QVTKRenderWindowInteractor(self.pic_box_vface)

        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.colors = vtkNamedColors()
        self.vtkpros = {}
        self.colors.SetColor('BkgColor', [0, 0, 0, 255])
        self.ren.SetBackground(self.colors.GetColor3d('BkgColor'))
        self.iren.Initialize()
        """Veiw123 放在0,0 0,1 1,0 picbox放在1,1 将状态栏添加到第 2 行第 0 列，并跨越 1 行 2 列"""
        viewLayout.addWidget(self.view1, 0, 0)
        viewLayout.addWidget(self.view2, 0, 1)
        viewLayout.addWidget(self.view3, 1, 0)
        viewLayout.addWidget(self.pic_box_vface, 1, 1)
        viewLayout.addWidget(self.statusbar, 2, 0, 1, 2)
        self.statusbar.showMessage('Ready')
        viewwidget.setLayout(viewLayout)

        # INPUT Box of info_layout--------------------------------------------------------------------------------------
        input_wdg = QtWidgets.QFrame()
        input_wdg.setFrameShape(QtWidgets.QFrame.Panel)
        input_wdg.setFrameShadow(QtWidgets.QFrame.Sunken)
        input_layout = QtWidgets.QGridLayout()
        input_label = QtWidgets.QLabel('InPut')
        Sodium_label = QtWidgets.QLabel('Sodium')
        Height_label = QtWidgets.QLabel('Height')
        RBC_label = QtWidgets.QLabel('RBC')
        INR_label = QtWidgets.QLabel('INR')
        Albumin_label = QtWidgets.QLabel('Albumin')
        Creatinine_label = QtWidgets.QLabel('Creatinine')
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(15)
        font.setBold(True)
        input_label.setFont(font)
    
        #创建输入框
        self.Sodium_line = QtWidgets.QLineEdit()
        self.Height_line = QtWidgets.QLineEdit()
        self.RBC_line = QtWidgets.QLineEdit()
        self.INR_line = QtWidgets.QLineEdit()
        self.Albumin_line = QtWidgets.QLineEdit()
        self.Creatinine_line = QtWidgets.QLineEdit()

        line_edit_style = "QLineEdit { border-radius: 3px; background-color: rgb(220, 220, 220)}"
        self.Sodium_line.setStyleSheet(line_edit_style)
        self.Height_line.setStyleSheet(line_edit_style)
        self.RBC_line.setStyleSheet(line_edit_style)
        self.INR_line.setStyleSheet(line_edit_style)
        self.Albumin_line.setStyleSheet(line_edit_style)
        self.Creatinine_line.setStyleSheet(line_edit_style)

        self.Sodium_line.setFixedHeight(37)
        self.Height_line.setFixedHeight(37)
        self.RBC_line.setFixedHeight(37)
        self.INR_line.setFixedHeight(37)
        self.Albumin_line.setFixedHeight(37)
        self.Creatinine_line.setFixedHeight(37)

        font_color = "QLineEdit {color: rgb(20,20,20); background-color:rgb(220,220,220);}"
        self.Sodium_line.setStyleSheet(font_color)
        self.Height_line.setStyleSheet(font_color)
        self.RBC_line.setStyleSheet(font_color)
        self.INR_line.setStyleSheet(font_color)
        self.Albumin_line.setStyleSheet(font_color)
        self.Creatinine_line.setStyleSheet(font_color)
        #设置标签字体
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.Sodium_line.setFont(font)
        self.Height_line.setFont(font)
        self.RBC_line.setFont(font)
        self.INR_line.setFont(font)
        self.Albumin_line.setFont(font)
        self.Creatinine_line.setFont(font)


        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        Sodium_label.setFont(font)
        Height_label.setFont(font)
        RBC_label.setFont(font)
        INR_label.setFont(font)
        Albumin_label.setFont(font)
        Creatinine_label.setFont(font)

         # 添加标签和输入框到布局
        input_layout.addWidget(input_label, 0, 0, 1, 4, Qt.AlignCenter)
          # 添加六个标签和输入框
        input_layout.addWidget(Sodium_label, 2, 0)
        input_layout.addWidget(self.Sodium_line, 2, 1)
        input_layout.addWidget(Height_label, 2, 2)
        input_layout.addWidget(self.Height_line, 2, 3)
        
        input_layout.addWidget(RBC_label, 3, 0)
        input_layout.addWidget(self.RBC_line, 3, 1)
        input_layout.addWidget(INR_label, 3, 2)
        input_layout.addWidget(self.INR_line, 3, 3)
        
        input_layout.addWidget(Albumin_label, 4, 0)
        input_layout.addWidget(self.Albumin_line, 4, 1)
        input_layout.addWidget(Creatinine_label, 4, 2)
        input_layout.addWidget(self.Creatinine_line, 4, 3)

         # 设置行伸缩因子
        input_layout.setRowStretch(0,1)
        input_layout.setRowStretch(1,1)
        input_layout.setRowStretch(2,1)
        input_layout.setRowStretch(3,1)
        input_layout.setRowStretch(4,1)
        input_wdg.setLayout(input_layout)
         # OUTPUT Box of info_layout-------------------------------------------------------------------------------------

        output_wdg = QtWidgets.QFrame()
        output_wdg.setFrameShape(QtWidgets.QFrame.Panel)
        output_wdg.setFrameShadow(QtWidgets.QFrame.Sunken)
        output_layout = QtWidgets.QGridLayout()
        output_label = QtWidgets.QLabel('Output')
        data_3d_label = QtWidgets.QLabel('3D data:')
        # 修改后的七个输出标签
        VF_Median_label = QtWidgets.QLabel('1. VF-Median')
        AM_Median_label = QtWidgets.QLabel('2. AM-Median')
        AM_Normal_label = QtWidgets.QLabel('3. AM-Normal')
        BM_1_4CT_label = QtWidgets.QLabel('4. BM-1/4CT')
        VF_1_4CT_label = QtWidgets.QLabel('5. VF-1/4CT')
        BM_Median_label = QtWidgets.QLabel('6. BM-Median')
        SF_Median_label = QtWidgets.QLabel('7. SF-Median')

        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(15)
        font.setBold(True)
        output_label.setFont(font)

         # 创建七个输出框
        self.VF_Median_line = QtWidgets.QLineEdit()
        self.AM_Median_line = QtWidgets.QLineEdit()
        self.AM_Normal_line = QtWidgets.QLineEdit()
        self.BM_1_4CT_line = QtWidgets.QLineEdit()
        self.VF_1_4CT_line = QtWidgets.QLineEdit()
        self.BM_Median_line = QtWidgets.QLineEdit()
        self.SF_Median_line = QtWidgets.QLineEdit()

        self.VF_Median_line.setFixedHeight(37)
        self.AM_Median_line.setFixedHeight(37)
        self.AM_Normal_line.setFixedHeight(37)
        self.BM_1_4CT_line.setFixedHeight(37)
        self.VF_1_4CT_line.setFixedHeight(37)
        self.BM_Median_line.setFixedHeight(37)
        self.SF_Median_line.setFixedHeight(37)

        self.VF_Median_line.setStyleSheet(line_edit_style)
        self.AM_Median_line.setStyleSheet(line_edit_style)
        self.AM_Normal_line.setStyleSheet(line_edit_style)
        self.BM_1_4CT_line.setStyleSheet(line_edit_style)
        self.VF_1_4CT_line.setStyleSheet(line_edit_style)
        self.BM_Median_line.setStyleSheet(line_edit_style)
        self.SF_Median_line.setStyleSheet(line_edit_style)

        font_color = "QLineEdit {color: rgb(20,20,20); background-color:rgb(220,220,220);}"
        self.VF_Median_line.setStyleSheet(font_color)
        self.AM_Median_line.setStyleSheet(font_color)
        self.AM_Normal_line.setStyleSheet(font_color)
        self.BM_1_4CT_line.setStyleSheet(font_color)
        self.VF_1_4CT_line.setStyleSheet(font_color)
        self.BM_Median_line.setStyleSheet(font_color)
        self.SF_Median_line.setStyleSheet(font_color)

        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(11)
        VF_Median_label.setFont(font)
        AM_Median_label.setFont(font)
        AM_Normal_label.setFont(font)
        BM_1_4CT_label.setFont(font)
        VF_1_4CT_label.setFont(font)
        BM_Median_label.setFont(font)
        SF_Median_label.setFont(font)

        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)
        self.VF_Median_line.setFont(font)
        self.AM_Median_line.setFont(font)
        self.AM_Normal_line.setFont(font)
        self.BM_1_4CT_line.setFont(font)
        self.VF_1_4CT_line.setFont(font)
        self.BM_Median_line.setFont(font)
        self.SF_Median_line.setFont(font)

        output_layout.addWidget(output_label, 0, 0, 1, 4, Qt.AlignCenter)
        output_layout.addWidget(data_3d_label, 1, 0)

        output_layout.addWidget(VF_Median_label, 2, 0)
        output_layout.addWidget(self.VF_Median_line, 2, 1)
        output_layout.addWidget(AM_Median_label, 2, 2)
        output_layout.addWidget(self.AM_Median_line, 2, 3)
        
        output_layout.addWidget(AM_Normal_label, 3, 0)
        output_layout.addWidget(self.AM_Normal_line, 3, 1)
        output_layout.addWidget(BM_1_4CT_label, 3, 2)
        output_layout.addWidget(self.BM_1_4CT_line, 3, 3)
        
        output_layout.addWidget(VF_1_4CT_label, 4, 0)
        output_layout.addWidget(self.VF_1_4CT_line, 4, 1)
        output_layout.addWidget(BM_Median_label, 4, 2)
        output_layout.addWidget(self.BM_Median_line, 4, 3)
        
        output_layout.addWidget(SF_Median_label, 5, 0)
        output_layout.addWidget(self.SF_Median_line, 5, 1)

        output_wdg.setLayout(output_layout)

        #Box of "Predicition 肝性脑病"----------------------------------------
        predict_layout = QtWidgets.QHBoxLayout()
        predict_wdg = QtWidgets.QFrame()
        predict_wdg.setFrameShape(QtWidgets.QFrame.Panel)
        predict_wdg.setFrameShadow(QtWidgets.QFrame.Sunken)
        predict_label = QtWidgets.QLabel('         Prediction Probability\n            > 0.311 High-risk\n            < 0.311 Low-risk')

        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(15)
        font.setBold(True)
        predict_label.setFont(font)

        self.plotresult = QtWidgets.QTextBrowser()  # 创建一个文本浏览器，用于显示预测结果
        self.plotresult.setStyleSheet('border-width: 1px;border-style: solid;background-color: rgb(220, 220, 220);')
        # 设置文本浏览器的样式表，包括边框宽度、边框样式和背景颜色
        self.plotresult.setLineWrapMode(QtWidgets.QTextBrowser.NoWrap)  # 禁用自动换行
        self.plotresult.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 禁用垂直滚动条
        self.plotresult.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 禁用水平滚动条
        self.plotresult.setFixedSize(140, 37)  # 设置文本浏览器的固定大小
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(12)  # 设置字体大小为 12
        self.plotresult.setFont(font)  # 将字体应用到文本浏览器

        predict_layout.addWidget(predict_label)  # 将标签添加到布局
        predict_layout.addWidget(self.plotresult)  # 将文本浏览器添加到布局
        predict_layout.setAlignment(self.plotresult, Qt.AlignLeft)  # 将文本浏览器向左对齐
        predict_wdg.setLayout(predict_layout)  # 将布局应用到框架

        info_layout = QtWidgets.QVBoxLayout()  # 创建垂直布局
        info_wdg = QtWidgets.QFrame()  # 创建一个框架，用于包含多个部分
        info_wdg.setFrameShape(QtWidgets.QFrame.Panel)  # 设置框架的形状
        info_wdg.setFrameShadow(QtWidgets.QFrame.Sunken)  # 设置框架的阴影效果

        info_layout.addWidget(input_wdg)  # 添加输入框架
        info_layout.addWidget(output_wdg)  # 添加输出框架
        info_layout.addWidget(predict_wdg)  # 添加预测框架
        info_layout.setStretch(2, 1)  # 设置预测框架在垂直布局中的伸缩因子
        info_layout.setStretch(0, 3)  # 设置输入框架在垂直布局中的伸缩因子
        info_layout.setStretch(1, 3)  # 设置输出框架在垂直布局中的伸缩因子
        info_wdg.setLayout(info_layout)  # 将垂直布局应用到信息框架

           # all layout ---------------------------------------------------------------------------------------------------
        allLayout.addWidget(viewwidget)
        allLayout.addWidget(info_wdg)
        allLayout.setStretch(0, 5)
        allLayout.setStretch(1, 4)

        self.centralwidget.setLayout(allLayout)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))

        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuSegmentation.setTitle(_translate("MainWindow", "Segmentation"))
        self.menu_display_mask.setTitle(_translate("MainWindow", "Display"))
        self.menuclinic_data.setTitle(_translate("MainWindow", "Predict"))
        self.actionopen_file.setText(_translate("MainWindow", "Open File"))
        self.actionsegmente_spine.setText(_translate("MainWindow", "Segment Spine"))
        self.actionsegmente_fat_and_muscle.setText(_translate("MainWindow", "Segment Fat and Muscle"))
        self.action_display_FM.setText(_translate("MainWindow", "Display Fat and Muscle"))
        self.action_display_S.setText(_translate("MainWindow", "Display Spine"))
        self.action_predict.setText(_translate("MainWindow", "Predicting the probability of hepatic encephalopathy"))
    
class Window_FM(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(Window_FM, self).__init__()

        self.setupUi()


if __name__ == "__main__":
    import sys
    from New_App_fatmus3 import Window_FM2
    from qt_material import apply_stylesheet

    app = QApplication(sys.argv)
    apply_stylesheet(app, theme='dark_teal.xml')
    # app.setStyleSheet(PyQt5_stylesheets.load_stylesheet_pyqt5(style='style_Classic'))
    import qdarkstyle
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
    widget = Window_FM2()

    widget.show()
    sys.exit(app.exec_())