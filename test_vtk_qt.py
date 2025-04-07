import sys
import logging
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# 配置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

class VTKWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        logging.info("初始化VTKWidget...")
        
        # 创建布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建VTK部件
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget)
        
        # 创建渲染器
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(1, 1, 1)  # 白色背景
        
        # 获取渲染窗口
        self.render_window = self.vtk_widget.GetRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        
        # 获取交互器
        self.interactor = self.render_window.GetInteractor()
        
        # 设置交互样式
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)
        
        # 初始化但不启动
        self.interactor.Initialize()
        
        logging.info("VTKWidget初始化完成")
    
    def GetRenderWindow(self):
        return self.render_window
    
    def GetRenderer(self):
        return self.renderer
    
    def GetInteractor(self):
        return self.interactor

class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        logging.info("初始化主窗口...")
        
        self.setWindowTitle("VTK测试窗口")
        self.setGeometry(100, 100, 800, 600)
        
        # 创建中央部件
        logging.info("创建中央部件...")
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建布局
        logging.info("创建布局...")
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # 添加标签
        logging.info("添加标签...")
        label = QLabel("这是一个测试窗口")
        layout.addWidget(label)
        
        # 创建VTK部件
        logging.info("创建VTK部件...")
        try:
            self.vtk_widget = VTKWidget()
            layout.addWidget(self.vtk_widget)
            
            # 创建一个简单的球体
            sphere_source = vtk.vtkSphereSource()
            sphere_source.SetCenter(0, 0, 0)
            sphere_source.SetRadius(1.0)
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere_source.GetOutputPort())
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # 红色
            
            self.vtk_widget.GetRenderer().AddActor(actor)
            self.vtk_widget.GetRenderer().ResetCamera()
            
            logging.info("VTK部件创建完成")
            
        except Exception as e:
            logging.error(f"创建VTK部件失败: {str(e)}")
            raise

    def showEvent(self, event):
        """重写showEvent以确保正确初始化"""
        super().showEvent(event)
        logging.info("窗口显示事件触发")
        
        if hasattr(self, 'vtk_widget'):
            logging.info("启动交互器...")
            self.vtk_widget.GetInteractor().Start()
            logging.info("交互器启动成功")

if __name__ == "__main__":
    try:
        logging.info("启动程序...")
        
        # 创建QApplication实例
        app = QApplication(sys.argv)
        logging.info("QApplication创建成功")
        
        # 创建主窗口
        window = TestWindow()
        logging.info("主窗口创建成功")
        
        # 显示窗口
        window.show()
        logging.info("窗口显示成功")
        
        # 启动事件循环
        logging.info("进入事件循环...")
        sys.exit(app.exec_())
        
    except Exception as e:
        logging.error(f"程序启动失败: {str(e)}")
        sys.exit(1) 