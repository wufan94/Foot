import sys
import os
import numpy as np
import trimesh
import vtk
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QComboBox, QPushButton, 
                           QGroupBox, QFileDialog, QMessageBox, QProgressBar, 
                           QSpinBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from scipy.spatial import cKDTree
from analysis_functions import (simplify_model, compute_distances, 
                              convert_distance_to_pressure, get_material_type)
import logging
import traceback

# 配置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

# 设置递归限制
sys.setrecursionlimit(10000)

def handle_exception(exc_type, exc_value, exc_traceback):
    """全局异常处理函数"""
    logging.error("未捕获的异常:", exc_info=(exc_type, exc_value, exc_traceback))
    QMessageBox.critical(None, "错误", f"程序发生错误:\n{str(exc_value)}\n\n详细信息请查看日志文件。")

# 设置全局异常处理器
sys.excepthook = handle_exception

class ModelLoader(QThread):
    """模型加载线程"""
    progress = pyqtSignal(int)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, foot_path, shoe_path):
        super().__init__()
        self.foot_path = foot_path
        self.shoe_path = shoe_path
        
    def run(self):
        try:
            logging.info("开始加载模型...")
            
            # 加载足部模型
            logging.info(f"加载足部模型: {self.foot_path}")
            self.progress.emit(10)
            try:
                self.foot_model = trimesh.load(self.foot_path)
                if self.foot_model is None:
                    raise ValueError("足部模型加载失败")
                logging.info(f"足部模型加载成功: {len(self.foot_model.vertices)} 个顶点")
            except Exception as e:
                raise ValueError(f"加载足部模型失败: {str(e)}")
            
            self.progress.emit(30)
            
            # 加载鞋内腔模型
            logging.info(f"加载鞋内腔模型: {self.shoe_path}")
            try:
                scene = trimesh.load(self.shoe_path)
                if scene is None:
                    raise ValueError("鞋内腔模型加载失败")
                
                # 检查是否为Scene对象
                if isinstance(scene, trimesh.Scene):
                    logging.info("检测到场景对象，提取所有网格...")
                    meshes = list(scene.geometry.values())
                    mesh_names = list(scene.geometry.keys())
                    
                    # 筛选网格
                    total_vertices = sum(len(mesh.vertices) for mesh in meshes)
                    total_faces = sum(len(mesh.faces) for mesh in meshes)
                    logging.info(f"\n保留 {len(meshes)} 个网格:")
                    for i, (name, mesh) in enumerate(zip(mesh_names, meshes)):
                        logging.info(f"网格 {i + 1}: 名称 = {name}, 面片数量 = {len(mesh.faces)}")
                    
                    self.shoe_model = meshes
                    self.mesh_names = mesh_names
                else:
                    self.shoe_model = [scene]
                    self.mesh_names = ["default"]
                
                logging.info(f"鞋内腔模型加载成功")
            except Exception as e:
                raise ValueError(f"加载鞋内腔模型失败: {str(e)}")
            
            self.progress.emit(50)
            
            # 读取材质信息
            logging.info("读取材质信息...")
            try:
                self.material_faces = self.read_obj_materials(self.shoe_path)
                if self.material_faces:
                    logging.info(f"成功读取 {len(self.material_faces)} 个材质")
                    for material, faces in self.material_faces.items():
                        logging.info(f"材质 '{material}': {len(faces)} 个面片")
                else:
                    logging.warning("未找到材质信息")
            except Exception as e:
                logging.warning(f"读取材质信息失败: {str(e)}")
                self.material_faces = {}
            
            self.progress.emit(100)
            self.finished.emit(True, "模型加载成功")
            logging.info("模型加载完成")
            
        except Exception as e:
            logging.error(f"模型加载失败: {str(e)}")
            logging.error(traceback.format_exc())
            self.finished.emit(False, str(e))
    
    def read_obj_materials(self, file_path):
        """读取OBJ文件中的材质信息"""
        try:
            materials = {}
            current_material = None
            face_count = 0
            material_faces = {}
            
            # 首先读取文件获取所有材质名称
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('usemtl '):
                        material = line[7:].strip()
                        # 标准化材质名称
                        if '鞋尖' in material or 'toe' in material.lower():
                            material = '鞋尖'
                        elif '鞋面' in material or 'upper' in material.lower():
                            material = '鞋面'
                        elif '鞋舌上部' in material or 'tongue_upper' in material.lower():
                            material = '鞋舌上部'
                        elif '鞋舌下部' in material or 'tongue_lower' in material.lower():
                            material = '鞋舌下部'
                        elif '后跟边缘' in material or 'heel_edge' in material.lower():
                            material = '后跟边缘'
                        elif '后跟' in material or 'heel' in material.lower():
                            material = '后跟'
                        elif '鞋底' in material or 'sole' in material.lower():
                            material = '鞋底'
                        else:
                            material = '鞋面'  # 默认材质
                            
                        if material not in material_faces:
                            material_faces[material] = []
            
            # 重新读取文件，记录每个面片的材质
            with open(file_path, 'r', encoding='utf-8') as f:
                current_material = '鞋面'  # 默认材质
                for line in f:
                    line = line.strip()
                    if line.startswith('usemtl '):
                        material = line[7:].strip()
                        # 标准化材质名称
                        if '鞋尖' in material or 'toe' in material.lower():
                            current_material = '鞋尖'
                        elif '鞋面' in material or 'upper' in material.lower():
                            current_material = '鞋面'
                        elif '鞋舌上部' in material or 'tongue_upper' in material.lower():
                            current_material = '鞋舌上部'
                        elif '鞋舌下部' in material or 'tongue_lower' in material.lower():
                            current_material = '鞋舌下部'
                        elif '后跟边缘' in material or 'heel_edge' in material.lower():
                            current_material = '后跟边缘'
                        elif '后跟' in material or 'heel' in material.lower():
                            current_material = '后跟'
                        elif '鞋底' in material or 'sole' in material.lower():
                            current_material = '鞋底'
                        else:
                            current_material = '鞋面'  # 默认材质
                    elif line.startswith('f '):
                        material_faces[current_material].append(face_count)
                        face_count += 1
            
            # 移除空的材质组
            material_faces = {k: v for k, v in material_faces.items() if v}
            
            # 打印材质信息
            logging.info(f"读取到的材质信息:")
            for material, faces in material_faces.items():
                logging.info(f"材质 '{material}': {len(faces)} 个面片")
            
            return material_faces
            
        except Exception as e:
            logging.error(f"读取材质信息时出错: {str(e)}")
            return {}

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

class FootShoeAnalysisUI(QMainWindow):
    def __init__(self):
        try:
            super().__init__()
            logging.info("初始化主窗口...")
            
            self.setWindowTitle("足鞋虚拟适配分析系统")
            self.setGeometry(100, 100, 1600, 900)
            
            # 初始化变量
            self.foot_model = None
            self.shoe_model = None
            self.simplified_shoe = None
            self.distances = None
            self.valid_mask = None
            self.material_faces = None
            self.pressures = None
            
            # 设置路径
            self.foot_model_base_path = r"C:\Users\wufan\Documents\MyFiles\3 研究院文件\项目文件\足鞋工效2024\2项目研发相关材料\3虚拟适配相关材料\虚拟适配250325\Foot_Model\Alignment"
            self.shoe_model_base_path = r"C:\Users\wufan\Documents\MyFiles\3 研究院文件\项目文件\足鞋工效2024\2项目研发相关材料\3虚拟适配相关材料\虚拟适配250325\Shoe_Model"
            
            # 创建UI
            self.init_ui()
            logging.info("主窗口初始化完成")
            
        except Exception as e:
            logging.error(f"初始化主窗口时出错: {str(e)}")
            logging.error(traceback.format_exc())
            QMessageBox.critical(None, "错误", f"初始化主窗口时出错:\n{str(e)}")
            raise

    def init_ui(self):
        try:
            logging.info("开始创建UI...")
            
            # 创建主窗口部件
            main_widget = QWidget()
            self.setCentralWidget(main_widget)
            
            # 创建主布局
            main_layout = QHBoxLayout()
            main_widget.setLayout(main_layout)
            
            # 左侧控制面板
            control_panel = QWidget()
            control_panel.setFixedWidth(300)
            control_layout = QVBoxLayout()
            control_panel.setLayout(control_layout)
            
            # 足部模型选择区域
            foot_group = QGroupBox("足部模型选择")
            foot_layout = QVBoxLayout()
            
            # 用户编号选择
            user_layout = QHBoxLayout()
            user_layout.addWidget(QLabel("用户编号:"))
            self.user_combo = QComboBox()
            self.user_combo.addItems([f"S{i}" for i in range(1, 22)])
            self.user_combo.currentTextChanged.connect(self.update_size_options)
            user_layout.addWidget(self.user_combo)
            foot_layout.addLayout(user_layout)
            
            # 尺码选择
            size_layout = QHBoxLayout()
            size_layout.addWidget(QLabel("尺码:"))
            self.size_combo = QComboBox()
            size_layout.addWidget(self.size_combo)
            foot_layout.addLayout(size_layout)
            
            # 左右脚选择
            side_layout = QHBoxLayout()
            side_layout.addWidget(QLabel("左右脚:"))
            self.side_combo = QComboBox()
            self.side_combo.addItems(["左", "右"])
            side_layout.addWidget(self.side_combo)
            foot_layout.addLayout(side_layout)
            
            foot_group.setLayout(foot_layout)
            control_layout.addWidget(foot_group)
            
            # 分析参数设置区域
            param_group = QGroupBox("分析参数设置")
            param_layout = QVBoxLayout()
            
            # 最大角度设置
            angle_layout = QHBoxLayout()
            angle_layout.addWidget(QLabel("最大角度(度):"))
            self.angle_spin = QSpinBox()
            self.angle_spin.setRange(0, 90)
            self.angle_spin.setValue(40)  # 默认值
            angle_layout.addWidget(self.angle_spin)
            param_layout.addLayout(angle_layout)
            
            # 最大距离设置
            max_dist_layout = QHBoxLayout()
            max_dist_layout.addWidget(QLabel("最大距离(mm):"))
            self.max_dist_spin = QDoubleSpinBox()
            self.max_dist_spin.setRange(0, 50)
            self.max_dist_spin.setValue(15.0)  # 默认值
            self.max_dist_spin.setDecimals(1)
            max_dist_layout.addWidget(self.max_dist_spin)
            param_layout.addLayout(max_dist_layout)
            
            # 距离范围设置
            dist_range_layout = QHBoxLayout()
            dist_range_layout.addWidget(QLabel("距离范围(mm):"))
            self.min_dist_spin = QDoubleSpinBox()
            self.min_dist_spin.setRange(-50, 0)
            self.min_dist_spin.setValue(-10.0)  # 默认值
            self.min_dist_spin.setDecimals(1)
            dist_range_layout.addWidget(self.min_dist_spin)
            dist_range_layout.addWidget(QLabel("到"))
            self.max_range_spin = QDoubleSpinBox()
            self.max_range_spin.setRange(0, 50)
            self.max_range_spin.setValue(25.0)  # 默认值
            self.max_range_spin.setDecimals(1)
            dist_range_layout.addWidget(self.max_range_spin)
            param_layout.addLayout(dist_range_layout)
            
            param_group.setLayout(param_layout)
            control_layout.addWidget(param_group)
            
            # 鞋内腔模型信息
            shoe_group = QGroupBox("鞋内腔模型信息")
            shoe_layout = QVBoxLayout()
            self.shoe_info_label = QLabel("请先选择足部模型")
            shoe_layout.addWidget(self.shoe_info_label)
            shoe_group.setLayout(shoe_layout)
            control_layout.addWidget(shoe_group)
            
            # 材质信息显示
            material_group = QGroupBox("材质信息")
            material_layout = QVBoxLayout()
            self.material_info_label = QLabel("请先加载鞋内腔模型")
            material_layout.addWidget(self.material_info_label)
            material_group.setLayout(material_layout)
            control_layout.addWidget(material_group)
            
            # 分析结果显示
            result_group = QGroupBox("分析结果")
            result_layout = QVBoxLayout()
            self.result_label = QLabel("请先执行分析")
            result_layout.addWidget(self.result_label)
            result_group.setLayout(result_layout)
            control_layout.addWidget(result_group)
            
            # 进度条
            self.progress_bar = QProgressBar()
            self.progress_bar.setVisible(False)
            control_layout.addWidget(self.progress_bar)
            
            # 按钮区域
            button_layout = QVBoxLayout()
            self.load_button = QPushButton("加载模型")
            self.load_button.clicked.connect(self.load_models)
            button_layout.addWidget(self.load_button)
            
            self.analyze_button = QPushButton("执行分析")
            self.analyze_button.clicked.connect(self.analyze_models)
            self.analyze_button.setEnabled(False)
            button_layout.addWidget(self.analyze_button)
            
            self.clear_button = QPushButton("清空显示")
            self.clear_button.clicked.connect(self.clear_display)
            button_layout.addWidget(self.clear_button)
            
            self.exit_button = QPushButton("退出")
            self.exit_button.clicked.connect(self.close)
            button_layout.addWidget(self.exit_button)
            
            control_layout.addLayout(button_layout)
            control_layout.addStretch()
            
            # 右侧显示区域
            display_panel = QWidget()
            display_layout = QVBoxLayout()
            display_panel.setLayout(display_layout)
            
            logging.info("创建VTK渲染窗口...")
            try:
                # 创建VTK部件
                self.vtk_widget = VTKWidget()
                display_layout.addWidget(self.vtk_widget)
                logging.info("VTK部件创建成功")
                
            except Exception as e:
                logging.error(f"VTK初始化失败: {str(e)}")
                logging.error(traceback.format_exc())
                raise
            
            # 添加布局到主窗口
            main_layout.addWidget(control_panel)
            main_layout.addWidget(display_panel)
            
            # 初始化用户尺码选项
            self.update_size_options()
            
            # 设置主窗口大小
            self.setMinimumSize(1200, 800)
            
            logging.info("UI创建完成")
            
        except Exception as e:
            logging.error(f"创建UI时出错: {str(e)}")
            logging.error(traceback.format_exc())
            QMessageBox.critical(None, "错误", f"创建UI时出错:\n{str(e)}")
            raise

    def showEvent(self, event):
        """重写showEvent以确保正确初始化"""
        try:
            super().showEvent(event)
            logging.info("窗口显示事件触发")
            
            if hasattr(self, 'vtk_widget'):
                logging.info("启动交互器...")
                self.vtk_widget.GetInteractor().Start()
                logging.info("交互器启动成功")
                
                # 强制更新窗口
                if hasattr(self, 'vtk_widget'):
                    logging.info("更新渲染窗口...")
                    self.vtk_widget.GetRenderWindow().Render()
                    logging.info("渲染窗口更新成功")
            
        except Exception as e:
            logging.error(f"窗口显示事件处理失败: {str(e)}")
            logging.error(traceback.format_exc())
            QMessageBox.critical(self, "错误", f"窗口显示事件处理失败:\n{str(e)}")

    def update_size_options(self):
        """更新尺码选项"""
        self.size_combo.clear()
        user = self.user_combo.currentText()
        user_path = os.path.join(self.foot_model_base_path, user)
        
        if os.path.exists(user_path):
            sizes = [d for d in os.listdir(user_path) if os.path.isdir(os.path.join(user_path, d))]
            self.size_combo.addItems(sorted(sizes))
    
    def load_models(self):
        """加载足部模型和鞋内腔模型"""
        try:
            # 获取选择的信息
            user = self.user_combo.currentText()
            size = self.size_combo.currentText()
            side = "L" if self.side_combo.currentText() == "左" else "R"
            
            # 构建足部模型路径 - 使用连字符而不是下划线
            foot_path = os.path.join(
                self.foot_model_base_path,
                user,
                size,
                f"{user}-{size}{side}.stl"
            )
            
            # 检查文件是否存在
            if not os.path.exists(foot_path):
                logging.error(f"找不到足部模型文件: {foot_path}")
                # 检查目录结构
                base_exists = os.path.exists(self.foot_model_base_path)
                user_dir = os.path.join(self.foot_model_base_path, user)
                user_exists = os.path.exists(user_dir)
                size_dir = os.path.join(user_dir, size)
                size_exists = os.path.exists(size_dir)
                
                error_msg = f"找不到足部模型文件: {foot_path}\n"
                error_msg += f"基础路径存在: {base_exists}\n"
                error_msg += f"用户目录存在: {user_exists}\n"
                error_msg += f"尺码目录存在: {size_exists}\n"
                error_msg += "\n请检查以下路径是否正确：\n"
                error_msg += f"基础路径: {self.foot_model_base_path}\n"
                error_msg += f"完整路径: {foot_path}"
                
                QMessageBox.warning(self, "错误", error_msg)
                return
            
            # 构建鞋内腔模型路径
            shoe_size = size
            if size == "40":
                shoe_size = "40M" if "M" in size else "40F"
            shoe_path = os.path.join(self.shoe_model_base_path, f"{shoe_size}{side}.obj")
            
            # 检查文件是否存在
            if not os.path.exists(shoe_path):
                logging.error(f"找不到鞋内腔模型文件: {shoe_path}")
                error_msg = f"找不到鞋内腔模型文件: {shoe_path}\n"
                error_msg += f"基础路径存在: {os.path.exists(self.shoe_model_base_path)}\n"
                error_msg += "\n请检查以下路径是否正确：\n"
                error_msg += f"基础路径: {self.shoe_model_base_path}\n"
                error_msg += f"完整路径: {shoe_path}"
                
                QMessageBox.warning(self, "错误", error_msg)
                return
            
            # 显示进度条
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # 创建加载线程
            self.loader = ModelLoader(foot_path, shoe_path)
            self.loader.progress.connect(self.update_progress)
            self.loader.finished.connect(self.load_finished)
            self.loader.start()
            
        except Exception as e:
            logging.error(f"加载模型时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"加载模型时出错: {str(e)}")
    
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
    
    def load_finished(self, success, message):
        """模型加载完成处理"""
        try:
            self.progress_bar.setVisible(False)
            
            if success:
                logging.info("开始处理加载完成的模型...")
                
                # 从加载器获取模型数据
                if not hasattr(self.loader, 'foot_model') or self.loader.foot_model is None:
                    raise ValueError("足部模型未成功加载")
                if not hasattr(self.loader, 'shoe_model') or self.loader.shoe_model is None:
                    raise ValueError("鞋内腔模型未成功加载")
                
                # 复制模型数据
                self.foot_model = self.loader.foot_model
                self.shoe_model = self.loader.shoe_model
                self.mesh_names = self.loader.mesh_names
                
                # 从加载器获取材质信息
                logging.info("获取材质信息...")
                self.material_faces = self.loader.material_faces
                if self.material_faces:
                    logging.info(f"成功获取 {len(self.material_faces)} 个材质")
                    for material, faces in self.material_faces.items():
                        logging.info(f"材质 '{material}': {len(faces)} 个面片")
                else:
                    logging.warning("未找到材质信息")
                
                # 验证模型数据
                if len(self.foot_model.vertices) == 0:
                    raise ValueError("足部模型没有顶点数据")
                
                # 验证鞋内腔模型数据
                total_vertices = sum(len(mesh.vertices) for mesh in self.shoe_model)
                total_faces = sum(len(mesh.faces) for mesh in self.shoe_model)
                if total_vertices == 0:
                    raise ValueError("鞋内腔模型没有顶点数据")
                
                logging.info(f"模型数据验证成功 - 足部模型: {len(self.foot_model.vertices)} 个顶点, " +
                           f"鞋内腔模型: {total_vertices} 个顶点, {total_faces} 个面片")
                
                # 更新UI显示
                self.shoe_info_label.setText(f"已加载鞋内腔模型: {os.path.basename(self.loader.shoe_path)}")
                self.update_material_info()
                
                # 显示模型
                self.display_models()
                
                # 启用分析按钮
                self.analyze_button.setEnabled(True)
                
                logging.info(message)
                QMessageBox.information(self, "成功", "模型加载成功！")
            else:
                QMessageBox.critical(self, "错误", message)
                logging.error(message)
                self.analyze_button.setEnabled(False)
                
                # 清理模型数据
                self.foot_model = None
                self.shoe_model = None
                self.material_faces = None
        except Exception as e:
            error_msg = f"处理加载完成事件时出错: {str(e)}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            QMessageBox.critical(self, "错误", error_msg)
            self.analyze_button.setEnabled(False)
            
            # 清理模型数据
            self.foot_model = None
            self.shoe_model = None
            self.material_faces = None
    
    def update_material_info(self):
        """更新材质信息显示"""
        if self.material_faces:
            info_text = "找到以下材质:\n"
            for material, faces in self.material_faces.items():
                info_text += f"{material}: {len(faces)} 个面片\n"
            self.material_info_label.setText(info_text)
    
    def display_models(self):
        """显示模型"""
        try:
            # 清空当前显示
            self.vtk_widget.GetRenderer().RemoveAllViewProps()
            
            # 显示足部模型
            if self.foot_model:
                self.display_model(self.foot_model, (0.8, 0.8, 0.8))
            
            # 显示鞋内腔模型
            if self.shoe_model:
                self.display_model_with_materials(self.shoe_model)
            
            # 重置相机
            self.vtk_widget.GetRenderer().ResetCamera()
            camera = self.vtk_widget.GetRenderer().GetActiveCamera()
            camera.Elevation(30)
            camera.Azimuth(30)
            
            self.vtk_widget.GetRenderWindow().Render()
            
            logging.info("模型显示完成")
            
        except Exception as e:
            logging.error(f"显示模型时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"显示模型时出错: {str(e)}")
    
    def display_model(self, model, color):
        """显示单个模型"""
        try:
            # 创建VTK点数据
            points = vtk.vtkPoints()
            for vertex in model.vertices:
                points.InsertNextPoint(vertex)
            
            # 创建VTK面片数据
            cells = vtk.vtkCellArray()
            for face in model.faces:
                cell = vtk.vtkTriangle()
                cell.GetPointIds().SetId(0, face[0])
                cell.GetPointIds().SetId(1, face[1])
                cell.GetPointIds().SetId(2, face[2])
                cells.InsertNextCell(cell)
            
            # 创建PolyData
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(points)
            polydata.SetPolys(cells)
            
            # 创建映射器
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polydata)
            
            # 创建actor
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(color)
            
            # 设置材质属性
            prop = actor.GetProperty()
            prop.SetAmbient(0.3)
            prop.SetDiffuse(0.7)
            prop.SetSpecular(0.5)
            prop.SetSpecularPower(20)
            
            # 设置透明度
            prop.SetOpacity(0.3)  # 设置透明度为0.3（70%透明）
            
            # 添加到渲染器
            self.vtk_widget.GetRenderer().AddActor(actor)
            
        except Exception as e:
            logging.error(f"显示单个模型时出错: {str(e)}")
            raise
    
    def display_model_with_materials(self, meshes):
        """显示带有材质的模型"""
        try:
            if not isinstance(meshes, list):
                meshes = [meshes]
            
            # 定义颜色列表
            colors = [
                (1, 0, 0),    # 红色
                (0, 1, 0),    # 绿色
                (0, 0, 1),    # 蓝色
                (1, 1, 0),    # 黄色
                (1, 0, 1),    # 紫色
                (0, 1, 1),    # 青色
                (1, 0.5, 0),  # 橙色
                (0.5, 1, 0),  # 黄绿色
                (0, 0.5, 1),  # 浅蓝色
                (1, 0, 0.5),  # 粉红色
                (0.5, 0.5, 0),# 棕色
                (0.5, 0, 0.5),# 深紫色
                (0, 0.5, 0.5) # 深青色
            ]
            
            # 为每个网格分配一个颜色
            for i, mesh in enumerate(meshes):
                color = colors[i % len(colors)]
                logging.info(f"网格 {i + 1} 使用颜色: {color}, 面片数量: {len(mesh.faces)}")
                
                # 创建VTK点数据
                points = vtk.vtkPoints()
                for vertex in mesh.vertices:
                    points.InsertNextPoint(vertex)
                
                # 创建VTK面片数据
                cells = vtk.vtkCellArray()
                for face in mesh.faces:
                    cell = vtk.vtkTriangle()
                    cell.GetPointIds().SetId(0, face[0])
                    cell.GetPointIds().SetId(1, face[1])
                    cell.GetPointIds().SetId(2, face[2])
                    cells.InsertNextCell(cell)
                
                # 创建PolyData
                polydata = vtk.vtkPolyData()
                polydata.SetPoints(points)
                polydata.SetPolys(cells)
                
                # 创建颜色数组
                colors_array = vtk.vtkUnsignedCharArray()
                colors_array.SetName("Colors")
                colors_array.SetNumberOfComponents(3)
                colors_array.SetNumberOfTuples(len(mesh.faces))
                
                # 设置网格颜色
                for j in range(len(mesh.faces)):
                    colors_array.SetTuple3(j, *[int(c * 255) for c in color])
                
                polydata.GetCellData().SetScalars(colors_array)
                
                # 创建映射器
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(polydata)
                
                # 创建actor
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                
                # 设置材质属性
                prop = actor.GetProperty()
                prop.SetAmbient(0.3)
                prop.SetDiffuse(0.7)
                prop.SetSpecular(0.5)
                prop.SetSpecularPower(20)
                
                # 添加到渲染器
                self.vtk_widget.GetRenderer().AddActor(actor)
                
        except Exception as e:
            logging.error(f"显示带材质模型时出错: {str(e)}")
            raise
    
    def analyze_models(self):
        """执行模型分析"""
        try:
            if not hasattr(self, 'foot_model') or not hasattr(self, 'shoe_model'):
                QMessageBox.warning(self, "警告", "请先加载模型")
                return
            
            logging.info("开始执行模型分析...")
            
            # 获取分析参数
            max_angle = self.angle_spin.value()
            max_distance = self.max_dist_spin.value()
            min_distance = self.min_dist_spin.value()
            max_range = self.max_range_spin.value()
            
            logging.info(f"分析参数: 最大角度={max_angle}度, 最大距离={max_distance}mm, " +
                        f"距离范围=[{min_distance}, {max_range}]mm")
            
            # 获取鞋模型的分组信息
            groups = {}
            if isinstance(self.shoe_model, trimesh.Scene):
                face_offset = 0
                for name, geometry in self.shoe_model.geometry.items():
                    face_indices = list(range(face_offset, face_offset + len(geometry.faces)))
                    groups[name] = {
                        'faces': face_indices,
                        'material': None,
                        'object': name
                    }
                    face_offset += len(geometry.faces)
            
            # 计算分区空间信息
            logging.info("计算分区空间信息...")
            group_regions = self.compute_group_regions(self.shoe_model, groups)
            
            # 计算距离
            logging.info("计算距离...")
            distances, valid_mask = self.compute_distances(
                self.shoe_model, 
                self.foot_model,
                max_angle=max_angle,
                max_distance=max_distance,
                min_distance=min_distance,
                max_range=max_range
            )
            
            # 显示统计信息
            valid_distances = distances[valid_mask]
            logging.info(f"总面片数: {len(distances)}")
            logging.info(f"有效面片数: {np.sum(valid_mask)}")
            if len(valid_distances) > 0:
                logging.info(f"距离范围: {np.min(valid_distances):.2f}mm 到 {np.max(valid_distances):.2f}mm")
            
            # 创建新窗口显示分析结果
            analysis_window = QMainWindow(self)
            analysis_window.setWindowTitle("分析结果")
            analysis_window.resize(1600, 800)
            
            # 创建VTK部件
            vtk_widget = QVTKRenderWindowInteractor()
            analysis_window.setCentralWidget(vtk_widget)
            
            # 获取渲染器和窗口
            renderer = vtk.vtkRenderer()
            window = vtk_widget.GetRenderWindow()
            window.AddRenderer(renderer)
            
            # 设置背景颜色为白色
            renderer.SetBackground(1, 1, 1)
            
            # 显示距离云图和力值分布
            logging.info("显示距离云图和力值分布...")
            self.display_model_with_distances_and_pressure(
                self.foot_model, 
                distances, 
                valid_mask, 
                renderer,
                self.shoe_model,
                group_regions
            )
            
            # 设置相机位置
            renderer.ResetCamera()
            camera = renderer.GetActiveCamera()
            camera.Elevation(20)
            camera.Azimuth(20)
            
            # 初始化交互器
            vtk_widget.Initialize()
            vtk_widget.Start()
            
            # 显示窗口
            analysis_window.show()
            
            logging.info("分析完成")
            
        except Exception as e:
            logging.error(f"分析过程出错: {str(e)}")
            logging.error(traceback.format_exc())
            QMessageBox.critical(self, "错误", f"分析过程出错: {str(e)}")

    def compute_group_regions(self, mesh, groups):
        """计算每个分区的空间区域信息
        Args:
            mesh: 原始网格模型
            groups: 分区信息字典
        Returns:
            group_regions: 包含每个分区空间信息的字典
        """
        group_regions = {}
        
        if isinstance(mesh, trimesh.Scene):
            for group_name, geometry in mesh.geometry.items():
                # 获取该分区所有面片的顶点
                vertices = geometry.vertices
                
                # 计算边界框
                min_bounds = np.min(vertices, axis=0)
                max_bounds = np.max(vertices, axis=0)
                center = (min_bounds + max_bounds) / 2
                
                material_type = self.get_material_type(group_name)
                
                group_regions[group_name] = {
                    'center': center,
                    'min_bounds': min_bounds,
                    'max_bounds': max_bounds,
                    'material': material_type,
                    'object': group_name
                }
        
        return group_regions

    def get_material_type(self, group_name):
        """根据组名确定材质类型"""
        group_name = group_name.lower()
        
        # 鞋尖
        if '鞋尖' in group_name or 'toe' in group_name:
            return '试样2'
        
        # 鞋面（1-5）
        if '鞋面' in group_name or 'upper' in group_name:
            return '试样1'
        
        # 鞋舌上部
        if '鞋舌上部' in group_name or 'tongue_upper' in group_name:
            return '鞋舌上部'
        
        # 鞋舌下部
        if '鞋舌下部' in group_name or 'tongue_lower' in group_name:
            return '试样4'
        
        # 后跟（1-3）
        if '后跟' in group_name and '边缘' not in group_name:
            return '后跟上部'
        
        # 后跟边缘（使用鞋舌上部的材质）
        if '后跟边缘' in group_name:
            return '鞋舌上部'
        
        # 鞋底
        if '鞋底' in group_name or 'sole' in group_name:
            return '鞋底'
        
        # 默认返回鞋面材质
        return '试样1'

    def display_model_with_distances_and_pressure(self, model, distances, valid_mask, renderer, shoe_model, group_regions):
        """显示距离云图和力值分布图"""
        # 创建两个子视口
        viewport1 = [0, 0, 0.5, 1.0]  # 左半边：距离云图
        viewport2 = [0.5, 0, 1.0, 1.0]  # 右半边：力值云图
        
        # 设置第一个渲染器（距离云图）
        renderer.SetViewport(*viewport1)
        
        # 创建第二个渲染器（力值云图）
        renderer2 = vtk.vtkRenderer()
        renderer2.SetViewport(*viewport2)
        renderer2.SetBackground(1, 1, 1)
        renderer.GetRenderWindow().AddRenderer(renderer2)
        
        # 添加标题文本
        title_text1 = vtk.vtkTextActor()
        title_text1.SetInput("Distance Map / 距离云图")
        title_text1.GetTextProperty().SetFontSize(24)
        title_text1.GetTextProperty().SetBold(True)
        title_text1.GetTextProperty().SetColor(0, 0, 0)
        title_text1.SetPosition(20, 550)
        renderer.AddActor2D(title_text1)
        
        title_text2 = vtk.vtkTextActor()
        title_text2.SetInput("Force Map / 力值云图")
        title_text2.GetTextProperty().SetFontSize(24)
        title_text2.GetTextProperty().SetBold(True)
        title_text2.GetTextProperty().SetColor(0, 0, 0)
        title_text2.SetPosition(20, 550)
        renderer2.AddActor2D(title_text2)
        
        # 创建基础几何数据
        points = vtk.vtkPoints()
        for vertex in model.vertices:
            points.InsertNextPoint(vertex)
        
        cells = vtk.vtkCellArray()
        for face in model.faces:
            cell = vtk.vtkTriangle()
            cell.GetPointIds().SetId(0, face[0])
            cell.GetPointIds().SetId(1, face[1])
            cell.GetPointIds().SetId(2, face[2])
            cells.InsertNextCell(cell)
        
        # 创建PolyData
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(cells)
        
        # 添加距离数据
        distance_data = vtk.vtkFloatArray()
        distance_data.SetName("Distance")
        
        # 添加力值数据
        force_data = vtk.vtkFloatArray()
        force_data.SetName("Force")
        
        # 将面的数据插值到顶点
        vertex_distances = np.full(len(model.vertices), np.nan)
        vertex_forces = np.full(len(model.vertices), np.nan)
        vertex_counts = np.zeros(len(model.vertices))
        
        # 计算每个面片的力值
        force_values = []
        for face_idx, (face, dist, valid) in enumerate(zip(model.faces, distances, valid_mask)):
            if valid:
                # 计算面片中心点
                face_vertices = model.vertices[face]
                face_center = np.mean(face_vertices, axis=0)
                
                # 根据面片位置分配材质
                material_type = self.assign_material_to_face(face_center, group_regions)
                
                # 计算力值
                force = self.convert_distance_to_pressure(dist, material_type)
                force_values.append(force)
                
                # 将力值分配给面片的三个顶点
                for vertex_idx in face:
                    if np.isnan(vertex_distances[vertex_idx]):
                        vertex_distances[vertex_idx] = dist
                        vertex_forces[vertex_idx] = force
                        vertex_counts[vertex_idx] = 1
                    else:
                        vertex_distances[vertex_idx] += dist
                        vertex_forces[vertex_idx] += force
                        vertex_counts[vertex_idx] += 1
        
        # 计算平均值
        valid_vertices = vertex_counts > 0
        vertex_distances[valid_vertices] /= vertex_counts[valid_vertices]
        vertex_forces[valid_vertices] /= vertex_counts[valid_vertices]
        
        # 添加到VTK数组
        for dist, force in zip(vertex_distances, vertex_forces):
            distance_data.InsertNextValue(float(dist))
            force_data.InsertNextValue(float(force))
        
        # 设置距离数据
        polydata.GetPointData().SetScalars(distance_data)
        
        # 创建力值数据的PolyData
        force_polydata = vtk.vtkPolyData()
        force_polydata.DeepCopy(polydata)
        force_polydata.GetPointData().SetScalars(force_data)
        
        # 设置距离范围
        min_dist = -12.5
        max_dist = 25.0
        
        # 设置力值范围
        max_force = np.nanmax(vertex_forces)
        if np.isnan(max_force) or max_force <= 0:
            max_force = 100.0
        
        # 创建映射器
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        mapper.SetScalarRange(min_dist, max_dist)
        
        force_mapper = vtk.vtkPolyDataMapper()
        force_mapper.SetInputData(force_polydata)
        force_mapper.SetScalarRange(0, max_force)
        
        # 创建颜色表
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(512)
        lut.SetTableRange(min_dist, max_dist)
        
        # 设置jet颜色映射
        for i in range(512):
            t = i / 511.0
            if t < 0.125:
                r = 0
                g = 0
                b = 0.5 + 4*t
            elif t < 0.375:
                r = 0
                g = 4*(t - 0.125)
                b = 1
            elif t < 0.625:
                r = 4*(t - 0.375)
                g = 1
                b = 1 - 4*(t - 0.375)
            elif t < 0.875:
                r = 1
                g = 1 - 4*(t - 0.625)
                b = 0
            else:
                r = 1 - 4*(t - 0.875)
                g = 0
                b = 0
            
            r = max(0.0, min(1.0, r))
            g = max(0.0, min(1.0, g))
            b = max(0.0, min(1.0, b))
            
            lut.SetTableValue(i, r, g, b, 1.0)
        
        # 创建力值颜色表
        force_lut = vtk.vtkLookupTable()
        force_lut.SetNumberOfTableValues(512)
        force_lut.SetTableRange(0, max_force)
        
        for i in range(512):
            t = i / 511.0
            if t < 0.25:
                r = 0.85 - t * 2
                g = 0.85 - t * 2
                b = 0.85
            elif t < 0.5:
                r = 0.35 - (t - 0.25) * 1.4
                g = 0.35 + (t - 0.25) * 1.4
                b = 0.85 - (t - 0.25) * 1.4
            elif t < 0.75:
                r = 0.35 + (t - 0.5) * 2.6
                g = 0.7 - (t - 0.5) * 1.4
                b = 0.15
            else:
                r = 1.0
                g = 0.35 - (t - 0.75) * 1.4
                b = 0.15
            
            force_lut.SetTableValue(i, r, g, b, 1.0)
        
        # 设置NaN值的颜色
        lut.SetNanColor(0.9, 0.9, 0.9, 1.0)
        force_lut.SetNanColor(0.9, 0.9, 0.9, 1.0)
        
        lut.Build()
        force_lut.Build()
        
        # 设置映射器使用颜色表
        mapper.SetLookupTable(lut)
        mapper.SetUseLookupTableScalarRange(True)
        
        force_mapper.SetLookupTable(force_lut)
        force_mapper.SetUseLookupTableScalarRange(True)
        
        # 创建Actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        force_actor = vtk.vtkActor()
        force_actor.SetMapper(force_mapper)
        
        # 添加颜色条
        scalar_bar = vtk.vtkScalarBarActor()
        scalar_bar.SetLookupTable(mapper.GetLookupTable())
        scalar_bar.SetTitle("距离 (mm)")
        scalar_bar.SetWidth(0.08)
        scalar_bar.SetHeight(0.6)
        scalar_bar.SetPosition(0.92, 0.2)
        scalar_bar.SetNumberOfLabels(5)
        scalar_bar.GetLabelTextProperty().SetColor(0, 0, 0)
        scalar_bar.GetTitleTextProperty().SetColor(0, 0, 0)
        
        force_scalar_bar = vtk.vtkScalarBarActor()
        force_scalar_bar.SetLookupTable(force_mapper.GetLookupTable())
        force_scalar_bar.SetTitle("压力值 (N)")
        force_scalar_bar.SetWidth(0.08)
        force_scalar_bar.SetHeight(0.6)
        force_scalar_bar.SetPosition(0.92, 0.2)
        force_scalar_bar.SetNumberOfLabels(5)
        force_scalar_bar.GetLabelTextProperty().SetColor(0, 0, 0)
        force_scalar_bar.GetTitleTextProperty().SetColor(0, 0, 0)
        
        # 添加到渲染器
        renderer.AddActor(actor)
        renderer.AddActor2D(scalar_bar)
        
        renderer2.AddActor(force_actor)
        renderer2.AddActor2D(force_scalar_bar)
        
        # 创建文本actor用于显示距离和力值
        text_actor = vtk.vtkTextActor()
        text_actor.GetTextProperty().SetFontSize(20)
        text_actor.GetTextProperty().SetBold(True)
        text_actor.GetTextProperty().SetColor(0, 0, 0)
        text_actor.SetPosition(10, 10)
        renderer.AddActor2D(text_actor)
        
        force_text_actor = vtk.vtkTextActor()
        force_text_actor.GetTextProperty().SetFontSize(20)
        force_text_actor.GetTextProperty().SetBold(True)
        force_text_actor.GetTextProperty().SetColor(0, 0, 0)
        force_text_actor.SetPosition(10, 10)
        renderer2.AddActor2D(force_text_actor)
        
        # 创建拾取器
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.001)
        
        force_picker = vtk.vtkCellPicker()
        force_picker.SetTolerance(0.001)
        
        # 创建鼠标移动回调函数
        def mouse_move_callback(obj, event):
            try:
                x, y = obj.GetEventPosition()
                
                # 获取当前活动的渲染器
                clicked_renderer = obj.FindPokedRenderer(x, y)
                
                if clicked_renderer == renderer:
                    # 距离视图
                    picker.Pick(x, y, 0, renderer)
                    if picker.GetCellId() != -1:
                        cell_id = picker.GetCellId()
                        cell = polydata.GetCell(cell_id)
                        point_ids = [cell.GetPointId(i) for i in range(3)]
                        point_distances = [distance_data.GetValue(pid) for pid in point_ids]
                        distance_value = np.mean(point_distances)
                        
                        if not np.isnan(distance_value):
                            text_actor.SetInput(f"Distance/距离值: {distance_value:.2f} mm")
                        else:
                            text_actor.SetInput("Distance/距离值: N/A")
                    else:
                        text_actor.SetInput("")
                    
                elif clicked_renderer == renderer2:
                    # 力值视图
                    force_picker.Pick(x, y, 0, renderer2)
                    if force_picker.GetCellId() != -1:
                        cell_id = force_picker.GetCellId()
                        cell = force_polydata.GetCell(cell_id)
                        point_ids = [cell.GetPointId(i) for i in range(3)]
                        point_forces = [force_data.GetValue(pid) for pid in point_ids]
                        force_value = np.mean(point_forces)
                        
                        if not np.isnan(force_value):
                            force_text_actor.SetInput(f"Pressure/压力值: {force_value:.2f} N")
                        else:
                            force_text_actor.SetInput("Pressure/压力值: N/A")
                    else:
                        force_text_actor.SetInput("")
                
                renderer.GetRenderWindow().Render()
                
            except Exception as e:
                logging.error(f"鼠标回调函数出错: {str(e)}")
                text_actor.SetInput("")
                force_text_actor.SetInput("")
                renderer.GetRenderWindow().Render()
        
        # 添加事件观察者
        interactor = renderer.GetRenderWindow().GetInteractor()
        interactor.AddObserver("MouseMoveEvent", mouse_move_callback)
        
        # 同步两个视图的相机
        renderer2.SetActiveCamera(renderer.GetActiveCamera())

    def assign_material_to_face(self, face_center, group_regions):
        """根据面片中心点分配材质类型"""
        min_dist = float('inf')
        assigned_material = '无限伸缩'
        
        for group_name, region in group_regions.items():
            # 检查是否在边界框内或附近
            min_bounds = region['min_bounds']
            max_bounds = region['max_bounds']
            
            # 计算点到边界框的距离
            dx = max(min_bounds[0] - face_center[0], 0, face_center[0] - max_bounds[0])
            dy = max(min_bounds[1] - face_center[1], 0, face_center[1] - max_bounds[1])
            dz = max(min_bounds[2] - face_center[2], 0, face_center[2] - max_bounds[2])
            dist = np.sqrt(dx*dx + dy*dy + dz*dz)
            
            if dist < min_dist:
                min_dist = dist
                assigned_material = self.get_material_type(group_name)
        
        return assigned_material

    def convert_distance_to_pressure(self, distance, material_type):
        """根据不同材质的拟合方程将干涉距离转换为力值"""
        if distance <= 0:  # 没有压缩，力为0
            return 0.0
        
        # 正的干涉距离就是压缩量
        x = distance
        
        # 根据不同材质使用对应的拟合方程计算力值(N)
        if material_type == '试样1':  # 鞋面
            force = -0.0089 * x**4 + 0.4375 * x**3 - 3.0884 * x**2 + 8.2764 * x - 3.3784
        elif material_type == '试样2':  # 鞋尖
            force = -0.0122 * x**4 + 0.4174 * x**3 + 1.9400 * x**2 - 11.9956 * x + 8.7726
        elif material_type == '试样4':  # 鞋舌下部
            force = -0.0107 * x**4 + 0.4648 * x**3 - 1.5420 * x**2 - 2.8860 * x + 5.6783
        elif material_type == '后跟上部':  # 后跟
            force = -0.0067 * x**4 + 0.4720 * x**3 - 6.3526 * x**2 + 27.9414 * x - 20.6310
        elif material_type == '鞋舌上部':  # 鞋舌上部和后跟边缘
            force = -0.0107 * x**4 + 0.4648 * x**3 - 1.5420 * x**2 - 2.8860 * x + 5.6783
        elif material_type == '鞋底':
            force = -0.0089 * x**4 + 0.4375 * x**3 - 3.0884 * x**2 + 8.2764 * x - 3.3784  # 暂用鞋面的方程
        else:
            force = -0.0089 * x**4 + 0.4375 * x**3 - 3.0884 * x**2 + 8.2764 * x - 3.3784  # 默认使用鞋面的方程
        
        return max(0.0, force)  # 确保力值不为负值
    
    def clear_display(self):
        """清空显示"""
        try:
            self.vtk_widget.GetRenderer().RemoveAllViewProps()
            self.vtk_widget.GetRenderWindow().Render()
            
            # 重置模型数据
            self.foot_model = None
            self.shoe_model = None
            self.simplified_shoe = None
            self.distances = None
            self.valid_mask = None
            self.material_faces = None
            self.pressures = None
            
            # 重置UI显示
            self.shoe_info_label.setText("请先选择足部模型")
            self.material_info_label.setText("请先加载鞋内腔模型")
            self.result_label.setText("请先执行分析")
            
            # 禁用分析按钮
            self.analyze_button.setEnabled(False)
            
            logging.info("显示已清空")
            
        except Exception as e:
            logging.error(f"清空显示时出错: {str(e)}")
            QMessageBox.critical(self, "错误", f"清空显示时出错: {str(e)}")

    def closeEvent(self, event):
        """处理窗口关闭事件"""
        try:
            reply = QMessageBox.question(
                self, '确认退出',
                "确定要退出程序吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                logging.info("开始清理VTK资源...")
                
                # 停止所有渲染器
                if hasattr(self, 'vtk_widget'):
                    renderer = self.vtk_widget.GetRenderer()
                    if renderer:
                        renderer.RemoveAllViewProps()
                
                # 停止交互器
                if hasattr(self, 'vtk_widget'):
                    interactor = self.vtk_widget.GetInteractor()
                    if interactor and interactor.GetInitialized():
                        interactor.TerminateApp()
                
                # 关闭渲染窗口
                if hasattr(self, 'vtk_widget'):
                    render_window = self.vtk_widget.GetRenderWindow()
                    if render_window:
                        render_window.Finalize()
                        self.vtk_widget.close()
                
                logging.info("VTK资源清理完成")
                event.accept()
            else:
                event.ignore()
                
        except Exception as e:
            logging.error(f"处理关闭事件时出错: {str(e)}")
            logging.error(traceback.format_exc())
            event.accept()

    def compute_distances(self, shoe_mesh, foot_mesh, max_angle=40, max_distance=15.0, min_distance=-10.0, max_range=25.0):
        """计算从鞋内腔到足部模型的距离
        Args:
            shoe_mesh: 鞋内腔模型（可以是网格列表或单个网格）
            foot_mesh: 足部模型
            max_angle: 最大允许角度（度）
            max_distance: 最大距离阈值(mm)
            min_distance: 最小距离阈值(mm)
            max_range: 最大距离范围(mm)
        Returns:
            distances: 距离数组（正值表示间隙，负值表示干涉）
            valid_mask: 有效距离的掩码
        """
        # 初始化结果数组
        distances = np.full(len(foot_mesh.faces), np.nan)
        valid_mask = np.zeros(len(foot_mesh.faces), dtype=bool)
        
        # 计算足部模型面的中心点和法向量
        foot_face_centers = np.mean(foot_mesh.vertices[foot_mesh.faces], axis=1)
        foot_face_normals = foot_mesh.face_normals
        
        # 合并所有鞋内腔网格的面片
        if isinstance(shoe_mesh, list):
            shoe_vertices = []
            shoe_faces = []
            face_offset = 0
            for mesh in shoe_mesh:
                shoe_vertices.append(mesh.vertices)
                shoe_faces.append(mesh.faces + face_offset)
                face_offset += len(mesh.vertices)
            shoe_vertices = np.vstack(shoe_vertices)
            shoe_faces = np.vstack(shoe_faces)
        else:
            shoe_vertices = shoe_mesh.vertices
            shoe_faces = shoe_mesh.faces
        
        # 计算鞋内腔模型面的中心点和法向量
        shoe_face_centers = np.mean(shoe_vertices[shoe_faces], axis=1)
        # 计算法向量并确保指向内部
        shoe_face_normals = np.cross(
            shoe_vertices[shoe_faces[:, 1]] - shoe_vertices[shoe_faces[:, 0]],
            shoe_vertices[shoe_faces[:, 2]] - shoe_vertices[shoe_faces[:, 0]]
        )
        norms = np.linalg.norm(shoe_face_normals, axis=1)
        valid_norms = norms > 1e-10
        shoe_face_normals[valid_norms] = shoe_face_normals[valid_norms] / norms[valid_norms, np.newaxis]
        shoe_face_normals = -shoe_face_normals  # 确保法向量指向内部
        
        # 构建KD树
        shoe_tree = cKDTree(shoe_face_centers)
        
        # 设置参数
        max_angle_rad = np.radians(max_angle)
        
        # 对每个足部模型面进行处理
        positive_distances = []  # 用于收集正的距离值
        for i, (center, normal) in enumerate(zip(foot_face_centers, foot_face_normals)):
            # 查找最近的k个鞋内腔面
            dists, indices = shoe_tree.query(center, k=1)  # 只查找最近的一个面
            
            dist = dists
            idx = indices
            
            if dist > max_distance:
                continue
            
            # 计算方向向量
            direction = shoe_face_centers[idx] - center
            direction_norm = np.linalg.norm(direction)
            
            if direction_norm < 1e-10:
                continue
            
            direction = direction / direction_norm
            
            # 计算法向量夹角
            shoe_normal = shoe_face_normals[idx]
            angle = np.arccos(np.abs(np.dot(normal, shoe_normal)))
            
            # 如果夹角超过阈值，跳过这个点
            if angle > max_angle_rad:
                continue
            
            # 计算有符号距离
            signed_dist = dist
            if np.dot(direction, shoe_face_normals[idx]) > 0:
                signed_dist = -dist  # 干涉情况
            
            distances[i] = signed_dist
            valid_mask[i] = True
            
            # 收集正的距离值（表示压缩）
            if signed_dist > 0:
                positive_distances.append(signed_dist)
        
        # 将超出范围的距离设置为NaN
        distances[distances > max_range] = np.nan  # 大于max_range的设为NaN
        distances[distances < min_distance] = np.nan  # 小于min_distance的区域设为NaN
        
        # 更新有效掩码
        valid_mask = ~np.isnan(distances)
        
        # 打印有效范围内的统计信息
        valid_distances = distances[valid_mask]
        print(f"\n距离计算统计信息:")
        print(f"总面片数: {len(distances)}")
        print(f"有效面片数: {np.sum(valid_mask)}")
        if len(valid_distances) > 0:
            print(f"距离范围: {np.min(valid_distances):.2f}mm 到 {np.max(valid_distances):.2f}mm")
            print(f"正距离值数量（压缩）: {len(positive_distances)}")
            if positive_distances:
                print(f"压缩距离范围: {min(positive_distances):.2f}mm 到 {max(positive_distances):.2f}mm")
                print(f"平均压缩距离: {np.mean(positive_distances):.2f}mm")
        
        return distances, valid_mask

if __name__ == "__main__":
    try:
        logging.info("启动程序...")
        
        # 创建QApplication实例
        app = QApplication(sys.argv)
        logging.info("QApplication创建成功")
        
        # 创建主窗口
        window = FootShoeAnalysisUI()
        logging.info("主窗口创建成功")
        
        # 显示窗口
        window.show()
        logging.info("窗口显示成功")
        
        # 强制更新窗口
        window.update()
        logging.info("窗口更新成功")
        
        logging.info("程序启动完成，进入事件循环")
        
        # 启动事件循环
        sys.exit(app.exec_())
        
    except Exception as e:
        logging.error(f"程序启动失败: {str(e)}")
        logging.error(traceback.format_exc())
        QMessageBox.critical(None, "错误", f"程序启动失败:\n{str(e)}")
        sys.exit(1) 