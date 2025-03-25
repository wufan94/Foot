import sys
import os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import logging
from main import VirtualFitting
from config import DEFAULT_MATERIALS

class VTKWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 创建VTK部件
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget)
        
        # 创建渲染器
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.9, 0.9, 0.9)
        
        # 获取渲染窗口和交互器
        self.render_window = self.vtk_widget.GetRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.interactor = self.render_window.GetInteractor()
        
        # 设置交互样式
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)
        
        # 初始化
        self.interactor.Initialize()
        
    def closeEvent(self, event):
        """处理VTK部件关闭事件"""
        if self.interactor:
            self.interactor.TerminateApp()
        event.accept()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("虚拟试穿分析系统")
        self.setGeometry(100, 100, 1600, 900)
        
        # 初始化虚拟试穿对象 (移到最前面)
        self.fitting = VirtualFitting()
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # 创建左侧控制面板
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, stretch=2)
        
        # 创建右侧显示区域
        display_panel = self.create_display_panel()
        main_layout.addWidget(display_panel, stretch=8)
        
        # 设置日志处理
        self.setup_logging()
        
    def create_control_panel(self):
        """创建左侧控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 文件选择区域
        file_group = QGroupBox("文件选择")
        file_layout = QVBoxLayout()
        
        # 足部模型选择
        self.foot_label = QLabel("未选择足部模型")
        foot_btn = QPushButton("选择足部模型")
        foot_btn.clicked.connect(self.select_foot_model)
        
        # 鞋楦模型选择
        self.shoe_label = QLabel("未选择鞋楦模型")
        shoe_btn = QPushButton("选择鞋楦模型")
        shoe_btn.clicked.connect(self.select_shoe_model)
        
        file_layout.addWidget(self.foot_label)
        file_layout.addWidget(foot_btn)
        file_layout.addWidget(self.shoe_label)
        file_layout.addWidget(shoe_btn)
        file_group.setLayout(file_layout)
        
        # 材质参数设置区域
        material_group = QGroupBox("材质参数")
        material_layout = QGridLayout()
        
        self.material_inputs = {}
        row = 0
        for material, props in DEFAULT_MATERIALS.items():
            # 材质名称
            material_layout.addWidget(QLabel(material), row, 0)
            
            # 杨氏模量输入
            e_input = QLineEdit()
            e_input.setPlaceholderText(str(props['E']))
            material_layout.addWidget(QLabel("E (MPa):"), row, 1)
            material_layout.addWidget(e_input, row, 2)
            
            # 泊松比输入
            v_input = QLineEdit()
            v_input.setPlaceholderText(str(props['v']))
            material_layout.addWidget(QLabel("v:"), row, 3)
            material_layout.addWidget(v_input, row, 4)
            
            self.material_inputs[material] = {'E': e_input, 'v': v_input}
            row += 1
            
        material_group.setLayout(material_layout)
        
        # 影响因子设置
        factor_group = QGroupBox("影响因子")
        factor_layout = QVBoxLayout()
        
        # 材质影响因子
        material_factor_layout = QHBoxLayout()
        material_factor_layout.addWidget(QLabel("材质影响:"))
        self.material_factor_slider = QSlider(Qt.Horizontal)
        self.material_factor_slider.setRange(0, 100)
        self.material_factor_slider.setValue(50)
        material_factor_layout.addWidget(self.material_factor_slider)
        
        # PPT影响因子
        ppt_factor_layout = QHBoxLayout()
        ppt_factor_layout.addWidget(QLabel("PPT影响:"))
        self.ppt_factor_slider = QSlider(Qt.Horizontal)
        self.ppt_factor_slider.setRange(0, 100)
        self.ppt_factor_slider.setValue(30)
        ppt_factor_layout.addWidget(self.ppt_factor_slider)
        
        factor_layout.addLayout(material_factor_layout)
        factor_layout.addLayout(ppt_factor_layout)
        factor_group.setLayout(factor_layout)
        
        # 运行按钮
        run_btn = QPushButton("运行分析")
        run_btn.clicked.connect(self.run_analysis)
        
        # 添加退出按钮
        exit_btn = QPushButton("安全退出")
        exit_btn.clicked.connect(self.close_application)
        exit_btn.setStyleSheet("""
            QPushButton {
                background-color: #d9534f;
                color: white;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #c9302c;
            }
        """)
        
        # 创建按钮布局
        button_layout = QHBoxLayout()
        button_layout.addWidget(run_btn)
        button_layout.addWidget(exit_btn)
        
        # 添加日志显示区域
        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        
        # 添加所有组件到面板
        layout.addWidget(file_group)
        layout.addWidget(material_group)
        layout.addWidget(factor_group)
        layout.addLayout(button_layout)
        layout.addWidget(log_group)
        layout.addStretch()
        
        return panel
        
    def create_display_panel(self):
        """创建右侧显示面板"""
        panel = QWidget()
        layout = QGridLayout(panel)
        
        # 创建四个VTK窗口
        titles = {
            'ppt': 'PPT分布',
            'base': '基础适配',
            'material': '材质优化',
            'final': '最终结果'
        }
        
        for i, (name, title) in enumerate(titles.items()):
            group = QGroupBox(title)
            group_layout = QVBoxLayout(group)
            
            # 创建VTK部件
            vtk_widget = QVTKRenderWindowInteractor(group)
            group_layout.addWidget(vtk_widget)
            
            # 设置VTK窗口
            render_window = vtk_widget.GetRenderWindow()
            render_window.SetSize(512, 512)
            
            # 设置交互器
            interactor = render_window.GetInteractor()
            interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
            
            # 更新visualizer的窗口和交互器
            self.fitting.visualizers[name].render_window = render_window
            self.fitting.visualizers[name].interactor = interactor
            self.fitting.visualizers[name].renderer.SetBackground(1, 1, 1)  # 白色背景
            
            # 初始化交互器
            render_window.AddRenderer(self.fitting.visualizers[name].renderer)
            interactor.Initialize()
            
            # 添加到布局
            layout.addWidget(group, i//2, i%2)
        
        return panel
        
    def select_foot_model(self):
        """选择足部模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择足部模型", "", "STL文件 (*.stl)"
        )
        if file_path:
            self.foot_label.setText(os.path.basename(file_path))
            self.foot_path = file_path
            
    def select_shoe_model(self):
        """选择鞋楦模型文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择鞋楦模型", "", "OBJ文件 (*.obj)"
        )
        if file_path:
            self.shoe_label.setText(os.path.basename(file_path))
            self.shoe_path = file_path
            
    def get_material_properties(self):
        """获取材质参数设置"""
        properties = {}
        for material, inputs in self.material_inputs.items():
            try:
                e_value = float(inputs['E'].text() or inputs['E'].placeholderText())
                v_value = float(inputs['v'].text() or inputs['v'].placeholderText())
                properties[material] = {
                    'E': e_value,
                    'v': v_value
                }
            except ValueError:
                QMessageBox.warning(self, "警告", f"{material}的参数输入无效，将使用默认值")
                properties[material] = DEFAULT_MATERIALS[material]
        return properties
        
    def run_analysis(self):
        """运行分析"""
        try:
            # 检查模型是否已加载
            if not hasattr(self, 'foot_path') or not hasattr(self, 'shoe_path'):
                QMessageBox.warning(self, "警告", "请先选择足部模型和鞋楦模型")
                return
                
            # 1. 加载模型
            if not self.fitting.load_models():
                QMessageBox.critical(self, "错误", "模型加载失败")
                return
                
            # 2. 设置材质属性
            properties = self.get_material_properties()
            if not properties:
                QMessageBox.critical(self, "错误", "材质参数设置无效")
                return
            self.fitting.set_material_properties(properties)
            
            # 3. 获取影响因子
            material_factor = self.material_factor_slider.value() / 50.0  # 0-2范围
            ppt_factor = self.ppt_factor_slider.value() / 100.0  # 0-1范围
            
            # 4. 运行分析
            logging.info("开始运行分析...")
            logging.info(f"材质影响因子: {material_factor:.2f}")
            logging.info(f"PPT影响因子: {ppt_factor:.2f}")
            
            results = self.fitting.run_analysis(material_factor, ppt_factor)
            if results is None:
                QMessageBox.critical(self, "错误", "分析失败")
                return
                
            # 5. 更新显示
            self.update_visualization(results)
            
            # 6. 显示统计结果
            self.show_statistics(results['stats'])
            
            logging.info("分析完成")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"分析过程出错: {str(e)}")
            logging.error(f"分析失败: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def update_visualization(self, results):
        """更新显示"""
        try:
            if results is None:
                raise ValueError("无效的结果数据")
            
            # 更新所有显示
            self.fitting.update_visualization(results)
            
            # 刷新所有窗口
            for name in ['ppt', 'base', 'material', 'final']:
                self.fitting.visualizers[name].render_window.Render()
            
        except Exception as e:
            logging.error(f"显示更新失败: {str(e)}")
            QMessageBox.critical(self, "错误", f"显示更新失败: {str(e)}")
            
    def show_statistics(self, stats):
        """显示统计结果"""
        msg = "分析结果统计：\n\n"
        for stage, data in stats.items():
            msg += f"{stage}阶段：\n"
            msg += f"  平均距离: {data['mean']:.2f} mm\n"
            msg += f"  最大干涉: {data['max_interference']:.2f} mm\n"
            msg += f"  最大间隙: {data['max_gap']:.2f} mm\n"
            msg += f"  干涉点数: {data['interference_points']}\n"
            msg += f"  间隙点数: {data['gap_points']}\n\n"
            
        QMessageBox.information(self, "分析结果", msg)
        
    def setup_logging(self):
        """设置日志处理"""
        class QTextEditLogger(logging.Handler):
            def __init__(self, text_widget):
                super().__init__()
                self.text_widget = text_widget
                
            def emit(self, record):
                msg = self.format(record)
                self.text_widget.append(msg)
        
        # 配置日志
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # 添加文本框处理器
        text_handler = QTextEditLogger(self.log_text)
        text_handler.setFormatter(formatter)
        logging.getLogger().addHandler(text_handler)
        
    def close_application(self):
        """安全退出程序"""
        try:
            # 确认对话框
            reply = QMessageBox.question(
                self, '确认退出',
                "确定要退出程序吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                logging.info("正在安全退出程序...")
                
                # 停止所有VTK交互器并清理资源
                for name, visualizer in self.fitting.visualizers.items():
                    if visualizer.interactor:
                        visualizer.interactor.TerminateApp()
                    if visualizer.renderer:
                        visualizer.renderer.RemoveAllViewProps()
                    if visualizer.render_window:
                        visualizer.render_window.Finalize()
                
                # 关闭主窗口
                self.close()
                
                # 退出应用程序
                QApplication.instance().quit()
                
        except Exception as e:
            logging.error(f"退出程序时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            # 强制退出
            QApplication.instance().quit()
        
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
                # 停止所有VTK交互器并清理资源
                for name, visualizer in self.fitting.visualizers.items():
                    if visualizer.interactor:
                        visualizer.interactor.TerminateApp()
                    if visualizer.renderer:
                        visualizer.renderer.RemoveAllViewProps()
                    if visualizer.render_window:
                        visualizer.render_window.Finalize()
                
                event.accept()
            else:
                event.ignore()
                
        except Exception as e:
            logging.error(f"处理关闭事件时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            event.accept()