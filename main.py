import vtk
import logging
import os
from models import FootModel, ShoeLast
from soft_tissue import PPTData
from fitting_process import FittingAnalyzer
from visualization import Visualizer
from config import PATHS, DEFAULT_MATERIALS
from registration import ModelRegistration
import numpy as np

class VirtualFitting:
    def __init__(self):
        """初始化虚拟试穿系统"""
        self.gur_model = None      # GUR模型
        self.foot_model = None     # 目标足部模型
        self.shoe_model = None     # 鞋楦模型
        self.ppt_data = PPTData()
        self.analyzer = FittingAnalyzer()
        self.registration = ModelRegistration()
        
        # 创建四个VTK渲染窗口
        self.visualizers = {}
        for name in ['ppt', 'base', 'material', 'final']:
            render_window = vtk.vtkRenderWindow()
            render_window.SetSize(512, 512)  # 每个窗口大小设为512x512
            
            interactor = vtk.vtkRenderWindowInteractor()
            interactor.SetRenderWindow(render_window)
            interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
            
            self.visualizers[name] = Visualizer(render_window, interactor)
        
    def load_models(self):
        """加载所有模型和数据"""
        try:
            # 检查文件是否存在
            for name, path in PATHS.items():
                if not os.path.exists(path):
                    logging.error(f"找不到文件: {path}")
                    return False
            
            # 1. 加载GUR模型
            logging.info("加载GUR模型...")
            self.gur_model = FootModel(PATHS['GUR_MODEL'])
            logging.info(f"GUR模型加载完成: {len(self.gur_model.vertices)} 个顶点")
            
            # 2. 加载目标足部模型
            logging.info("加载足部模型...")
            self.foot_model = FootModel(PATHS['FOOT_MODEL'])
            logging.info(f"足部模型加载完成: {len(self.foot_model.vertices)} 个顶点")
            
            # 3. 加载鞋楦模型
            logging.info("加载鞋楦模型...")
            self.shoe_model = ShoeLast()
            if not self.shoe_model.load_model(PATHS['SHOE_MODEL']):
                return False
                
            # 4. 加载PPT数据
            logging.info("加载PPT数据...")
            if not self.ppt_data.load_data(PATHS['PPT_DATA']):
                return False
                
            # 5. 设置默认材质属性
            for material, props in DEFAULT_MATERIALS.items():
                self.shoe_model.set_material_property(material, props['E'], props['v'])
                
            return True
            
        except Exception as e:
            logging.error(f"模型加载失败: {str(e)}")
            return False
            
    def run_analysis(self, material_factor=1.0, ppt_factor=0.3):
        """运行分析流程"""
        try:
            # 1. 加载模型
            if not self.load_models():
                return None
                
            # 2. 设置影响因子
            self.analyzer.set_factors(material_factor, ppt_factor)
            
            # 3. 处理PPT数据和适配计算
            ppt_values = None
            ppt_mapping = None
            
            # 先进行GUR配准和PPT映射
            if not self.registration.register_gur_to_target(self.gur_model, self.foot_model):
                return None
                
            ppt_mapping = self.registration.transform_ppt_to_target(
                self.ppt_data.coordinates,
                self.foot_model
            )
            if ppt_mapping is None:
                return None
                
            self.ppt_data.mapped_coords = ppt_mapping['positions']
            ppt_values = self.ppt_data.get_interpolated_ppt(self.foot_model.vertices)
            if ppt_values is None:
                return None
                
            # 计算适配距离（包括PPT影响）
            distances = self.analyzer.calculate_distances(
                self.foot_model,
                self.shoe_model,
                ppt_values
            )
            
            if distances is None:
                return None
                
            # 4. 分析结果
            stats = self.analyzer.analyze_results(distances, self.foot_model)
            
            return {
                'distances': distances,
                'ppt_values': ppt_values,
                'stats': stats,
                'ppt_mapping': ppt_mapping
            }
            
        except Exception as e:
            logging.error(f"分析过程失败: {str(e)}")
            return None
            
    def update_visualization(self, results):
        """更新可视化显示"""
        try:
            if results is None:
                return
            
            # 获取基础适配的距离范围
            base_distances = results['distances']['base']
            scalar_range = [np.min(base_distances), np.max(base_distances)]
            
            # 使用相同的颜色范围显示所有云图
            self.visualizers['base'].display_fitting_result(
                self.foot_model,
                results['distances']['base'],
                "基础适配距离",
                scalar_range
            )
            
            self.visualizers['material'].display_fitting_result(
                self.foot_model,
                results['distances']['material'],
                "材质优化距离",
                scalar_range
            )
            
            self.visualizers['final'].display_fitting_result(
                self.foot_model,
                results['distances']['final'],
                "最终优化距离",
                scalar_range
            )
            
            # PPT分布使用自己的颜色范围
            self.visualizers['ppt'].display_ppt_distribution(
                self.foot_model,
                results['ppt_values']
            )
            
        except Exception as e:
            logging.error(f"可视化更新失败: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def set_material_properties(self, properties):
        """设置材质属性"""
        if self.shoe_model:
            for material, props in properties.items():
                self.shoe_model.set_material_property(
                    material,
                    props['E'],
                    props['v']
                )