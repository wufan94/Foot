import vtk
import numpy as np
import logging
from vtk.util import numpy_support

class Visualizer:
    def __init__(self, render_window, interactor):
        """初始化可视化器"""
        self.render_window = render_window
        self.interactor = interactor
        self.renderer = vtk.vtkRenderer()
        self.render_window.AddRenderer(self.renderer)
        
        # 设置背景和相机
        self.renderer.SetBackground(1, 1, 1)  # 白色背景
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(0, 200, 0)
        camera.SetFocalPoint(0, 0, 0)
        camera.SetViewUp(0, 0, 1)
        
        # 保存当前显示的actors
        self.current_actors = []
    
    def _clear_display(self):
        """清除当前显示"""
        for actor in self.current_actors:
            self.renderer.RemoveActor(actor)
        self.current_actors = []
    
    def _to_vtk(self, vertices, faces):
        """转换为VTK格式"""
        points = vtk.vtkPoints()
        points.SetData(numpy_support.numpy_to_vtk(vertices))
        
        cells = vtk.vtkCellArray()
        for face in faces:
            cells.InsertNextCell(len(face), face)
            
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(cells)
        return polydata
        
    def display_models(self, foot_model, shoe_model):
        """显示基础模型"""
        try:
            self.renderer.RemoveAllViewProps()
            
            # 显示足部模型
            foot_polydata = self._to_vtk(foot_model.vertices, foot_model.faces)
            foot_mapper = vtk.vtkPolyDataMapper()
            foot_mapper.SetInputData(foot_polydata)
            foot_actor = vtk.vtkActor()
            foot_actor.SetMapper(foot_mapper)
            foot_actor.GetProperty().SetColor(0.8, 0.8, 0.8)
            self.renderer.AddActor(foot_actor)
            
            # 显示鞋楦模型（半透明）
            shoe_polydata = self._to_vtk(shoe_model.vertices, shoe_model.faces)
            shoe_mapper = vtk.vtkPolyDataMapper()
            shoe_mapper.SetInputData(shoe_polydata)
            shoe_actor = vtk.vtkActor()
            shoe_actor.SetMapper(shoe_mapper)
            shoe_actor.GetProperty().SetColor(0.6, 0.8, 0.8)
            shoe_actor.GetProperty().SetOpacity(0.3)
            self.renderer.AddActor(shoe_actor)
            
            self.renderer.ResetCamera()
            self.render_window.Render()
            
        except Exception as e:
            logging.error(f"显示模型失败: {str(e)}")
    
    def display_ppt_distribution(self, model, ppt_values):
        """显示PPT分布"""
        try:
            if ppt_values is None:
                raise ValueError("无效的PPT数据")
                
            # 获取实际的压力值范围
            min_pressure = np.min(ppt_values[ppt_values > 0])  # 忽略0值
            max_pressure = np.max(ppt_values)
            
            logging.info(f"PPT值范围: {min_pressure:.2f} - {max_pressure:.2f} kPa")
            
            # 创建网格模型
            polydata = self._create_mesh(model.vertices, model.faces)
            
            # 添加压力值作为标量数据
            scalars = vtk.vtkFloatArray()
            for value in ppt_values:
                scalars.InsertNextValue(value)
            polydata.GetPointData().SetScalars(scalars)
            
            # 创建颜色映射
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polydata)
            
            # 设置颜色映射范围为实际压力值范围
            mapper.SetScalarRange(min_pressure, max_pressure)
            
            # 使用红蓝色标
            lut = vtk.vtkLookupTable()
            lut.SetHueRange(0.667, 0.0)  # 从蓝到红
            lut.SetSaturationRange(1.0, 1.0)
            lut.SetValueRange(1.0, 1.0)
            lut.Build()
            mapper.SetLookupTable(lut)
            
            # 创建actor
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            
            # 清除当前渲染器并添加新的actor
            self.renderer.RemoveAllViewProps()
            self.renderer.AddActor(actor)
            
            # 添加颜色条
            scalar_bar = vtk.vtkScalarBarActor()
            scalar_bar.SetLookupTable(lut)
            scalar_bar.SetTitle("压力值 (kPa)")
            scalar_bar.SetNumberOfLabels(5)
            scalar_bar.SetWidth(0.08)
            scalar_bar.SetHeight(0.6)
            scalar_bar.SetPosition(0.92, 0.2)
            self.renderer.AddActor2D(scalar_bar)
            
            # 重置相机并渲染
            self.renderer.ResetCamera()
            self.render_window.Render()
            
        except Exception as e:
            logging.error(f"PPT分布显示失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def display_fitting_result(self, foot_model, distances, title, scalar_range=None):
        """显示适配结果云图"""
        try:
            self._clear_display()
            
            # 计算实际的距离范围
            min_dist = np.min(distances)  # 最大干涉（负值）
            max_dist = np.max(distances)  # 最大间隙（正值）
            
            # 如果没有提供颜色范围，使用实际范围
            if scalar_range is None:
                scalar_range = [min_dist, max_dist]
            
            # 创建颜色映射
            lut = vtk.vtkLookupTable()
            lut.SetNumberOfColors(256)
            
            # 使用jet颜色方案
            # 红色->黄色->青色->蓝色
            lut.SetHueRange(0.0, 0.667)  # 从红到蓝
            lut.SetSaturationRange(1.0, 1.0)
            lut.SetValueRange(1.0, 1.0)
            lut.Build()
            
            # 创建模型显示
            polydata = self._create_mesh(foot_model.vertices, foot_model.faces)
            polydata.GetPointData().SetScalars(numpy_support.numpy_to_vtk(distances))
            
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polydata)
            mapper.SetLookupTable(lut)
            mapper.SetScalarRange(scalar_range)
            
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            self.current_actors.append(actor)
            
            # 添加颜色条
            scalar_bar = vtk.vtkScalarBarActor()
            scalar_bar.SetTitle(f"{title} (mm)")
            scalar_bar.SetLookupTable(lut)
            scalar_bar.SetWidth(0.08)
            scalar_bar.SetHeight(0.6)
            scalar_bar.SetPosition(0.92, 0.2)
            scalar_bar.SetNumberOfLabels(5)
            self.current_actors.append(scalar_bar)
            
            # 添加到渲染器
            for actor in self.current_actors:
                if isinstance(actor, vtk.vtkActor):
                    self.renderer.AddActor(actor)
                elif isinstance(actor, vtk.vtkActor2D):
                    self.renderer.AddActor2D(actor)
            
            self.renderer.ResetCamera()
            self.render_window.Render()
            
        except Exception as e:
            logging.error(f"适配结果显示失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _create_mesh(self, vertices, faces):
        """创建VTK网格"""
        polydata = vtk.vtkPolyData()
        
        # 设置顶点
        points = vtk.vtkPoints()
        points.SetData(numpy_support.numpy_to_vtk(vertices))
        polydata.SetPoints(points)
        
        # 设置面片
        cells = vtk.vtkCellArray()
        for face in faces:
            cells.InsertNextCell(len(face), face)
        polydata.SetPolys(cells)
        
        return polydata