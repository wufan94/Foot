import vtk
import numpy as np
import os
import random
import trimesh

def create_vtk_renderer():
    """创建VTK渲染器和交互器"""
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(1.0, 1.0, 1.0)  # 白色背景
    
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)
    
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.Initialize()
    
    return renderer, render_window, interactor

def display_model(model, renderer, color=(1, 0, 0)):
    """显示普通模型
    Args:
        model: trimesh对象
        renderer: VTK渲染器
        color: RGB颜色元组
    """
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
    
    # 添加到渲染器
    renderer.AddActor(actor)
    return actor

def main():
    # 设置文件路径
    file_path = r"C:\Users\wufan\Documents\MyFiles\3 研究院文件\项目文件\足鞋工效2024\2项目研发相关材料\3虚拟适配相关材料\虚拟适配250325\Shoe_Model\35L.obj"
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        return
    
    try:
        # 使用trimesh加载模型
        print("加载模型...")
        shoe_model = trimesh.load(file_path)
        
        # 检查并打印模型信息
        print("\n检查模型加载情况:")
        if isinstance(shoe_model, trimesh.Scene):
            print("鞋模型是一个Scene对象，包含以下几何体：")
            
            # 创建渲染器和窗口
            renderer, render_window, interactor = create_vtk_renderer()
            
            # 为每个几何体创建不同颜色的显示
            for name, geometry in shoe_model.geometry.items():
                print(f"\n几何体名称: {name}")
                print(f"  顶点数: {len(geometry.vertices)}")
                print(f"  面片数: {len(geometry.faces)}")
                
                # 生成随机颜色
                color = (random.random(), random.random(), random.random())
                
                # 显示该几何体
                actor = display_model(geometry, renderer, color)
                
                # 创建标签
                label = vtk.vtkVectorText()
                label.SetText(name)
                
                # 创建标签mapper
                label_mapper = vtk.vtkPolyDataMapper()
                label_mapper.SetInputConnection(label.GetOutputPort())
                
                # 创建标签actor
                label_actor = vtk.vtkFollower()
                label_actor.SetMapper(label_mapper)
                label_actor.SetScale(2.0, 2.0, 2.0)
                
                # 设置标签位置（使用几何体的中心）
                center = geometry.vertices.mean(axis=0)
                label_actor.SetPosition(center)
                
                # 添加标签
                renderer.AddActor(label_actor)
                
                # 如果有材质信息，打印出来
                if hasattr(geometry, 'visual') and hasattr(geometry.visual, 'material'):
                    material = geometry.visual.material
                    print(f"  材质名称: {material.name if hasattr(material, 'name') else '未命名'}")
            
            # 重置相机
            renderer.ResetCamera()
            
            # 启动交互
            render_window.Render()
            interactor.Start()
            
        else:
            print("模型不是Scene对象，无法显示分区信息")
            
    except Exception as e:
        print(f"发生错误：{str(e)}")
        return

if __name__ == "__main__":
    main() 