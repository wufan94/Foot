import trimesh
import numpy as np
import vtk
from vtk.util import numpy_support

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

def display_model_with_groups(model, renderer, group_colors=None):
    """显示带有分组颜色的模型
    Args:
        model: trimesh对象
        renderer: VTK渲染器
        group_colors: 分组颜色字典
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
    
    # 如果有分组信息，创建颜色数组
    if group_colors:
        colors = vtk.vtkUnsignedCharArray()
        colors.SetName("Colors")
        colors.SetNumberOfComponents(3)
        
        # 为每个面设置颜色
        for face_idx in range(len(model.faces)):
            # 获取面片所属的组
            face_group = None
            for group_name, face_indices in group_colors.items():
                if face_idx in face_indices:
                    face_group = group_name
                    break
            
            if face_group and face_group in group_colors:
                color = group_colors[face_group]
                colors.InsertNextTuple3(*[int(c * 255) for c in color])
            else:
                colors.InsertNextTuple3(200, 200, 200)  # 默认灰色
        
        polydata.GetCellData().SetScalars(colors)
    
    # 创建映射器
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    
    # 创建actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    # 添加到渲染器
    renderer.AddActor(actor)
    return actor

def read_obj_groups(file_path):
    """直接读取OBJ文件中的顶点组信息
    Args:
        file_path: OBJ文件路径
    Returns:
        vertex_groups: 顶点组信息字典
    """
    vertex_groups = {}
    current_group = None
    face_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('g '):  # 组定义
                current_group = line[2:].strip()
                if current_group:
                    vertex_groups[current_group] = []
            elif line.startswith('f '):  # 面片
                if current_group:
                    vertex_groups[current_group].append(face_count)
                face_count += 1
    
    return vertex_groups

def print_model_info(model):
    """打印模型的详细信息"""
    print("\n模型基本信息:")
    print(f"顶点数量: {len(model.vertices)}")
    print(f"面片数量: {len(model.faces)}")
    print(f"表面积: {model.area:.2f} 平方毫米")
    
    print("\n模型属性:")
    for attr in dir(model):
        if not attr.startswith('_'):  # 跳过私有属性
            try:
                value = getattr(model, attr)
                if not callable(value):  # 跳过方法
                    print(f"{attr}: {value}")
            except:
                pass
    
    if hasattr(model, 'metadata'):
        print("\n元数据信息:")
        for key, value in model.metadata.items():
            print(f"{key}: {value}")
    
    if hasattr(model, 'visual'):
        print("\n视觉属性:")
        for attr in dir(model.visual):
            if not attr.startswith('_'):
                try:
                    value = getattr(model.visual, attr)
                    if not callable(value):
                        print(f"{attr}: {value}")
                except:
                    pass

def main():
    # 设置文件路径
    file_path = r"C:\Users\wufan\Documents\MyFiles\3 研究院文件\项目文件\足鞋工效2024\2项目研发相关材料\3虚拟适配相关材料\虚拟适配250325\Shoe_Model\35R.obj"
    
    print("正在加载模型...")
    try:
        # 直接读取OBJ文件中的顶点组信息
        print("\n检查OBJ文件中的顶点组信息...")
        vertex_groups = read_obj_groups(file_path)
        
        if vertex_groups:
            print("找到顶点组信息：")
            for group_name, face_indices in vertex_groups.items():
                print(f"  - {group_name}: {len(face_indices)} 个面片")
        else:
            print("未找到顶点组信息")
        
        # 加载模型
        scene = trimesh.load(file_path)
        
        # 如果是场景，获取第一个网格
        if isinstance(scene, trimesh.Scene):
            print("\n检测到场景对象，获取第一个网格...")
            if len(scene.geometry) > 0:
                model = list(scene.geometry.values())[0]
            else:
                raise ValueError("场景中没有网格对象")
        else:
            model = scene
        
        # 打印模型的详细信息
        print_model_info(model)
        
        # 创建渲染器和窗口
        renderer, render_window, interactor = create_vtk_renderer()
        
        # 设置分组颜色
        if vertex_groups:
            # 为每个分组分配不同的颜色
            colors = {
                'group1': (1, 0, 0),      # 红色
                'group2': (0, 1, 0),      # 绿色
                'group3': (0, 0, 1),      # 蓝色
                'group4': (1, 1, 0),      # 黄色
                'group5': (1, 0, 1),      # 紫色
                'group6': (0, 1, 1),      # 青色
                'group7': (1, 0.5, 0),    # 橙色
                'group8': (0.5, 1, 0),    # 黄绿色
                'group9': (0, 0.5, 1),    # 浅蓝色
                'group10': (1, 0, 0.5),   # 粉红色
            }
            
            # 将颜色分配给实际的分组
            group_colors = {}
            for i, (group_name, _) in enumerate(vertex_groups.items()):
                color_key = f'group{(i % len(colors)) + 1}'
                group_colors[group_name] = colors[color_key]
            
            print("\n显示模型顶点组...")
            display_model_with_groups(model, renderer, group_colors)
        else:
            print("\n显示模型（无顶点组信息）...")
            display_model_with_groups(model, renderer)
        
        # 设置相机位置
        renderer.ResetCamera()
        camera = renderer.GetActiveCamera()
        camera.Elevation(20)
        camera.Azimuth(20)
        
        # 开始显示
        render_window.Render()
        interactor.Start()
        
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 