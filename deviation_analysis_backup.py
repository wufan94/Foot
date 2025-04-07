import vtk
import numpy as np
import trimesh
from models import FootModel, ShoeLast
from scipy.spatial import cKDTree
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

def display_model(model, renderer, color=(1, 0, 0)):
    """显示普通模型（无云图）
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

def compute_interference(foot_model, shoe_model):
    """计算干涉距离
    Args:
        foot_model: 足部模型（trimesh对象）
        shoe_model: 鞋内腔模型（trimesh对象）
    Returns:
        distances: 每个顶点的干涉距离（负值表示干涉，正值表示间隙）
    """
    # 初始化结果数组
    distances = np.full(len(foot_model.vertices), np.nan)
    
    # 获取足部模型的顶点和法向量
    foot_vertices = foot_model.vertices
    foot_vertex_normals = foot_model.vertex_normals
    
    # 获取鞋内腔模型的面和法向量
    shoe_faces = shoe_model.faces
    shoe_vertices = shoe_model.vertices
    shoe_face_centers = np.mean(shoe_vertices[shoe_faces], axis=1)
    shoe_face_normals = np.cross(
        shoe_vertices[shoe_faces[:, 1]] - shoe_vertices[shoe_faces[:, 0]],
        shoe_vertices[shoe_faces[:, 2]] - shoe_vertices[shoe_faces[:, 0]]
    )
    # 归一化鞋内腔法向量并确保指向内部
    shoe_norms = np.linalg.norm(shoe_face_normals, axis=1)
    valid_shoe_norms = shoe_norms > 1e-10
    shoe_face_normals[valid_shoe_norms] = shoe_face_normals[valid_shoe_norms] / shoe_norms[valid_shoe_norms, np.newaxis]
    shoe_face_normals = -shoe_face_normals  # 确保法向量指向内部
    
    # 构建KD树用于快速查找最近面
    shoe_face_tree = cKDTree(shoe_face_centers)
    
    # 设置参数
    max_distance = 25.0  # 最大距离阈值(mm)
    
    # 对每个足部顶点进行处理
    for vertex_idx, (vertex, normal) in enumerate(zip(foot_vertices, foot_vertex_normals)):
        # 查找最近的鞋内腔面（返回k个最近邻）
        distances_to_shoe, shoe_face_indices = shoe_face_tree.query(vertex, k=10)
        
        # 初始化当前顶点的距离计算
        total_weight = 0
        total_distance = 0
        
        for dist, shoe_idx in zip(distances_to_shoe, shoe_face_indices):
            if dist > max_distance:
                continue
                
            shoe_normal = shoe_face_normals[shoe_idx]
            shoe_center = shoe_face_centers[shoe_idx]
            
            # 计算从顶点到鞋面中心的向量
            direction = vertex - shoe_center
            direction_norm = np.linalg.norm(direction)
            if direction_norm < 1e-10:
                continue
            
            # 计算从鞋面到顶点的有符号距离
            signed_distance = np.dot(direction, shoe_normal)
            
            # 权重仅基于距离
            weight = 1.0 - (dist / max_distance)
            
            total_weight += weight
            total_distance += signed_distance * weight
        
        # 如果找到有效的对应关系
        if total_weight > 0:
            distances[vertex_idx] = total_distance / total_weight
    
    # 将超出范围的距离设置为NaN
    distances[distances > 15.0] = np.nan  # 大于25mm的设为NaN
    distances[distances < -10.0] = np.nan  # 删除小于-10mm的区域
    
    # 打印统计信息
    valid_distances = distances[~np.isnan(distances)]
    if len(valid_distances) > 0:
        print(f"总顶点数: {len(distances)}")
        print(f"有效顶点数: {len(valid_distances)}")
        print(f"距离范围: {np.min(valid_distances):.2f}mm 到 {np.max(valid_distances):.2f}mm")
    
    return distances

def display_model_with_interference(model, distances, renderer, is_shoe=False, shoe_model=None):
    """显示带有干涉云图的模型
    Args:
        model: trimesh对象
        distances: 干涉距离数组
        renderer: VTK渲染器
        is_shoe: 是否是鞋模型（不再使用）
        shoe_model: 不再使用
    """
    # 如果是鞋模型，直接返回（不再显示）
    if is_shoe:
        return None
    
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
    
    # 添加干涉距离数据
    interference_data = vtk.vtkFloatArray()
    interference_data.SetName("Interference")
    
    # 处理NaN值并设置数据
    valid_distances = np.copy(distances)
    
    # 将数据添加到VTK数组
    for dist in valid_distances:
        interference_data.InsertNextValue(float(dist))
    
    polydata.GetPointData().SetScalars(interference_data)
    
    # 设置固定的距离范围（-25mm到25mm）
    min_dist = -25.0
    max_dist = 25.0
    
    # 创建映射器
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.SetScalarRange(min_dist, max_dist)
    
    # 创建颜色查找表
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(512)  # 增加颜色表的分辨率
    lut.SetTableRange(min_dist, max_dist)
    
    # 设置jet颜色映射
    for i in range(512):
        t = i / 511.0  # 归一化到[0,1]
        
        # jet颜色方案的实现
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
        
        # 确保颜色值在[0,1]范围内
        r = max(0.0, min(1.0, r))
        g = max(0.0, min(1.0, g))
        b = max(0.0, min(1.0, b))
        
        lut.SetTableValue(i, r, g, b, 1.0)
    
    # 设置NaN值的颜色为浅灰色
    lut.SetNanColor(0.9, 0.9, 0.9, 1.0)
    lut.Build()
    
    # 设置映射器使用颜色查找表
    mapper.SetLookupTable(lut)
    mapper.SetUseLookupTableScalarRange(True)
    
    # 创建actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    # 添加颜色条
    scalar_bar = vtk.vtkScalarBarActor()
    scalar_bar.SetLookupTable(mapper.GetLookupTable())
    scalar_bar.SetTitle("干涉距离 (mm)")
    scalar_bar.SetWidth(0.08)
    scalar_bar.SetHeight(0.6)
    scalar_bar.SetPosition(0.92, 0.2)
    scalar_bar.SetNumberOfLabels(5)
    scalar_bar.GetLabelTextProperty().SetColor(0, 0, 0)  # 黑色文字
    scalar_bar.GetTitleTextProperty().SetColor(0, 0, 0)  # 黑色标题
    
    # 添加到渲染器
    renderer.AddActor(actor)
    renderer.AddActor2D(scalar_bar)
    return actor

def get_mesh_density(mesh):
    """计算网格密度（每平方毫米的面片数）
    Args:
        mesh: trimesh对象
    Returns:
        density: 网格密度（面片数/表面积）
    """
    surface_area = np.sum(mesh.area_faces)  # 总表面积（平方毫米）
    face_count = len(mesh.faces)
    return face_count / surface_area

def simplify_to_target_density(mesh, target_density):
    """根据目标网格密度简化模型
    Args:
        mesh: 要简化的trimesh模型
        target_density: 目标网格密度（面片数/平方毫米）
    Returns:
        simplified_mesh: 简化后的模型
    """
    try:
        # 计算当前网格密度
        current_area = mesh.area
        current_faces = len(mesh.faces)
        current_density = current_faces / current_area
        
        print(f"当前网格密度: {current_density:.4f} 面片/平方毫米")
        print(f"目标网格密度: {target_density:.4f} 面片/平方毫米")
        
        # 如果当前密度已经小于目标密度，直接返回原始模型
        if current_density <= target_density:
            print("当前网格密度已经小于目标密度，无需简化")
            return mesh
        
        # 计算目标面片数
        target_faces = max(int(current_area * target_density), 100)  # 确保至少保留100个面片
        
        # 计算简化比例
        ratio = 1.0 - (target_faces / current_faces)
        ratio = min(max(ratio, 0.0), 0.99)  # 限制简化比例在0-0.99之间
        
        print(f"简化比例: {ratio:.2%}")
        
        # 创建VTK点数据
        points = vtk.vtkPoints()
        for vertex in mesh.vertices:
            points.InsertNextPoint(vertex)
        
        # 创建VTK面片数据
        cells = vtk.vtkCellArray()
        for face in mesh.faces:
            try:
                cell = vtk.vtkTriangle()
                cell.GetPointIds().SetId(0, int(face[0]))
                cell.GetPointIds().SetId(1, int(face[1]))
                cell.GetPointIds().SetId(2, int(face[2]))
                cells.InsertNextCell(cell)
            except Exception as e:
                print(f"警告：跳过无效面片 {face}: {str(e)}")
                continue
        
        # 创建PolyData
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(cells)
        
        # 清理数据
        cleaner = vtk.vtkCleanPolyData()
        cleaner.SetInputData(polydata)
        cleaner.Update()
        cleaned_polydata = cleaner.GetOutput()
        
        # 创建简化过滤器
        decimate = vtk.vtkDecimatePro()
        decimate.SetInputData(cleaned_polydata)
        decimate.SetTargetReduction(ratio)
        decimate.PreserveTopologyOn()
        decimate.BoundaryVertexDeletionOff()
        decimate.Update()
        
        # 获取简化后的模型
        simplified = decimate.GetOutput()
        
        if simplified.GetNumberOfPoints() == 0 or simplified.GetNumberOfCells() == 0:
            print("警告：简化失败，返回原始模型")
            return mesh
        
        # 转换回trimesh格式
        # 获取顶点
        points = simplified.GetPoints()
        vertices = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])
        
        # 获取面片
        faces = []
        for i in range(simplified.GetNumberOfCells()):
            cell = simplified.GetCell(i)
            if cell.GetNumberOfPoints() == 3:
                face = [cell.GetPointId(j) for j in range(3)]
                faces.append(face)
        faces = np.array(faces)
        
        if len(faces) == 0:
            print("警告：简化后没有有效面片，返回原始模型")
            return mesh
        
        # 创建新的trimesh对象
        simplified_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # 打印简化信息
        print(f"网格密度: {current_density:.4f} -> {len(faces)/simplified_mesh.area:.4f} 面片/平方毫米")
        print(f"网格面片: {current_faces} -> {len(faces)} 个")
        print(f"表面积: {current_area:.2f} 平方毫米")
        
        return simplified_mesh
        
    except Exception as e:
        print(f"简化过程出错: {str(e)}")
        print("返回原始模型")
        return mesh

def compute_distances(shoe_mesh, foot_mesh, max_angle=40):
    """计算从鞋内腔到足部模型的距离
    Args:
        shoe_mesh: 简化后的鞋内腔模型
        foot_mesh: 足部模型
        max_angle: 最大允许角度（度）
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
    
    # 计算鞋内腔模型面的中心点和法向量
    shoe_face_centers = np.mean(shoe_mesh.vertices[shoe_mesh.faces], axis=1)
    shoe_face_normals = -shoe_mesh.face_normals  # 取反使其指向内部
    
    # 构建KD树
    shoe_tree = cKDTree(shoe_face_centers)
    
    # 设置参数
    max_angle_rad = np.radians(max_angle)
    max_distance = 15.0  # 最大距离阈值(mm)  设置这里可以去除一些非常强的干涉
    
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
    distances[distances > 25.0] = np.nan  # 大于25mm的设为NaN
    distances[distances < -10] = np.nan  # 小于-10mm的区域设为NaN
    
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

def convert_distance_to_pressure(distance, material_type):
    """根据不同材质的拟合方程将干涉距离转换为力值
    Args:
        distance: 干涉距离（mm，正值表示压缩）
        material_type: 材质类型（'试样1', '试样2', '试样4', '后跟上部', '无限伸缩', '鞋底'）
    Returns:
        force: 力值（N）
    """
    if distance <= 0:  # 没有压缩，力为0
        return 0.0
    
    # 正的干涉距离就是压缩量
    x = distance
    
    # 根据不同材质使用对应的拟合方程计算力值(N)
    if material_type == '试样1':  # 鞋面
        # F = -0.0089x⁴ + 0.4375x³ - 3.0884x² + 8.2764x - 3.3784
        force = -0.0089 * x**4 + 0.4375 * x**3 - 3.0884 * x**2 + 8.2764 * x - 3.3784
    elif material_type == '试样2':  # 鞋尖
        # F = -0.0122x⁴ + 0.4174x³ + 1.9400x² - 11.9956x + 8.7726
        force = -0.0122 * x**4 + 0.4174 * x**3 + 1.9400 * x**2 - 11.9956 * x + 8.7726
    elif material_type == '试样4':  # 鞋舌下部
        # F = -0.0107x⁴ + 0.4648x³ - 1.5420x² - 2.8860x + 5.6783
        force = -0.0107 * x**4 + 0.4648 * x**3 - 1.5420 * x**2 - 2.8860 * x + 5.6783
    elif material_type == '后跟上部':  # 后跟
        # F = -0.0067x⁴ + 0.4720x³ - 6.3526x² + 27.9414x - 20.6310
        force = -0.0067 * x**4 + 0.4720 * x**3 - 6.3526 * x**2 + 27.9414 * x - 20.6310
    elif material_type == '鞋舌上部':  # 鞋舌上部（试样7）
        # F = -0.0084x⁴ + 0.4342x³ - 3.7214x² + 8.8407x - 1.5163
        force = -0.0084 * x**4 + 0.4342 * x**3 - 3.7214 * x**2 + 8.8407 * x - 1.5163
    elif material_type == '后跟下部':  # 后跟下部（试样6）
        # F = -0.0140x⁴ + 0.5468x³ - 0.9791x² - 5.9761x + 8.0921
        force = -0.0140 * x**4 + 0.5468 * x**3 - 0.9791 * x**2 - 5.9761 * x + 8.0921
    elif material_type in ['无限伸缩', '鞋底']:  # 鞋底
        force = 0.0
    else:
        force = 0.0
    
    force = max(0.0, force)  # 确保力值不为负值
    
    return force

def get_material_type(group_name):
    """根据组名确定材质类型
    Args:
        group_name: 组名
    Returns:
        material_type: 材质类型
    """
    # 根据组名判断材质类型
    if '鞋尖' in group_name:
        return '试样2'
    elif '鞋面' in group_name:
        return '试样1'
    elif '鞋舌下' in group_name:
        return '试样4'
    elif '后跟' in group_name:
        return '后跟上部'
    elif '鞋舌上' in group_name:
        return '无限伸缩'
    elif '鞋底' in group_name:
        return '鞋底'
    else:
        return '无限伸缩'  # 默认为无限伸缩

def compute_group_regions(mesh, groups):
    """计算每个分区的空间区域信息
    Args:
        mesh: 原始网格模型
        groups: 分区信息字典
    Returns:
        group_regions: 包含每个分区空间信息的字典
    """
    group_regions = {}
    for group_name, group_info in groups.items():
        face_indices = group_info['faces']
        if not face_indices:
            continue
            
        # 获取该分区所有面片的顶点
        face_vertices = mesh.vertices[mesh.faces[face_indices]]
        vertices = face_vertices.reshape(-1, 3)
        
        # 计算边界框
        min_bounds = np.min(vertices, axis=0)
        max_bounds = np.max(vertices, axis=0)
        center = (min_bounds + max_bounds) / 2
        
        group_regions[group_name] = {
            'center': center,
            'min_bounds': min_bounds,
            'max_bounds': max_bounds,
            'material': group_info.get('material'),
            'object': group_info.get('object')
        }
    
    return group_regions

def assign_material_to_face(face_center, group_regions):
    """根据面片中心点分配材质类型
    Args:
        face_center: 面片中心点坐标
        group_regions: 分区空间信息字典
    Returns:
        material_type: 材质类型
    """
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
            assigned_material = get_material_type(group_name)
    
    return assigned_material

def display_model_with_distances_and_pressure(model, distances, valid_mask, renderer, shoe_model=None, group_regions=None):
    """显示距离云图和力值分布图
    Args:
        model: trimesh模型（足部）
        distances: 距离数组
        valid_mask: 有效距离的掩码
        renderer: VTK渲染器
        shoe_model: 简化后的鞋模型
        group_regions: 分区空间信息字典
    """
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
    
    # 添加标题文本（左侧视图）
    title_text1 = vtk.vtkTextActor()
    title_text1.SetInput("Distance Map / 距离云图")
    title_text1.GetTextProperty().SetFontSize(24)
    title_text1.GetTextProperty().SetBold(True)
    title_text1.GetTextProperty().SetColor(0, 0, 0)
    title_text1.SetPosition(20, 550)
    renderer.AddActor2D(title_text1)
    
    # 添加标题文本（右侧视图）
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
    
    # 获取每个面片所属的组
    face_groups = {}
    
    # 添加调试信息
    print("\n分区信息统计:")
    if group_regions:
        for group_name, region in group_regions.items():
            print(f"分区 {group_name}:")
            print(f"  - 边界范围:")
            print(f"    X: {region['min_bounds'][0]:.2f} 到 {region['max_bounds'][0]:.2f}")
            print(f"    Y: {region['min_bounds'][1]:.2f} 到 {region['max_bounds'][1]:.2f}")
            print(f"    Z: {region['min_bounds'][2]:.2f} 到 {region['max_bounds'][2]:.2f}")
    else:
        print("没有分区信息！")
    
    # 统计每个材质类型的面片数量
    material_counts = {}
    face_material_stats = []
    
    # 计算每个面片的力值
    force_values = []
    for face_idx, (face, dist, valid) in enumerate(zip(model.faces, distances, valid_mask)):
        if valid:
            # 计算面片中心点
            face_vertices = model.vertices[face]
            face_center = np.mean(face_vertices, axis=0)
            
            # 根据面片位置分配材质
            material_type = assign_material_to_face(face_center, group_regions) if group_regions else '无限伸缩'
            
            # 统计材质类型
            if material_type not in material_counts:
                material_counts[material_type] = 0
            material_counts[material_type] += 1
            
            # 计算力值
            force = convert_distance_to_pressure(dist, material_type)
            
            face_material_stats.append({
                'face_idx': face_idx,
                'distance': dist,
                'material': material_type,
                'force': force
            })
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
    
    print("\n材质分配统计:")
    for material, count in material_counts.items():
        print(f"{material}: {count}个面片")
    
    print("\n力值计算示例（前10个有效面片）:")
    for stat in face_material_stats[:10]:
        print(f"面片{stat['face_idx']}: 材质={stat['material']}, 距离={stat['distance']:.2f}mm, 力值={stat['force']:.2f}N")
    
    # 计算平均值
    valid_vertices = vertex_counts > 0
    vertex_distances[valid_vertices] /= vertex_counts[valid_vertices]
    vertex_forces[valid_vertices] /= vertex_counts[valid_vertices]
    
    # 打印顶点力值统计信息
    valid_forces = vertex_forces[~np.isnan(vertex_forces)]
    print("\n顶点力值统计信息:")
    if len(valid_forces) > 0:
        print(f"有效力值顶点数量: {len(valid_forces)}")
        print(f"顶点力值范围: {np.min(valid_forces):.4f}N 到 {np.max(valid_forces):.4f}N")
        print(f"顶点平均力值: {np.mean(valid_forces):.4f}N")
    else:
        print("警告：没有有效的顶点力值！")
    
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
        max_force = 100.0  # 默认最大力值（N）
    
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
        t = i / 511.0  # 归一化到[0,1]
        
        # jet颜色方案的实现
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
        
        # 确保颜色值在[0,1]范围内
        r = max(0.0, min(1.0, r))
        g = max(0.0, min(1.0, g))
        b = max(0.0, min(1.0, b))
        
        lut.SetTableValue(i, r, g, b, 1.0)
    
    # 创建力值颜色表（使用twilight颜色方案）
    force_lut = vtk.vtkLookupTable()
    force_lut.SetNumberOfTableValues(512)
    force_lut.SetTableRange(0, max_force)
    
    for i in range(512):
        t = i / 511.0
        
        # twilight颜色方案
        if t < 0.25:  # 深紫到蓝
            r = 0.85 - t * 2
            g = 0.85 - t * 2
            b = 0.85
        elif t < 0.5:  # 蓝到绿
            r = 0.35 - (t - 0.25) * 1.4
            g = 0.35 + (t - 0.25) * 1.4
            b = 0.85 - (t - 0.25) * 1.4
        elif t < 0.75:  # 绿到橙
            r = 0.35 + (t - 0.5) * 2.6
            g = 0.7 - (t - 0.5) * 1.4
            b = 0.15
        else:  # 橙到红
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
            print(f"鼠标回调函数出错: {str(e)}")
            text_actor.SetInput("")
            force_text_actor.SetInput("")
            renderer.GetRenderWindow().Render()
    
    # 添加事件观察者
    interactor = renderer.GetRenderWindow().GetInteractor()
    interactor.AddObserver("MouseMoveEvent", mouse_move_callback)
    
    # 同步两个视图的相机
    renderer2.SetActiveCamera(renderer.GetActiveCamera())
    
    return actor

def read_obj_groups(file_path):
    """读取OBJ文件中的曲面集合信息
    Args:
        file_path: OBJ文件路径
    Returns:
        groups: 曲面集合信息字典
    """
    groups = {}
    current_group = None
    current_object = None
    current_material = None
    face_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('g '):  # 组定义
                current_group = line[2:].strip()
                if current_group:
                    if current_group not in groups:
                        groups[current_group] = {
                            'faces': [],
                            'material': current_material,
                            'object': current_object
                        }
            elif line.startswith('o '):  # 对象定义
                current_object = line[2:].strip()
            elif line.startswith('usemtl '):  # 材质定义
                current_material = line[7:].strip()
                if current_group and current_group in groups:
                    groups[current_group]['material'] = current_material
            elif line.startswith('f '):  # 面片
                if current_group and current_group in groups:
                    groups[current_group]['faces'].append(face_count)
                face_count += 1
    
    return groups

def main():
    # 加载模型
    print("加载模型...")
    foot_model = trimesh.load(r"C:\Users\wufan\Desktop\适配测试\foot-R.stl")
    shoe_model = trimesh.load(r"C:\Users\wufan\Desktop\适配测试\shoe-R.obj")
    
    # 检查并打印模型信息
    print("\n检查模型加载情况:")
    groups = {}
    if isinstance(shoe_model, trimesh.Scene):
        print("鞋模型是一个Scene对象，包含以下几何体：")
        total_vertices = 0
        total_faces = 0
        face_offset = 0
        
        for name, geometry in shoe_model.geometry.items():
            print(f"\n几何体名称: {name}")
            print(f"  顶点数: {len(geometry.vertices)}")
            print(f"  面片数: {len(geometry.faces)}")
            
            # 构建groups字典
            face_indices = list(range(face_offset, face_offset + len(geometry.faces)))
            groups[name] = {
                'faces': face_indices,
                'material': None,
                'object': name
            }
            
            face_offset += len(geometry.faces)
            total_vertices += len(geometry.vertices)
            total_faces += len(geometry.faces)
            
            if hasattr(geometry, 'visual') and hasattr(geometry.visual, 'material'):
                material = geometry.visual.material
                print(f"  材质名称: {material.name if hasattr(material, 'name') else '未命名'}")
        
        print(f"\n总计:")
        print(f"  总顶点数: {total_vertices}")
        print(f"  总面片数: {total_faces}")
        
        # 合并所有几何体
        vertices_list = []
        faces_list = []
        offset = 0
        for geometry in shoe_model.geometry.values():
            vertices_list.append(geometry.vertices)
            faces_list.append(geometry.faces + offset)
            offset += len(geometry.vertices)
        
        # 创建合并后的网格
        shoe_model = trimesh.Trimesh(
            vertices=np.vstack(vertices_list),
            faces=np.vstack(faces_list)
        )
        print(f"\n合并后的模型:")
        print(f"  顶点数: {len(shoe_model.vertices)}")
        print(f"  面片数: {len(shoe_model.faces)}")
    
    # 打印groups信息
    print("\n检查鞋内腔模型曲面集合信息...")
    if groups:
        print("找到曲面集合信息：")
        for group_name, group_info in groups.items():
            print(f"  - {group_name}:")
            print(f"    面片数量: {len(group_info['faces'])}")
            if group_info['material']:
                print(f"    材质: {group_info['material']}")
            if group_info['object']:
                print(f"    对象: {group_info['object']}")
    else:
        print("未找到曲面集合信息")
    
    # 计算并显示原始模型信息
    foot_area = foot_model.area
    foot_faces = len(foot_model.faces)
    foot_density = foot_faces / foot_area
    
    print("\n原始模型信息:")
    print("足部模型:")
    print(f"  - 面片数: {foot_faces}")
    print(f"  - 表面积: {foot_area:.2f} 平方毫米")
    print(f"  - 网格密度: {foot_density:.4f} 面片/平方毫米")
    
    shoe_area = shoe_model.area
    shoe_faces = len(shoe_model.faces)
    shoe_density = shoe_faces / shoe_area
    
    print("\n鞋内腔模型:")
    print(f"  - 面片数: {shoe_faces}")
    print(f"  - 表面积: {shoe_area:.2f} 平方毫米")
    print(f"  - 网格密度: {shoe_density:.4f} 面片/平方毫米")
    
    # 简化鞋内腔模型
    print("\n简化鞋内腔模型...")
    simplified_shoe = simplify_to_target_density(shoe_model, foot_density)
    
    # 显示简化后的信息
    simplified_area = simplified_shoe.area
    simplified_faces = len(simplified_shoe.faces)
    simplified_density = simplified_faces / simplified_area
    
    print("\n简化后的鞋内腔模型:")
    print(f"  - 面片数: {simplified_faces}")
    print(f"  - 表面积: {simplified_area:.2f} 平方毫米")
    print(f"  - 网格密度: {simplified_density:.4f} 面片/平方毫米")
    
    # 计算原始模型的分区空间信息
    print("\n计算分区空间信息...")
    group_regions = compute_group_regions(shoe_model, groups)
    
    # 计算距离
    print("\n计算距离...")
    distances, valid_mask = compute_distances(simplified_shoe, foot_model)
    
    # 显示统计信息
    valid_distances = distances[valid_mask]
    print(f"总面片数: {len(distances)}")
    print(f"有效面片数: {np.sum(valid_mask)}")
    if len(valid_distances) > 0:
        print(f"距离范围: {np.min(valid_distances):.2f}mm 到 {np.max(valid_distances):.2f}mm")
    
    # 创建VTK渲染器和窗口
    renderer = vtk.vtkRenderer()
    window = vtk.vtkRenderWindow()
    window.AddRenderer(renderer)
    window.SetSize(1600, 800)
    
    # 创建交互器
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    
    # 设置背景颜色为白色
    renderer.SetBackground(1, 1, 1)
    
    print("\n显示距离云图和力值分布...")
    # 显示带有距离云图和力值分布的模型，传入分区空间信息
    display_model_with_distances_and_pressure(foot_model, distances, valid_mask, renderer, simplified_shoe, group_regions)
    
    # 设置相机位置
    renderer.ResetCamera()
    camera = renderer.GetActiveCamera()
    camera.Elevation(20)
    camera.Azimuth(20)
    
    # 开始显示
    window.Render()
    interactor.Start()

if __name__ == "__main__":
    main() 