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
    distances[distances > 25.0] = np.nan  # 大于25mm的设为NaN
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
    
    # 设置颜色映射（使用HSV颜色空间）
    for i in range(512):
        t = i / 511.0
        # 将范围重新映射到[-25, 25]
        value = min_dist + t * (max_dist - min_dist)
        
        # 归一化到[0,1]区间
        normalized_t = (value - min_dist) / (max_dist - min_dist)
        
        # 使用HSV颜色空间
        if normalized_t < 0.5:  # 负值区域（干涉）：从红到白
            t2 = normalized_t * 2
            h = 0  # 红色
            s = 1.0 - t2  # 从全饱和到无饱和
            v = 1.0  # 保持最大亮度
        else:  # 正值区域（间隙）：从白到蓝
            t2 = (normalized_t - 0.5) * 2
            h = 240  # 蓝色
            s = t2  # 从无饱和到全饱和
            v = 1.0  # 保持最大亮度
        
        # 将HSV转换为RGB
        if s == 0:
            r, g, b = v, v, v  # 白色
        else:
            h = h / 360.0  # 归一化到[0,1]区间
            i = int(h * 6)
            f = h * 6 - i
            p = v * (1 - s)
            q = v * (1 - f * s)
            t = v * (1 - (1 - f) * s)
            
            if i % 6 == 0:
                r, g, b = v, t, p
            elif i % 6 == 1:
                r, g, b = q, v, p
            elif i % 6 == 2:
                r, g, b = p, v, t
            elif i % 6 == 3:
                r, g, b = p, q, v
            elif i % 6 == 4:
                r, g, b = t, p, v
            else:
                r, g, b = v, p, q
        
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
    # 计算当前网格密度
    current_area = mesh.area
    current_faces = len(mesh.faces)
    current_density = current_faces / current_area
    
    # 计算目标面片数
    target_faces = int(current_area * target_density)
    
    # 计算简化比例
    ratio = 1.0 - (target_faces / current_faces)
    
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
    
    # 创建简化过滤器
    decimate = vtk.vtkDecimatePro()
    decimate.SetInputData(polydata)
    decimate.SetTargetReduction(ratio)
    decimate.PreserveTopologyOn()
    decimate.BoundaryVertexDeletionOff()
    decimate.Update()
    
    # 获取简化后的模型
    simplified = decimate.GetOutput()
    
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
    
    # 创建新的trimesh对象
    simplified_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # 打印简化信息
    print(f"网格密度: {current_density:.4f} -> {len(faces)/simplified_mesh.area:.4f} 面片/平方毫米")
    print(f"网格面片: {current_faces} -> {len(faces)} 个")
    print(f"表面积: {current_area:.2f} 平方毫米")
    
    return simplified_mesh

def compute_distances(shoe_mesh, foot_mesh, max_angle=90):
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
    max_distance = 25.0  # 最大距离阈值(mm)
    
    # 对每个足部模型面进行处理
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
        
        # 计算有符号距离
        signed_dist = dist
        if np.dot(direction, shoe_face_normals[idx]) > 0:
            signed_dist = -dist  # 干涉情况
        
        distances[i] = signed_dist
        valid_mask[i] = True
    
    # 将超出范围的距离设置为NaN
    distances[distances > 25.0] = np.nan  # 大于25mm的设为NaN
    distances[distances < -10] = np.nan  # 小于-12.5mm的区域设为NaN，显示为浅灰色
    
    # 更新有效掩码
    valid_mask = ~np.isnan(distances)
    
    # 打印有效范围内的统计信息
    valid_distances = distances[valid_mask]
    print(f"总面片数: {len(distances)}")
    print(f"有效面片数: {np.sum(valid_mask)}")
    if len(valid_distances) > 0:
        print(f"距离范围: {np.min(valid_distances):.2f}mm 到 {np.max(valid_distances):.2f}mm")
    
    return distances, valid_mask

def display_model_with_distances(model, distances, valid_mask, renderer, shoe_model=None):
    """使用jet颜色方案显示距离云图和距离向量
    Args:
        model: trimesh模型（足部）
        distances: 距离数组
        valid_mask: 有效距离的掩码
        renderer: VTK渲染器
        shoe_model: 简化后的鞋内腔模型
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
    
    # 添加距离数据
    distance_data = vtk.vtkFloatArray()
    distance_data.SetName("Distance")
    
    # 将面的数据插值到顶点
    vertex_distances = np.full(len(model.vertices), np.nan)
    vertex_counts = np.zeros(len(model.vertices))
    
    for face_idx, (face, dist, valid) in enumerate(zip(model.faces, distances, valid_mask)):
        if valid:
            for vertex_idx in face:
                if np.isnan(vertex_distances[vertex_idx]):
                    vertex_distances[vertex_idx] = dist
                    vertex_counts[vertex_idx] = 1
                else:
                    vertex_distances[vertex_idx] += dist
                    vertex_counts[vertex_idx] += 1
    
    # 计算平均值
    valid_vertices = vertex_counts > 0
    vertex_distances[valid_vertices] /= vertex_counts[valid_vertices]
    
    # 添加到VTK数组
    for dist in vertex_distances:
        distance_data.InsertNextValue(float(dist))
    
    polydata.GetPointData().SetScalars(distance_data)
    
    # 使用实际的距离范围
    min_dist = -12.5  # 修改最小值为-12.5mm
    max_dist = 25.0
    
    # 创建映射器
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.SetScalarRange(min_dist, max_dist)
    
    # 创建颜色查找表
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
    scalar_bar.SetTitle("距离 (mm)")
    scalar_bar.SetWidth(0.08)
    scalar_bar.SetHeight(0.6)
    scalar_bar.SetPosition(0.92, 0.2)
    scalar_bar.SetNumberOfLabels(5)
    scalar_bar.GetLabelTextProperty().SetColor(0, 0, 0)
    scalar_bar.GetTitleTextProperty().SetColor(0, 0, 0)
    
    # 添加到渲染器
    renderer.AddActor(actor)
    renderer.AddActor2D(scalar_bar)
    
    # 添加距离向量连线
    if shoe_model is not None:
        lines = vtk.vtkAppendPolyData()
        
        # 获取足部面的中心点
        foot_face_centers = np.mean(model.vertices[model.faces], axis=1)
        
        # 获取鞋内腔面的中心点
        shoe_face_centers = np.mean(shoe_model.vertices[shoe_model.faces], axis=1)
        shoe_face_normals = -shoe_model.face_normals  # 取反使其指向内部
        
        # 构建KD树
        shoe_tree = cKDTree(shoe_face_centers)
        
        # 创建标量数组存储距离值
        scalar_array = vtk.vtkFloatArray()
        scalar_array.SetName("Distance")
        
        # 对每个有效的足部面创建连线
        for i, (center, dist, is_valid) in enumerate(zip(foot_face_centers, distances, valid_mask)):
            if not is_valid:
                continue
                
            # 查找最近的鞋内腔面
            _, idx = shoe_tree.query(center, k=1)
            shoe_center = shoe_face_centers[idx]
            
            # 创建线段
            line = vtk.vtkLineSource()
            line.SetPoint1(center[0], center[1], center[2])
            line.SetPoint2(shoe_center[0], shoe_center[1], shoe_center[2])
            
            # 添加到lines中
            lines.AddInputConnection(line.GetOutputPort())
            
            # 添加距离值用于着色
            scalar_array.InsertNextValue(float(dist))
        
        # 创建线段的PolyData
        line_polydata = lines.GetOutput()
        line_polydata.GetPointData().SetScalars(scalar_array)
        
        # 创建线段的mapper
        line_mapper = vtk.vtkPolyDataMapper()
        line_mapper.SetInputConnection(lines.GetOutputPort())
        line_mapper.SetLookupTable(lut)
        line_mapper.SetScalarRange(min_dist, max_dist)
        
        # 创建线段actor
        line_actor = vtk.vtkActor()
        line_actor.SetMapper(line_mapper)
        line_actor.GetProperty().SetLineWidth(1)  # 设置线宽
        
        # 添加到渲染器
        renderer.AddActor(line_actor)
    
    return actor

def main():
    # 加载模型
    print("加载模型...")
    foot_model = trimesh.load(r"C:\Users\wufan\Desktop\适配测试\foot-R.stl")
    shoe_model = trimesh.load(r"C:\Users\wufan\Desktop\适配测试\shoe-R.obj")
    
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
    
    # 创建交互器
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(window)
    
    # 设置背景颜色为白色
    renderer.SetBackground(1, 1, 1)
    
    print("\n显示距离云图...")
    # 显示带有距离云图的模型，并传入简化后的鞋模型用于显示连接线
    display_model_with_distances(foot_model, distances, valid_mask, renderer, simplified_shoe)
    
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