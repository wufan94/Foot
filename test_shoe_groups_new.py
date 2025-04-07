import trimesh
import vtk

def read_obj_materials(file_path):
    """读取OBJ文件中的材质信息"""
    materials = {}
    current_material = None
    face_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        # 首先计算有多少个面片
        total_faces = sum(1 for line in lines if line.strip().startswith('f '))
        
        # 创建一个数组来存储每个面片的材质
        face_materials = [None] * total_faces
        
        # 遍历文件，记录每个面片的材质
        for line in lines:
            line = line.strip()
            if line.startswith('usemtl '):  # 材质使用
                current_material = line[7:].strip()
                if current_material not in materials:
                    materials[current_material] = []
            elif line.startswith('f '):  # 面片
                if current_material:
                    face_materials[face_count] = current_material
                face_count += 1
    
    # 构建材质到面片的映射
    material_faces = {material: [] for material in materials}
    for face_idx, material in enumerate(face_materials):
        if material:
            material_faces[material].append(face_idx)
    
    return material_faces

def display_model_with_materials(meshes, renderer):
    """显示带有材质颜色的模型"""
    if not isinstance(meshes, list):
        meshes = [meshes]
    
    # 定义颜色列表
    colors = [
        (255, 0, 0),    # 红色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 蓝色
        (255, 255, 0),  # 黄色
        (255, 0, 255),  # 紫色
        (0, 255, 255),  # 青色
        (255, 128, 0),  # 橙色
        (128, 255, 0),  # 黄绿色
        (0, 128, 255),  # 浅蓝色
        (255, 0, 128),  # 粉红色
        (128, 128, 0),  # 棕色
        (128, 0, 128),  # 深紫色
        (0, 128, 128),  # 深青色
    ]
    
    actors = []
    # 为每个网格分配一个颜色
    for i, mesh in enumerate(meshes):
        color = colors[i % len(colors)]
        print(f"网格 {i + 1} 使用颜色: {color}, 面片数量: {len(mesh.faces)}")
        actor = create_mesh_actor(mesh, color)
        renderer.AddActor(actor)
        actors.append(actor)
    return actors

def create_mesh_actor(mesh, color):
    """为单个网格创建VTK actor"""
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
    colors = vtk.vtkUnsignedCharArray()
    colors.SetName("Colors")
    colors.SetNumberOfComponents(3)
    colors.SetNumberOfTuples(len(mesh.faces))
    
    # 设置网格颜色
    for i in range(len(mesh.faces)):
        colors.SetTuple3(i, *color)
    
    polydata.GetCellData().SetScalars(colors)
    
    # 创建映射器和actor
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    # 设置材质属性
    prop = actor.GetProperty()
    prop.SetAmbient(0.3)
    prop.SetDiffuse(0.7)
    prop.SetSpecular(0.5)
    prop.SetSpecularPower(20)
    
    return actor

def filter_meshes(meshes, mesh_names, max_count=13):
    """筛选保留面片数量最多的网格
    Args:
        meshes: 网格列表
        mesh_names: 网格名称列表
        max_count: 最大保留数量
    Returns:
        filtered_meshes: 筛选后的网格列表
        filtered_names: 筛选后的网格名称列表
    """
    if len(meshes) <= max_count:
        return meshes, mesh_names
    
    # 创建(面片数量, 网格, 名称)的元组列表
    mesh_info = [(len(mesh.faces), mesh, name) for mesh, name in zip(meshes, mesh_names)]
    # 按面片数量降序排序
    mesh_info.sort(reverse=True, key=lambda x: x[0])
    
    # 取前max_count个
    filtered_info = mesh_info[:max_count]
    print(f"\n由于网格数量超过{max_count}个，删除以下网格：")
    for face_count, _, name in mesh_info[max_count:]:
        print(f"删除网格 '{name}', 面片数量: {face_count}")
    
    # 解压缩筛选后的信息
    _, filtered_meshes, filtered_names = zip(*filtered_info)
    return list(filtered_meshes), list(filtered_names)

def main():
    # 设置文件路径
    file_path = r"C:\Users\wufan\Documents\MyFiles\3 研究院文件\项目文件\足鞋工效2024\2项目研发相关材料\3虚拟适配相关材料\虚拟适配250325\Shoe_Model\42L.obj"
    
    try:
        # 读取材质信息
        print("\n正在读取材质信息...")
        material_faces = read_obj_materials(file_path)
        
        if material_faces:
            print("\n找到以下材质：")
            for material_name, face_indices in material_faces.items():
                print(f"材质名称: {material_name}, 面片数量: {len(face_indices)}")
        else:
            print("未找到材质信息")
            return
        
        # 加载模型
        print("\n正在加载模型...")
        scene = trimesh.load(file_path)
        
        # 获取所有网格
        if isinstance(scene, trimesh.Scene):
            print("\n检测到场景对象，提取所有网格...")
            meshes = list(scene.geometry.values())
            mesh_names = list(scene.geometry.keys())
            
            # 筛选网格
            meshes, mesh_names = filter_meshes(meshes, mesh_names)
            
            total_vertices = sum(len(mesh.vertices) for mesh in meshes)
            total_faces = sum(len(mesh.faces) for mesh in meshes)
            print(f"\n保留 {len(meshes)} 个网格:")
            for i, (name, mesh) in enumerate(zip(mesh_names, meshes)):
                print(f"网格 {i + 1}: 名称 = {name}, 面片数量 = {len(mesh.faces)}")
        else:
            meshes = [scene]
            mesh_names = ["default"]
            total_vertices = len(scene.vertices)
            total_faces = len(scene.faces)
            
        print(f"\n模型加载完成: {total_vertices} 个顶点, {total_faces} 个面片")
        
        # 创建渲染器
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(1.0, 1.0, 1.0)  # 白色背景
        
        render_window = vtk.vtkRenderWindow()
        render_window.AddRenderer(renderer)
        render_window.SetSize(1024, 768)
        
        # 显示模型
        print("\n正在显示模型...")
        actors = display_model_with_materials(meshes, renderer)
        
        # 设置相机
        renderer.ResetCamera()
        camera = renderer.GetActiveCamera()
        camera.Elevation(30)
        camera.Azimuth(30)
        
        # 开始显示
        interactor = vtk.vtkRenderWindowInteractor()
        interactor.SetRenderWindow(render_window)
        interactor.Initialize()
        render_window.Render()
        interactor.Start()
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 