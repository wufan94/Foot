import os
import sys
import vtk
import trimesh
import numpy as np
import argparse
from deviation_analysis_backup import (
    compute_distances, 
    compute_group_regions, 
    display_model_with_distances_and_pressure, 
    simplify_to_target_density
)

def run_deviation_analysis(foot_path, shoe_path, max_angle=40):
    """运行偏差分析
    
    Args:
        foot_path: 足部模型路径
        shoe_path: 鞋内腔模型路径
        max_angle: 最大法向量角度（度）
    """
    print("加载模型...")
    foot_model = trimesh.load(foot_path)
    shoe_model = trimesh.load(shoe_path)
    
    # 检查模型
    print("\n模型信息:")
    print(f"足部模型: {len(foot_model.vertices)} 个顶点, {len(foot_model.faces)} 个面")
    print(f"鞋内腔模型: {len(shoe_model.vertices)} 个顶点, {len(shoe_model.faces)} 个面")
    
    # 处理鞋内腔模型
    groups = {}
    if isinstance(shoe_model, trimesh.Scene):
        print("\n鞋内腔模型是一个Scene对象，包含以下几何体：")
        total_vertices = 0
        total_faces = 0
        face_offset = 0
        
        for name, geometry in shoe_model.geometry.items():
            print(f"几何体名称: {name}")
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
    
    # 计算网格密度
    foot_density = len(foot_model.faces) / foot_model.area
    shoe_density = len(shoe_model.faces) / shoe_model.area
    
    print("\n网格密度:")
    print(f"足部模型: {foot_density:.4f} 面片/平方毫米")
    print(f"鞋内腔模型: {shoe_density:.4f} 面片/平方毫米")
    
    # 简化鞋内腔模型
    print("\n简化鞋内腔模型...")
    simplified_shoe = simplify_to_target_density(shoe_model, foot_density)
    
    # 计算分区空间信息
    print("\n计算分区空间信息...")
    group_regions = compute_group_regions(shoe_model, groups)
    
    # 计算距离
    print("\n计算距离...")
    distances, valid_mask = compute_distances(simplified_shoe, foot_model, max_angle)
    
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
    parser = argparse.ArgumentParser(description="足部与鞋内腔模型偏差分析")
    parser.add_argument("--foot", type=str, help="足部模型路径", 
                        default=r"C:\Users\wufan\Desktop\适配测试\foot-R.stl")
    parser.add_argument("--shoe", type=str, help="鞋内腔模型路径", 
                        default=r"C:\Users\wufan\Desktop\适配测试\shoe-R.obj")
    parser.add_argument("--angle", type=int, help="最大法向量角度（度）", default=40)
    
    args = parser.parse_args()
    
    # 检查路径是否存在
    if not os.path.exists(args.foot):
        print(f"错误: 找不到足部模型文件 '{args.foot}'")
        sys.exit(1)
        
    if not os.path.exists(args.shoe):
        print(f"错误: 找不到鞋内腔模型文件 '{args.shoe}'")
        sys.exit(1)
    
    run_deviation_analysis(args.foot, args.shoe, args.angle) 