import numpy as np
import trimesh
from scipy.spatial import cKDTree
import logging
import traceback

def simplify_model(model, target_faces=5000):
    """简化模型网格
    Args:
        model: trimesh模型
        target_faces: 目标面片数量
    Returns:
        simplified: 简化后的模型
    """
    try:
        logging.info(f"开始简化模型，原始面片数: {len(model.faces)}")
        
        # 如果面片数已经小于目标数量，直接返回原模型
        if len(model.faces) <= target_faces:
            logging.info("模型面片数已经小于目标数量，无需简化")
            return model
        
        # 计算简化比例
        ratio = target_faces / len(model.faces)
        logging.info(f"简化比例: {ratio:.2f}")
        
        try:
            # 尝试使用快速简化方法
            simplified = model.simplify_quadric_decimation(target_faces)
            logging.info("使用quadric_decimation方法简化成功")
        except:
            # 如果快速简化失败，使用基础简化方法
            logging.info("quadric_decimation失败，尝试使用基础简化方法")
            simplified = model.simplify_vertex_clustering(
                ratio,
                preserve_volume=True
            )
            logging.info("使用vertex_clustering方法简化成功")
        
        logging.info(f"简化完成，简化后面片数: {len(simplified.faces)}")
        return simplified
        
    except Exception as e:
        logging.error(f"网格简化失败: {str(e)}")
        logging.error(traceback.format_exc())
        # 如果简化失败，返回原始模型
        logging.warning("简化失败，返回原始模型")
        return model

def compute_distances(shoe_mesh, foot_mesh, max_angle=40):
    """
    计算鞋内腔模型到足部模型的距离
    
    参数:
        shoe_mesh: 鞋内腔模型
        foot_mesh: 足部模型
        max_angle: 最大法向量夹角（度）
    
    返回:
        distances: 距离数组
        valid_mask: 有效距离掩码
    """
    try:
        # 计算鞋内腔模型的面片中心点和法向量
        shoe_face_centers = np.mean(shoe_mesh.vertices[shoe_mesh.faces], axis=1)
        shoe_face_normals = shoe_mesh.face_normals
        
        # 创建鞋内腔模型的KD树
        shoe_tree = cKDTree(shoe_face_centers)
        
        # 计算足部模型的面片中心点和法向量
        foot_face_centers = np.mean(foot_mesh.vertices[foot_mesh.faces], axis=1)
        foot_face_normals = foot_mesh.face_normals
        
        # 初始化距离数组和有效掩码
        distances = np.zeros(len(foot_face_centers))
        valid_mask = np.zeros(len(foot_face_centers), dtype=bool)
        
        # 计算每个足部面片到最近鞋内腔面片的距离
        for i, (center, normal) in enumerate(zip(foot_face_centers, foot_face_normals)):
            # 查找最近的鞋内腔面片
            dist, idx = shoe_tree.query(center)
            
            # 计算法向量夹角
            angle = np.degrees(np.arccos(np.clip(
                np.dot(normal, shoe_face_normals[idx]), -1.0, 1.0)))
            
            # 如果夹角小于阈值，则记录距离
            if angle <= max_angle:
                distances[i] = dist
                valid_mask[i] = True
        
        logging.info(f"距离计算完成: {np.sum(valid_mask)}/{len(valid_mask)} 个有效距离")
        return distances, valid_mask
        
    except Exception as e:
        logging.error(f"距离计算失败: {str(e)}")
        raise

def convert_distance_to_pressure(distance, material_type):
    """
    将距离转换为压力值
    
    参数:
        distance: 距离值（mm）
        material_type: 材质类型
    
    返回:
        压力值（N）
    """
    try:
        # 根据材质类型选择不同的拟合方程
        if material_type == '鞋面':
            # 鞋面材料拟合方程
            return max(0, 100 * np.exp(-0.1 * distance))
        elif material_type == '鞋尖':
            # 鞋尖材料拟合方程
            return max(0, 150 * np.exp(-0.15 * distance))
        elif material_type == '鞋舌':
            # 鞋舌材料拟合方程
            return max(0, 80 * np.exp(-0.08 * distance))
        elif material_type == '后跟':
            # 后跟材料拟合方程
            return max(0, 200 * np.exp(-0.2 * distance))
        elif material_type == '鞋底':
            # 鞋底材料拟合方程
            return max(0, 300 * np.exp(-0.25 * distance))
        else:
            # 默认材料拟合方程
            return max(0, 120 * np.exp(-0.12 * distance))
            
    except Exception as e:
        logging.error(f"压力转换失败: {str(e)}")
        return 0

def get_material_type(material_name):
    """
    根据材质名称获取材质类型
    
    参数:
        material_name: 材质名称
    
    返回:
        材质类型
    """
    try:
        # 材质名称到类型的映射
        material_map = {
            'upper': '鞋面',
            'toe': '鞋尖',
            'tongue': '鞋舌',
            'heel': '后跟',
            'sole': '鞋底'
        }
        
        # 查找材质类型
        for key, value in material_map.items():
            if key in material_name.lower():
                return value
        
        # 如果找不到匹配的类型，返回默认类型
        return '鞋面'
        
    except Exception as e:
        logging.error(f"获取材质类型失败: {str(e)}")
        return '鞋面' 