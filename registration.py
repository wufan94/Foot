import numpy as np
from scipy.spatial import KDTree
import logging

class ModelRegistration:
    def __init__(self):
        self.transform_matrix = None
        
    def register_models(self, source_model, target_model, sample_size=30000, max_iterations=200):
        """配准两个模型"""
        try:
            print(f"源模型顶点数: {len(source_model.vertices)}")
            print(f"目标模型顶点数: {len(target_model.vertices)}")
            
            # 1. 采样点进行ICP
            print("\n1. 采样点进行配准...")
            source_points = self._sample_points(source_model, sample_size)
            target_points = self._sample_points(target_model, sample_size)
            
            # 2. 构建KD树
            target_tree = KDTree(target_points)
            current_points = source_points.copy()
            current_transform = np.eye(4)
            
            # 3. ICP迭代
            prev_error = float('inf')
            convergence_count = 0
            convergence_threshold = 0.00001
            max_convergence_count = 10
            
            print("\n开始ICP迭代...")
            for iteration in range(max_iterations):
                # 找到最近点对
                distances, indices = target_tree.query(current_points)
                corresponding_points = target_points[indices]
                
                # 计算当前误差
                current_error = np.mean(distances)
                print(f"迭代 {iteration + 1}, 误差: {current_error:.3f} mm")
                
                # 检查收敛
                if abs(prev_error - current_error) < convergence_threshold:
                    convergence_count += 1
                    if convergence_count >= max_convergence_count:
                        print(f"配准收敛，连续{max_convergence_count}次误差变化小于{convergence_threshold}")
                        break
                else:
                    convergence_count = 0
                prev_error = current_error
                
                # 计算变换
                transform = self._compute_transform(current_points, corresponding_points)
                current_transform = np.dot(transform, current_transform)
                current_points = self._apply_transform(current_points, transform)
                
                # 每50次迭代重新采样点
                if iteration % 50 == 49:
                    print("重新采样点...")
                    source_points = self._sample_points(source_model, sample_size)
                    current_points = self._apply_transform(source_points, current_transform)
            
            self.transform_matrix = current_transform
            return {'transform_matrix': current_transform}
            
        except Exception as e:
            logging.error(f"模型配准失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
            
    def _sample_points(self, model, n_points):
        """从模型表面采样点"""
        if hasattr(model, 'sample'):
            points, _ = model.sample(n_points)
            return points
        else:
            if len(model.vertices) > n_points:
                indices = np.random.choice(len(model.vertices), n_points, replace=False)
                return model.vertices[indices]
            return model.vertices
            
    def _compute_transform(self, source, target):
        """计算两组点之间的变换矩阵"""
        # 计算质心
        source_centroid = np.mean(source, axis=0)
        target_centroid = np.mean(target, axis=0)
        
        # 将点云中心移到原点
        source_centered = source - source_centroid
        target_centered = target - target_centroid
        
        # 计算协方差矩阵
        H = np.dot(source_centered.T, target_centered)
        
        # SVD分解
        U, _, Vt = np.linalg.svd(H)
        
        # 计算旋转矩阵
        R = np.dot(Vt.T, U.T)
        
        # 如果行列式为负，需要修正以确保是右手坐标系
        if np.linalg.det(R) < 0:
            Vt[-1,:] *= -1
            R = np.dot(Vt.T, U.T)
        
        # 计算平移向量
        t = target_centroid - np.dot(source_centroid, R)
        
        # 构建变换矩阵
        transform = np.eye(4)
        transform[:3, :3] = R
        transform[:3, 3] = t
        
        return transform
        
    def _apply_transform(self, points, transform):
        """应用变换矩阵到点云"""
        homogeneous = np.hstack([points, np.ones((len(points), 1))])
        transformed = np.dot(homogeneous, transform.T)
        return transformed[:, :3]
        
    def register_gur_to_target(self, gur_model, target_model):
        """仅用于PPT数据映射的GUR模型配准"""
        try:
            # 使用完整的配准方法
            registration_info = self.register_models(gur_model, target_model)
            if registration_info is None:
                return False
            
            # 保存变换矩阵供PPT点使用
            self.transform_matrix = registration_info['transform_matrix']
            return True
            
        except Exception as e:
            logging.error(f"GUR模型配准失败: {str(e)}")
            return False
            
    def transform_ppt_to_target(self, ppt_coords, target_model):
        """将PPT点转换到目标模型坐标系"""
        try:
            if self.transform_matrix is None:
                raise ValueError("未找到变换矩阵，请先执行GUR模型配准")
            
            # 将PPT坐标转换为齐次坐标
            homogeneous_coords = np.ones((len(ppt_coords), 4))
            homogeneous_coords[:,:3] = ppt_coords
            
            # 应用变换
            transformed_coords = np.dot(homogeneous_coords, self.transform_matrix.T)
            
            return {
                'positions': transformed_coords[:,:3],  # 转换后的3D坐标
                'transform': self.transform_matrix      # 保存使用的变换矩阵
            }
            
        except Exception as e:
            logging.error(f"PPT点转换失败: {str(e)}")
            return None