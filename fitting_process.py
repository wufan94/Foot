import numpy as np
from scipy.spatial import KDTree
import logging

class FittingAnalyzer:
    def __init__(self):
        """初始化适配分析器"""
        self.material_factor = 1.0  # 材质影响因子
        self.ppt_factor = 0.3      # PPT影响因子
    
    def calculate_distances(self, foot_model, shoe_model, ppt_values=None):
        """计算适配距离"""
        try:
            # 1. 计算基础距离
            base_distances = self._compute_base_distances(foot_model, shoe_model)
            
            # 2. 根据材质属性修正
            material_distances = self._apply_material_effect(base_distances, shoe_model)
            
            # 3. 根据PPT数据修正
            final_distances = material_distances.copy()
            if ppt_values is not None:
                # 标准化PPT值到0-1范围
                ppt_normalized = (ppt_values - np.min(ppt_values)) / (np.max(ppt_values) - np.min(ppt_values))
                
                # PPT影响系数：耐受阈值低的地方（PPT小），干涉距离的影响要放大
                ppt_effect = (1 - ppt_normalized) * self.ppt_factor
                
                # 只对干涉区域（负值）应用PPT影响
                mask = final_distances < 0
                final_distances[mask] *= (1 + ppt_effect[mask])
            
            return {
                'base': base_distances,        # 基础适配距离
                'material': material_distances, # 考虑材质后的距离
                'final': final_distances       # 考虑PPT后的最终距离
            }
            
        except Exception as e:
            logging.error(f"距离计算失败: {str(e)}")
            return None
    
    def analyze_results(self, distances, foot_model):
        """分析适配结果"""
        try:
            if not isinstance(distances, dict):
                raise ValueError(f"无效的距离数据类型: {type(distances)}")
            
            stats = {}
            for dist_type, dist in distances.items():
                if not isinstance(dist, np.ndarray):
                    logging.warning(f"无效的距离数据类型 {dist_type}: {type(dist)}")
                    continue
                
                if len(dist) == 0:
                    logging.warning(f"空的距离数据: {dist_type}")
                    continue
                
                # 过滤掉无效值
                valid_dist = dist[~np.isnan(dist) & ~np.isinf(dist)]
                if len(valid_dist) == 0:
                    logging.warning(f"{dist_type} 没有有效数据")
                    continue
                
                try:
                    interference_mask = valid_dist < 0
                    gap_mask = valid_dist > 0
                    
                    stats[dist_type] = {
                        'max_interference': float(-np.min(valid_dist[interference_mask])) if np.any(interference_mask) else 0.0,
                        'max_gap': float(np.max(valid_dist[gap_mask])) if np.any(gap_mask) else 0.0,
                        'mean_interference': float(np.mean(valid_dist[interference_mask])) if np.any(interference_mask) else 0.0,
                        'mean_gap': float(np.mean(valid_dist[gap_mask])) if np.any(gap_mask) else 0.0,
                        'interference_points': int(np.sum(interference_mask)),
                        'gap_points': int(np.sum(gap_mask)),
                        'interference_percentage': float(np.sum(interference_mask)) / len(valid_dist) * 100,
                        'gap_percentage': float(np.sum(gap_mask)) / len(valid_dist) * 100,
                        'mean': float(np.mean(valid_dist)),
                        'std': float(np.std(valid_dist))
                    }
                except Exception as e:
                    logging.error(f"计算 {dist_type} 统计数据时出错: {str(e)}")
                    continue
            
            if not stats:
                raise ValueError("没有有效的距离数据可分析")
            
            return stats
            
        except Exception as e:
            logging.error(f"结果分析失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def set_factors(self, material_factor=None, ppt_factor=None):
        """设置影响因子"""
        if material_factor is not None:
            self.material_factor = material_factor
        if ppt_factor is not None:
            self.ppt_factor = ppt_factor
    
    def _calculate_deformation(self, ppt_values, shoe_model):
        """计算材料变形量"""
        try:
            # 获取材质属性
            material_properties = shoe_model.get_material_properties()
            vertex_materials = shoe_model.get_vertex_materials()
            
            # 如果没有材质信息，使用默认值
            if not material_properties or not vertex_materials:
                logging.warning("使用默认材质属性")
                material_properties = {'default': {'E': 1000.0, 'v': 0.3}}
                vertex_materials = ['default'] * len(shoe_model.vertices)
            
            # 初始化变形数组
            deformation = np.zeros_like(ppt_values)
            
            # 设定基准参数
            base_deformation = 0.3  # 30% 的基准变形率
            reference_E = 50.0      # 参考杨氏模量 (50MPa)
            
            # 对每个材质区域计算变形
            for material_type, props in material_properties.items():
                # 获取当前材质的顶点
                material_mask = (vertex_materials == material_type)
                if not np.any(material_mask):
                    continue
                
                # 获取材质属性
                E = props.get('E', 1000.0)  # 杨氏模量，默认1000MPa
                v = props.get('v', 0.3)     # 泊松比，默认0.3
                
                # 计算相对刚度系数（越软的材料，变形越大）
                stiffness_ratio = reference_E / max(E, 0.001)
                
                # 计算变形系数（考虑泊松比的影响）
                deformation_factor = base_deformation * stiffness_ratio * (1 + v)
                deformation_factor = min(deformation_factor, 0.8)  # 限制最大变形
                
                # 计算该材质区域的变形量
                # 变形量与压力值成正比，与材料刚度成反比
                deformation[material_mask] = (
                    ppt_values[material_mask] * deformation_factor
                )
            
            # 确保变形量非负
            deformation = np.maximum(deformation, 0)
            
            # 打印变形统计信息
            logging.info("\n=== 材料变形统计 ===")
            logging.info(f"最大变形: {np.max(deformation):.2f} mm")
            logging.info(f"平均变形: {np.mean(deformation):.2f} mm")
            logging.info(f"变形点数: {np.sum(deformation > 0)}")
            
            return deformation
            
        except Exception as e:
            logging.error(f"变形计算失败: {str(e)}")
            import traceback
            traceback.print_exc()
            # 发生错误时返回零变形
            return np.zeros_like(ppt_values)
    
    def _compute_base_distances(self, foot_model, shoe_model):
        """计算基础适配距离"""
        # 构建鞋楦模型的KD树
        shoe_tree = KDTree(shoe_model.vertices)
        
        # 计算每个足部顶点到鞋楦的最近距离
        distances, indices = shoe_tree.query(foot_model.vertices)
        
        # 计算方向向量（从鞋楦表面指向足部顶点）
        direction_vectors = foot_model.vertices - shoe_model.vertices[indices]
        
        # 计算法向量的点积来确定内外
        dot_products = np.sum(direction_vectors * shoe_model.vertex_normals[indices], axis=1)
        
        # 根据点积符号调整距离的正负
        base_distances = np.where(dot_products > 0, distances, -distances)
        
        return base_distances
    
    def _apply_material_effect(self, base_distances, shoe_model):
        """应用材质效应"""
        try:
            material_distances = base_distances.copy()
            
            # 构建鞋楦模型的KD树用于找到最近点
            shoe_tree = KDTree(shoe_model.vertices)
            
            # 获取每个足部顶点对应的鞋楦最近点索引
            # 使用base_distances的长度来创建足部顶点的索引
            foot_vertices = np.arange(len(base_distances))
            _, indices = shoe_tree.query(shoe_model.vertices[foot_vertices])
            
            # 获取材质属性和分区信息
            material_properties = shoe_model.material_properties
            vertex_materials = shoe_model.vertex_materials
            
            # 获取每个足部顶点对应的鞋楦顶点的材质
            corresponding_materials = np.array(vertex_materials)[indices]
            
            # 设定基准变形参数
            base_deformation = 0.3  # 30% 的基准变形率
            reference_E = 50e6     # 参考杨氏模量 (50MPa)
            
            # 对每个材质区域分别处理
            for material_type, material_data in material_properties.items():
                # 获取当前材质的顶点
                material_mask = corresponding_materials == material_type
                
                if not np.any(material_mask):
                    continue
                
                # 获取材质属性
                E = material_data['E']  # 已经是Pa单位
                nu = material_data.get('nu', 0.3)  # 获取泊松比，默认0.3
                
                # 计算相对刚度系数（越软的材料，系数越大）
                stiffness_ratio = reference_E / E
                
                # 计算变形系数（考虑泊松比的影响）
                deformation_factor = base_deformation * stiffness_ratio * (1 + nu)
                
                # 限制最大变形系数
                deformation_factor = min(deformation_factor, 0.8)  # 最大变形不超过80%
                
                # 只对干涉区域（距离为负值）应用材质变形
                interference_mask = material_mask & (material_distances < 0)
                
                # 干涉区域：考虑材质变形减小干涉量
                original_interference = material_distances[interference_mask]
                material_distances[interference_mask] = original_interference * (1 - deformation_factor)
                
                # 记录修正信息
                if len(original_interference) > 0:
                    logging.info(f"\n{material_type}区域:")
                    logging.info(f"  - 变形系数: {deformation_factor:.3f} ({deformation_factor*100:.1f}%)")
                    logging.info(f"  - 原始平均干涉深度: {np.mean(original_interference):.2f} mm")
                    logging.info(f"  - 修正后平均干涉深度: {np.mean(material_distances[interference_mask]):.2f} mm")
                    logging.info(f"  - 变形量: {np.mean(original_interference * deformation_factor):.2f} mm")
            
            return material_distances
            
        except Exception as e:
            logging.error(f"材质效应应用失败: {str(e)}")
            return base_distances  # 如果出错，返回原始距离