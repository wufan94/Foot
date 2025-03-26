import numpy as np
import trimesh
from scipy.spatial import KDTree
import vtk
from vtk.util import numpy_support
import pandas as pd
import logging

class FootModel:
    def __init__(self, file_path):
        """加载足部模型"""
        # 创建VTK读取器
        reader = vtk.vtkVRMLImporter()
        reader.SetFileName(file_path)
        reader.Read()
        
        # 获取模型数据
        actors = reader.GetRenderer().GetActors()
        actors.InitTraversal()
        actor = actors.GetNextActor()
        
        # 获取PolyData
        self.model = actor.GetMapper().GetInput()
        
        # 获取顶点和面片数据
        self.vertices = numpy_support.vtk_to_numpy(self.model.GetPoints().GetData())
        faces = numpy_support.vtk_to_numpy(self.model.GetPolys().GetData())
        self.faces = faces.reshape(-1, 4)[:, 1:4]  # 跳过第一个元素（面片顶点数）
        
        # 计算法向量
        normal_generator = vtk.vtkPolyDataNormals()
        normal_generator.SetInputData(self.model)
        normal_generator.ComputePointNormalsOn()
        normal_generator.ComputeCellNormalsOn()
        normal_generator.Update()
        self.vertex_normals = numpy_support.vtk_to_numpy(normal_generator.GetOutput().GetPointData().GetNormals())

class ShoeLast:
    def __init__(self):
        """初始化鞋楦模型"""
        self.model = None
        self.vertices = None
        self.faces = None
        self.vertex_normals = None
        self.vertex_materials = []  # 每个顶点对应的材质名称
        self.material_properties = {}  # 材质属性字典
        
        # 材质名称映射（确保与OBJ文件中的材质名称对应）
        self.material_mapping = {
            '鞋舌上部': 'tongue_upper',
            '鞋舌下部': 'tongue_lower',
            '鞋尖': 'toe',
            '鞋面': 'upper',
            '后跟': 'heel',  # 将"后跟上部"改为"后跟"
            '鞋底': 'sole'
        }
        
        # 默认材质属性
        self.default_properties = {
            'tongue_upper': {'E': 10.0, 'v': 0.3},
            'tongue_lower': {'E': 10.0, 'v': 0.3},
            'toe': {'E': 20.0, 'v': 0.3},
            'upper': {'E': 1500.0, 'v': 0.42},
            'heel': {'E': 100.0, 'v': 0.3},  # 将"后跟上部"改为"后跟"
            'sole': {'E': 50.0, 'v': 0.45}
        }
    
    def load_model(self, file_path):
        """加载鞋楦模型"""
        try:
            print("加载鞋楦模型...")
            # 直接加载OBJ文件
            mesh = trimesh.load(file_path, encoding='utf-8', process=False)
            
            if isinstance(mesh, trimesh.Trimesh):
                # 如果是单个网格，直接使用
                self.model = mesh
                self.vertices = np.array(mesh.vertices, dtype=np.float64)
                self.faces = np.array(mesh.faces, dtype=np.int32)
                self.vertex_normals = np.array(mesh.vertex_normals, dtype=np.float64)
                
                # 默认所有顶点使用同一个材质
                self.vertex_materials = ['upper'] * len(self.vertices)
                
                print(f"\n模型加载完成:")
                print(f"  - 顶点数量: {len(self.vertices)}")
                print(f"  - 面片数量: {len(self.faces)}")
                
                # 设置默认材质属性
                for material, properties in self.default_properties.items():
                    self.set_material_property(material, 
                                            properties['E'],
                                            properties['v'])
                
                return True
                
            elif isinstance(mesh, trimesh.Scene):
                # 如果是场景（包含多个网格），合并所有网格
                meshes = []
                for geometry in mesh.geometry.values():
                    if isinstance(geometry, trimesh.Trimesh):
                        meshes.append(geometry)
                
                if not meshes:
                    raise ValueError("场景中没有有效的网格")
                
                # 合并所有网格
                self.model = trimesh.util.concatenate(meshes)
                
                # 保存顶点、面片和法向量数据
                self.vertices = np.array(self.model.vertices, dtype=np.float64)
                self.faces = np.array(self.model.faces, dtype=np.int32)
                self.vertex_normals = np.array(self.model.vertex_normals, dtype=np.float64)
                
                # 默认所有顶点使用同一个材质
                self.vertex_materials = ['upper'] * len(self.vertices)
                
                print(f"\n模型加载完成:")
                print(f"  - 顶点数量: {len(self.vertices)}")
                print(f"  - 面片数量: {len(self.faces)}")
                
                # 设置默认材质属性
                for material, properties in self.default_properties.items():
                    self.set_material_property(material, 
                                            properties['E'],
                                            properties['v'])
                
                return True
                
            else:
                raise ValueError(f"不支持的模型类型: {type(mesh)}")
                
        except Exception as e:
            print(f"加载鞋楦模型时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_material_regions_by_position(self):
        """根据位置划分材质区域"""
        try:
            print("根据位置确定材质区域...")
            vertices = self.model.vertices
            self.vertex_materials = np.zeros(len(vertices), dtype=object)
            
            y_max = np.max(vertices[:, 1])
            y_min = np.min(vertices[:, 1])
            z_max = np.max(vertices[:, 2])
            z_min = np.min(vertices[:, 2])
            
            # 划分区域
            for i, vertex in enumerate(vertices):
                y_norm = (vertex[1] - y_min) / (y_max - y_min)
                z_norm = (vertex[2] - z_min) / (z_max - z_min)
                
                if z_norm > 0.8:  # 鞋舌区域
                    self.vertex_materials[i] = '鞋舌'
                elif y_norm > 0.7:  # 鞋尖区域
                    self.vertex_materials[i] = '鞋尖'
                elif y_norm < 0.3:  # 鞋后跟区域
                    self.vertex_materials[i] = '后跟'
                elif z_norm < 0.2:  # 鞋底区域
                    self.vertex_materials[i] = '鞋底'
                else:  # 鞋面区域
                    self.vertex_materials[i] = '鞋面'
                    
            print("材质区域划分完成")
            
        except Exception as e:
            print(f"确定材质区域时出错: {str(e)}") 
    
    def get_vertex_materials(self, indices):
        """获取指定顶点的材质名称"""
        return [self.vertex_materials[i] for i in indices]
        
    def get_material_properties(self):
        """获取所有材质属性"""
        return self.material_properties
        
    def set_material_property(self, material, E, v):
        """设置材质属性
        Args:
            material: 材质名称
            E: 杨氏模量 (MPa)
            v: 泊松比
        """
        self.material_properties[material] = {
            'E': float(E),
            'v': float(v)
        }
        
    def get_material_property(self, material):
        """获取指定材质的属性"""
        return self.material_properties.get(material, {
            'E': 1000.0,  # 默认值
            'v': 0.3
        }) 

class PPTData:
    def __init__(self):
        self.coordinates = None  # GUR模型上的原始坐标
        self.values = None      # 原始压力值
        self.mapped_coords = None  # 映射到目标模型后的坐标
        
    def load_data(self, file_path):
        """加载PPT数据文件"""
        try:
            logging.info(f"读取PPT数据文件: {file_path}")
            
            # 读取Excel文件
            df = pd.read_excel(file_path, header=None)
            data = df.values
            
            # 分离坐标和PPT值
            self.coordinates = data[:, :3]  # 前三列为XYZ坐标
            self.values = data[:, 3]    # 第四列为PPT值
            
            # 验证数据
            if not self._validate_data():
                return False
            
            logging.info(f"PPT数据加载完成:")
            logging.info(f"  - 测量点数量: {len(self.coordinates)}")
            logging.info(f"  - PPT值范围: {np.min(self.values):.2f} - {np.max(self.values):.2f}")
            
            return True
            
        except Exception as e:
            logging.error(f"PPT数据加载失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
    def _validate_data(self):
        """验证数据有效性"""
        try:
            # 检查数据是否存在
            if self.coordinates is None or self.values is None:
                logging.error("PPT数据为空")
                return False
                
            # 检查维度匹配
            if len(self.coordinates) != len(self.values):
                logging.error("坐标点数量与压力值数量不匹配")
                return False
                
            # 检查是否有无效值
            if np.any(np.isnan(self.coordinates)) or np.any(np.isnan(self.values)):
                logging.error("数据中包含无效值(NaN)")
                return False
                
            # 检查坐标维度
            if self.coordinates.shape[1] != 3:
                logging.error("坐标数据维度错误")
                return False
                
            # 检查压力值是否为负
            if np.any(self.values < 0):
                logging.error("存在负压力值")
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"数据验证失败: {str(e)}")
            return False
            
    def get_interpolated_ppt(self, vertices):
        """在足部模型顶点上插值PPT值"""
        try:
            if self.mapped_coords is None:
                raise ValueError("PPT点未完成映射")
            
            # 构建KD树用于最近点搜索
            ppt_tree = KDTree(self.mapped_coords)
            
            # 对每个顶点找到最近的PPT点
            distances, indices = ppt_tree.query(vertices, k=4)  # 使用4个最近点
            
            # 计算权重（使用距离的倒数）
            weights = 1.0 / (distances + 1e-6)  # 避免除零
            weights = weights / np.sum(weights, axis=1, keepdims=True)
            
            # 对PPT值进行加权插值
            interpolated_values = np.zeros(len(vertices))
            for i in range(4):  # 4个最近点
                interpolated_values += weights[:,i] * self.values[indices[:,i]]
            
            return interpolated_values
            
        except Exception as e:
            logging.error(f"PPT插值失败: {str(e)}")
            return None 