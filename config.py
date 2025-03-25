# 文件路径配置
PATHS = {
    'GUR_MODEL': "C:/Users/wufan/Documents/MyFiles/3 研究院文件/项目文件/足鞋工效2024/2项目研发相关材料/3虚拟适配相关材料/最终虚拟适配/软组织及PPT映射模型/GURNEW.stl",    # GUR模型
    'PPT_DATA': "C:/Users/Wufan/Desktop/PPT R.xlsx",  # PPT数据文件
    'FOOT_MODEL': "C:/Users/Wufan/Desktop/S8 FootR.stl",  # 足部模型
    'SHOE_MODEL': "C:/Users/Wufan/Desktop/S8 lastR.obj"   # 鞋楦模型
}

# 默认材质属性
DEFAULT_MATERIALS = {
    '鞋舌': {'E': 10.0, 'v': 0.3},      # 鞋舌: 10 MPa
    '鞋面': {'E': 1500.0, 'v': 0.42},   # 鞋面: 1500 MPa
    '鞋尖': {'E': 20.0, 'v': 0.3},      # 鞋尖: 20 MPa
    '鞋跟': {'E': 100.0, 'v': 0.3},     # 鞋跟: 100 MPa
    '鞋底': {'E': 50.0, 'v': 0.45}      # 鞋底: 50 MPa
}

# 区域划分阈值
REGION_THRESHOLDS = {
    'TOE': 0.7,    # 前掌区域起始位置（相对总长度）
    'ARCH': 0.3    # 足弓区域起始位置（相对总长度）
} 