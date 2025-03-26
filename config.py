# 文件路径配置
PATHS = {
    'GUR_MODEL': "C:/Users/wufan/Documents/MyFiles/3 研究院文件/项目文件/足鞋工效2024/2项目研发相关材料/3虚拟适配相关材料/虚拟适配250325/PPT_Model",    # PPT数据依附模型
    'PPT_DATA': "C:/Users/wufan/Documents/MyFiles/3 研究院文件/项目文件/足鞋工效2024/2项目研发相关材料/3虚拟适配相关材料/虚拟适配250325/PPT_Data",  # PPT数据文件
    'FOOT_MODEL': "C:/Users/wufan/Documents/MyFiles/3 研究院文件/项目文件/足鞋工效2024/2项目研发相关材料/3虚拟适配相关材料/虚拟适配250325/Foot_Model/S1_R.wrl",  # 被试足部扫描模型
    'FOOT_POINTS': "C:/Users/wufan/Documents/MyFiles/3 研究院文件/项目文件/足鞋工效2024/2项目研发相关材料/3虚拟适配相关材料/虚拟适配250325/Foot_Point",  # 被试足部标记点文件
    'SHOE_MODEL': "C:/Users/wufan/Documents/MyFiles/3 研究院文件/项目文件/足鞋工效2024/2项目研发相关材料/3虚拟适配相关材料/虚拟适配250325/Shoe_Model/35R.obj"   # 鞋内腔模型
}

# 鞋内腔顶点组定义
SHOE_REGIONS = {
    '鞋舌上部': 'tongue_upper',
    '鞋舌下部': 'tongue_lower',
    '鞋尖': 'toe',
    '鞋面': 'upper',
    '后跟上部': 'heel_upper',
    '后跟下部': 'heel_lower',
    '鞋底': 'sole'
}

# 区域划分阈值
REGION_THRESHOLDS = {
    'TOE': 0.7,    # 前掌区域起始位置（相对总长度）
    'ARCH': 0.3    # 足弓区域起始位置（相对总长度）
} 