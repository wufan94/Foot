# 项目开发指南

## 目录
1. [开发环境设置](#开发环境设置)
2. [项目结构](#项目结构)
3. [代码规范](#代码规范)
4. [开发流程](#开发流程)
5. [测试规范](#测试规范)
6. [文档规范](#文档规范)
7. [提交规范](#提交规范)

## 开发环境设置

### 必需环境
- Python 3.6+
- Git
- IDE推荐：VS Code或PyCharm

### 依赖安装
```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 开发依赖
```bash
# 安装开发依赖
pip install -r requirements-dev.txt
```

## 项目结构
```
Foot/
├── models/                 # 模型相关类
│   ├── __init__.py
│   ├── foot_model.py      # 足部模型类
│   └── shoe_model.py      # 鞋内腔模型类
├── registration/          # 配准相关模块
│   ├── __init__.py
│   └── icp.py            # ICP配准算法
├── fitting/              # 适配分析模块
│   ├── __init__.py
│   └── analyzer.py       # 适配分析器
├── visualization/        # 可视化模块
│   ├── __init__.py
│   └── renderer.py      # 渲染器
├── utils/               # 工具函数
│   ├── __init__.py
│   └── helpers.py      # 辅助函数
├── tests/              # 测试目录
│   ├── __init__.py
│   ├── test_models.py
│   └── test_fitting.py
├── docs/               # 文档目录
│   ├── api/           # API文档
│   └── guides/        # 使用指南
├── examples/          # 示例代码
├── requirements.txt   # 项目依赖
├── requirements-dev.txt # 开发依赖
├── setup.py          # 安装配置
├── README.md         # 项目说明
└── CONTRIBUTING.md   # 开发指南
```

## 代码规范

### Python代码风格
- 遵循PEP 8规范
- 使用4个空格缩进
- 行长度限制在79个字符
- 使用类型注解

### 命名规范
- 类名：使用CamelCase（如`FootModel`）
- 函数名：使用snake_case（如`compute_distances`）
- 变量名：使用snake_case（如`vertex_count`）
- 常量名：使用大写snake_case（如`MAX_DISTANCE`）

### 注释规范
- 每个模块开头添加模块说明
- 每个类添加类说明
- 每个方法添加方法说明
- 复杂算法添加详细注释

### 文档字符串
```python
def compute_distances(foot_model, shoe_model):
    """计算足部模型与鞋内腔模型之间的距离。

    Args:
        foot_model (FootModel): 足部模型对象
        shoe_model (ShoeModel): 鞋内腔模型对象

    Returns:
        numpy.ndarray: 距离数组，正值表示间隙，负值表示干涉

    Raises:
        ValueError: 当输入模型无效时
    """
```

## 开发流程

### 1. 创建分支
```bash
# 创建功能分支
git checkout -b feature/your-feature-name

# 创建修复分支
git checkout -b fix/your-fix-name
```

### 2. 开发流程
1. 编写代码
2. 运行测试
3. 更新文档
4. 提交代码

### 3. 代码审查
- 提交前进行自我审查
- 确保代码符合规范
- 确保测试通过
- 确保文档更新

## 测试规范

### 单元测试
- 使用pytest框架
- 测试覆盖率要求>80%
- 每个功能至少一个测试用例

### 测试文件命名
- 测试文件以`test_`开头
- 测试类以`Test`开头
- 测试方法以`test_`开头

### 测试示例
```python
def test_compute_distances():
    """测试距离计算功能"""
    # 准备测试数据
    foot_model = create_test_foot_model()
    shoe_model = create_test_shoe_model()
    
    # 执行测试
    distances = compute_distances(foot_model, shoe_model)
    
    # 验证结果
    assert isinstance(distances, np.ndarray)
    assert len(distances) > 0
```

## 文档规范

### 代码文档
- 使用Google风格的文档字符串
- 包含参数说明
- 包含返回值说明
- 包含异常说明

### API文档
- 使用Sphinx生成
- 包含示例代码
- 包含类型注解

### 更新文档
- 修改代码时同步更新文档
- 保持文档与代码一致
- 定期检查文档完整性

## 提交规范

### 提交信息格式
```
<type>(<scope>): <subject>

<body>

<footer>
```

### 类型说明
- feat: 新功能
- fix: 修复bug
- docs: 文档更新
- style: 代码格式
- refactor: 重构
- test: 测试相关
- chore: 构建过程或辅助工具的变动

### 示例
```
feat(fitting): 添加材质影响因子计算

- 实现材质影响因子的计算逻辑
- 添加相关测试用例
- 更新API文档

Closes #123
```

## 联系方式
- 项目负责人：Fan
- 邮箱：wufanspecial@outlook.com
- 问题反馈：提交Issue
- 代码提交：提交Pull Request 