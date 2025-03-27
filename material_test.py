import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy import stats

def load_test_data(folder_path, sample_number):
    """加载指定试样文件夹中的所有测试数据
    Args:
        folder_path: 基础文件夹路径
        sample_number: 试样编号（1-7）
    Returns:
        data_list: 包含所有重复测试数据的列表
    """
    # 构建试样文件夹路径
    sample_folder = os.path.join(folder_path, f"试样{sample_number}")
    data_list = []
    
    # 读取该试样下的所有CSV文件
    for i in range(1, 4):
        file_name = f"{sample_number}-{i}.csv"
        file_path = os.path.join(sample_folder, file_name)
        
        try:
            # 读取CSV文件
            df = pd.read_csv(file_path, header=None)
            # 重命名列
            df.columns = ['displacement', 'force']
            # 添加到数据列表
            data_list.append(df)
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {str(e)}")
    
    return data_list

def load_test_data_with_3b(folder_path, sample_number):
    """加载指定试样文件夹中的测试数据，包括3B数据
    Args:
        folder_path: 基础文件夹路径
        sample_number: 试样编号（2-5）
    Returns:
        data_list: 包含所有重复测试数据的列表
    """
    # 构建试样文件夹路径
    sample_folder = os.path.join(folder_path, f"试样{sample_number}")
    data_list = []
    
    # 读取前两次测试数据
    for i in range(1, 3):
        file_name = f"{sample_number}-{i}.csv"
        file_path = os.path.join(sample_folder, file_name)
        
        try:
            df = pd.read_csv(file_path, header=None)
            df.columns = ['displacement', 'force']
            data_list.append(df)
        except Exception as e:
            print(f"读取文件 {file_path} 时出错: {str(e)}")
    
    # 读取3B数据
    file_name = f"{sample_number}-3B.csv"
    file_path = os.path.join(sample_folder, file_name)
    
    try:
        df_3b = pd.read_csv(file_path, header=None)
        df_3b.columns = ['displacement', 'force']
        
        # 修正3B数据
        if len(data_list) == 2:  # 确保有前两次测试数据
            # 计算前两次测试的平均曲线
            displacements = np.linspace(
                min(df_3b['displacement']), 
                max(df_3b['displacement']), 
                100
            )
            
            # 对前两次测试数据进行插值
            forces = []
            for df in data_list:
                f = interp1d(df['displacement'], df['force'], bounds_error=False, fill_value='extrapolate')
                forces.append(f(displacements))
            
            # 计算平均曲线
            avg_force = np.mean(forces, axis=0)
            
            # 修正3B数据
            f_3b = interp1d(df_3b['displacement'], df_3b['force'], bounds_error=False, fill_value='extrapolate')
            force_3b = f_3b(displacements)
            
            # 计算修正系数（基于前两次测试的平均值）
            # 避免除零错误
            force_3b = np.where(force_3b == 0, 1e-10, force_3b)
            correction_factor = avg_force / force_3b
            
            # 应用修正（使用中位数而不是平均值，避免异常值影响）
            correction = np.median(correction_factor)
            df_3b['force'] = df_3b['force'] * correction
            
            # 打印修正信息
            print(f"Sample {sample_number} 3B correction factor: {correction:.4f}")
            
            # 保存修正后的数据
            new_file_name = f"{sample_number}-3.csv"
            new_file_path = os.path.join(sample_folder, new_file_name)
            df_3b.to_csv(new_file_path, index=False, header=False)
            print(f"保存修正后的数据到: {new_file_path}")
            
            data_list.append(df_3b)
    except Exception as e:
        print(f"读取或修正3B文件 {file_path} 时出错: {str(e)}")
    
    return data_list

def fit_curve(x, y):
    """拟合曲线
    Args:
        x: 位移数据
        y: 力数据
    Returns:
        popt: 拟合参数
        pcov: 参数协方差
        equation: 拟合方程
    """
    # 定义拟合函数（使用多项式函数）
    def func(x, a, b, c, d, e):
        return a * x**4 + b * x**3 + c * x**2 + d * x + e
    
    # 进行曲线拟合
    popt, pcov = curve_fit(func, x, y)
    
    # 生成拟合方程
    equation = f"F = {popt[0]:.4f}x⁴ + {popt[1]:.4f}x³ + {popt[2]:.4f}x² + {popt[3]:.4f}x + {popt[4]:.4f}"
    
    return popt, pcov, equation

def calculate_variability(data_list):
    """计算三次测试的变异系数
    Args:
        data_list: 包含三次测试数据的列表
    Returns:
        cv: 变异系数
        stability: 稳定性评估
    """
    # 对每个位移点计算三次测试的平均值和标准差
    displacements = np.unique(np.concatenate([df['displacement'] for df in data_list]))
    forces = []
    
    for disp in displacements:
        force_values = []
        for df in data_list:
            # 找到最接近的位移值
            idx = np.abs(df['displacement'] - disp).argmin()
            force_values.append(df['force'].iloc[idx])
        forces.append(force_values)
    
    forces = np.array(forces)
    means = np.mean(forces, axis=1)
    stds = np.std(forces, axis=1)
    
    # 计算变异系数（避免除零）
    valid_indices = means != 0
    if np.any(valid_indices):
        cv = np.mean(stds[valid_indices] / means[valid_indices]) * 100
    else:
        cv = 0
    
    # 评估稳定性
    if cv < 5:
        stability = "Excellent"
    elif cv < 10:
        stability = "Good"
    elif cv < 15:
        stability = "Fair"
    else:
        stability = "Poor"
    
    return cv, stability

def calculate_fit_accuracy(x, y, popt):
    """计算拟合精度
    Args:
        x: 位移数据
        y: 实际力数据
        popt: 拟合参数
    Returns:
        r_squared: R²值
        rmse: 均方根误差
    """
    # 定义拟合函数
    def func(x, a, b, c, d, e):
        return a * x**4 + b * x**3 + c * x**2 + d * x + e
    
    # 计算拟合值
    y_fit = func(x, *popt)
    
    # 计算R²值
    residuals = y - y_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # 计算RMSE
    rmse = np.sqrt(np.mean(residuals**2))
    
    return r_squared, rmse

def plot_test_data(data_list, sample_number):
    """绘制测试数据
    Args:
        data_list: 包含所有重复测试数据的列表
        sample_number: 试样编号
    """
    # 创建图形
    plt.figure(figsize=(10, 6))
    
    # 设置颜色列表
    colors = ['red', 'blue', 'green']
    
    # 绘制每条曲线
    for i, data in enumerate(data_list):
        plt.plot(data['displacement'], data['force'], 
                color=colors[i], 
                label=f'Test {i+1}',
                linewidth=2)
    
    # 设置图形属性
    plt.title(f'Sample {sample_number} Burst Test Data', fontsize=14)
    plt.xlabel('Displacement (mm)', fontsize=12)
    plt.ylabel('Force (N)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # 设置坐标轴范围
    # 获取所有数据的最大最小值
    all_displacement = np.concatenate([data['displacement'] for data in data_list])
    all_force = np.concatenate([data['force'] for data in data_list])
    
    plt.xlim(min(all_displacement), max(all_displacement))
    plt.ylim(0, max(all_force) * 1.1)  # 留出10%的空间
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图形
    plt.savefig(f'Sample_{sample_number}_Test_Data.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_all_tests(data_dict):
    """绘制所有试样的测试数据
    Args:
        data_dict: 包含所有试样数据的字典
    """
    # 创建图形
    fig = plt.figure(figsize=(20, 15))
    
    # 计算子图的行数和列数
    n_samples = len(data_dict)
    n_cols = 3
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    # 绘制每个试样的数据
    for i, (sample_number, data_list) in enumerate(data_dict.items()):
        # 创建子图
        ax = plt.subplot(n_rows, n_cols, i + 1)
        
        # 设置颜色列表
        colors = ['red', 'blue', 'green']
        
        # 绘制三次测试数据
        for j, data in enumerate(data_list):
            ax.plot(data['displacement'], data['force'], 
                    color=colors[j], 
                    label=f'Test {j+1}',
                    linewidth=2)
        
        # 设置子图属性
        ax.set_title(f'Sample {sample_number}', fontsize=12)
        ax.set_xlabel('Displacement (mm)', fontsize=10)
        ax.set_ylabel('Force (N)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=8)
        
        # 设置坐标轴范围
        all_displacement = np.concatenate([data['displacement'] for data in data_list])
        all_force = np.concatenate([data['force'] for data in data_list])
        
        ax.set_xlim(min(all_displacement), max(all_displacement))
        ax.set_ylim(0, max(all_force) * 1.1)  # 留出10%的空间
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    # 保存图形
    plt.savefig('All_Samples_Test_Data.png', dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_fit_quality(r_squared):
    """评估拟合质量
    Args:
        r_squared: R²值
    Returns:
        quality: 拟合质量评估
    """
    if r_squared > 0.95:
        return "Excellent"
    elif r_squared > 0.90:
        return "Good"
    elif r_squared > 0.80:
        return "Fair"
    else:
        return "Poor"

def plot_fitted_curves(data_dict):
    """绘制所有试样的拟合曲线
    Args:
        data_dict: 包含所有试样数据的字典
    """
    # 创建图形
    fig = plt.figure(figsize=(20, 15))
    
    # 计算子图的行数和列数
    n_samples = len(data_dict)
    n_cols = 3
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    # 绘制每个试样的拟合曲线
    for i, (sample_number, data_list) in enumerate(data_dict.items()):
        # 创建子图
        ax = plt.subplot(n_rows, n_cols, i + 1)
        
        # 计算平均曲线
        displacements = np.unique(np.concatenate([df['displacement'] for df in data_list]))
        forces = []
        
        for disp in displacements:
            force_values = []
            for df in data_list:
                idx = np.abs(df['displacement'] - disp).argmin()
                force_values.append(df['force'].iloc[idx])
            forces.append(np.mean(force_values))
        
        # 拟合曲线
        popt, pcov, equation = fit_curve(displacements, forces)
        
        # 计算拟合精度
        r_squared, rmse = calculate_fit_accuracy(displacements, forces, popt)
        quality = evaluate_fit_quality(r_squared)
        
        # 绘制拟合曲线
        ax.plot(displacements, forces, 
                color='blue', 
                label=f'Fitted Curve\n{equation}\nR² = {r_squared:.4f} ({quality})\nRMSE = {rmse:.2f}',
                linewidth=2)
        
        # 设置子图属性
        ax.set_title(f'Sample {sample_number}', fontsize=12)
        ax.set_xlabel('Displacement (mm)', fontsize=10)
        ax.set_ylabel('Force (N)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=8)
        
        # 设置坐标轴范围
        ax.set_xlim(min(displacements), max(displacements))
        ax.set_ylim(0, max(forces) * 1.1)  # 留出10%的空间
        
        # 打印拟合质量信息
        print(f"Sample {sample_number} - R² = {r_squared:.4f}, Quality: {quality}")
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    # 保存图形
    plt.savefig('All_Samples_Fitted_Curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_report(data_dict, base_path):
    """生成详细的测试报告
    Args:
        data_dict: 包含所有试样数据的字典
        base_path: 基础路径
    """
    # 创建报告文件
    report_path = os.path.join(base_path, "测试报告.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("材料测试详细报告\n")
        f.write("=" * 50 + "\n\n")
        
        # 处理每个试样
        for sample_number, data_list in data_dict.items():
            f.write(f"试样 {sample_number}\n")
            f.write("-" * 30 + "\n")
            
            # 计算变异系数和稳定性
            cv, stability = calculate_variability(data_list)
            f.write(f"变异系数 (CV): {cv:.2f}%\n")
            f.write(f"稳定性评估: {stability}\n\n")
            
            # 计算平均曲线
            displacements = np.unique(np.concatenate([df['displacement'] for df in data_list]))
            forces = []
            
            for disp in displacements:
                force_values = []
                for df in data_list:
                    idx = np.abs(df['displacement'] - disp).argmin()
                    force_values.append(df['force'].iloc[idx])
                forces.append(np.mean(force_values))
            
            # 拟合曲线
            popt, pcov, equation = fit_curve(displacements, forces)
            
            # 计算拟合精度
            r_squared, rmse = calculate_fit_accuracy(displacements, forces, popt)
            quality = evaluate_fit_quality(r_squared)
            
            # 写入拟合信息
            f.write("拟合函数信息:\n")
            f.write(f"方程: {equation}\n")
            f.write(f"R²值: {r_squared:.4f}\n")
            f.write(f"拟合质量: {quality}\n")
            f.write(f"RMSE: {rmse:.2f}\n")
            f.write("\n拟合参数:\n")
            f.write(f"a (x⁴): {popt[0]:.4f}\n")
            f.write(f"b (x³): {popt[1]:.4f}\n")
            f.write(f"c (x²): {popt[2]:.4f}\n")
            f.write(f"d (x): {popt[3]:.4f}\n")
            f.write(f"e (常数): {popt[4]:.4f}\n")
            f.write("\n" + "=" * 50 + "\n\n")
        
        # 添加总结信息
        f.write("测试总结\n")
        f.write("-" * 30 + "\n")
        f.write(f"总试样数量: {len(data_dict)}\n")
        f.write(f"测试时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n")

def main():
    # 设置基础路径
    base_path = r"C:\Users\wufan\Documents\MyFiles\3 研究院文件\项目文件\足鞋工效2024\2项目研发相关材料\3虚拟适配相关材料\材料测试\修正位移"
    
    # 要测试的试样编号
    sample_numbers = [1, 2, 3, 4, 5, 6, 7]
    
    # 存储所有试样数据
    data_dict = {}
    
    # 处理每个试样
    for sample_number in sample_numbers:
        print(f"\nProcessing Sample {sample_number}...")
        
        # 加载数据
        data_list = load_test_data(base_path, sample_number)
        
        if len(data_list) > 0:
            # 计算变异系数和稳定性
            cv, stability = calculate_variability(data_list)
            print(f"Sample {sample_number} - CV: {cv:.2f}%, Stability: {stability}")
            
            # 存储数据
            data_dict[sample_number] = data_list
            
            # 绘制数据
            plot_test_data(data_list, sample_number)
            print(f"Sample {sample_number} data processed, figure saved as 'Sample_{sample_number}_Test_Data.png'")
        else:
            print(f"No valid data for Sample {sample_number}")
    
    # 绘制所有试样的测试数据
    plot_all_tests(data_dict)
    print("\n已生成所有试样的测试数据图：All_Samples_Test_Data.png")
    
    # 绘制所有试样的拟合曲线
    plot_fitted_curves(data_dict)
    print("已生成所有试样的拟合曲线图：All_Samples_Fitted_Curves.png")
    
    # 生成详细报告
    generate_report(data_dict, base_path)
    print(f"已生成详细报告：{os.path.join(base_path, '测试报告.txt')}")

if __name__ == "__main__":
    main() 