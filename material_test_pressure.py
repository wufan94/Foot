import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from scipy import stats
import math

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

def convert_force_to_pressure(df):
    """将力值转换为压力值
    Args:
        df: 包含位移和力值的DataFrame
    Returns:
        new_df: 包含位移、力值和压力值的新DataFrame
    """
    # 创建新的DataFrame，复制原始数据
    new_df = df.copy()
    
    # 计算有效接触面积和等效压力
    R = 12.5  # 球半径，单位mm
    
    # 添加压力列
    new_df['pressure'] = np.zeros(len(df))
    
    for i, row in df.iterrows():
        d = row['displacement']  # 压入深度，单位mm
        if d > 0:  # 避免负位移
            # 计算有效接触面积 (mm²)
            A_effective = math.pi * R * d
            
            # 计算等效压力 (kPa)
            # 将力 (N) 除以面积 (mm²) 得到 N/mm²，然后乘以1000得到kPa
            pressure = (row['force'] / A_effective) * 1000
        else:
            pressure = 0
        
        new_df.loc[i, 'pressure'] = pressure
    
    return new_df

def plot_pressure_curves(data_dict):
    """绘制所有试样的压力曲线
    Args:
        data_dict: 包含所有试样数据的字典
    """
    # 创建图形
    fig = plt.figure(figsize=(20, 15))
    
    # 计算子图的行数和列数
    n_samples = len(data_dict)
    n_cols = 3
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    # 绘制每个试样的压力曲线
    for i, (sample_number, data_list) in enumerate(data_dict.items()):
        # 创建子图
        ax = plt.subplot(n_rows, n_cols, i + 1)
        
        # 设置颜色列表
        colors = ['red', 'blue', 'green']
        
        # 收集所有转换后的数据
        processed_data_list = []
        for df in data_list:
            processed_df = convert_force_to_pressure(df)
            processed_data_list.append(processed_df)
        
        # 绘制三次测试数据的压力曲线
        for j, data in enumerate(processed_data_list):
            ax.plot(data['displacement'], data['pressure'], 
                    color=colors[j], 
                    label=f'Test {j+1}',
                    linewidth=2)
        
        # 设置子图属性
        ax.set_title(f'样本 {sample_number} 压力-位移曲线', fontsize=12)
        ax.set_xlabel('位移 (mm)', fontsize=10)
        ax.set_ylabel('压力 (kPa)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=8)
        
        # 设置坐标轴范围
        all_displacement = np.concatenate([data['displacement'] for data in processed_data_list])
        all_pressure = np.concatenate([data['pressure'] for data in processed_data_list])
        
        # 过滤掉异常值（例如无限大或NaN）
        valid_pressure = all_pressure[np.isfinite(all_pressure)]
        
        ax.set_xlim(0, max(all_displacement))
        if len(valid_pressure) > 0:
            max_pressure = max(valid_pressure)
            ax.set_ylim(0, min(max_pressure * 1.1, 2000))  # 限制最大显示压力为2000 kPa
        
        # 添加标记线 - 例如800 kPa的疼痛阈值
        ax.axhline(y=800, color='red', linestyle='--', alpha=0.7)
        ax.text(max(all_displacement)*0.05, 820, '疼痛阈值 (800 kPa)', 
                fontsize=8, color='red')
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    # 保存图形
    plt.savefig('All_Samples_Pressure_Curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return processed_data_list

def plot_force_and_pressure(data_dict):
    """在同一图中绘制力和压力曲线，以便比较
    Args:
        data_dict: 包含所有试样数据的字典
    """
    # 创建图形
    fig = plt.figure(figsize=(20, 15))
    
    # 计算子图的行数和列数
    n_samples = len(data_dict)
    n_cols = 3
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    # 绘制每个试样的力和压力曲线
    for i, (sample_number, data_list) in enumerate(data_dict.items()):
        # 创建子图
        ax = plt.subplot(n_rows, n_cols, i + 1)
        
        # 收集所有转换后的数据
        processed_data_list = []
        for df in data_list:
            processed_df = convert_force_to_pressure(df)
            processed_data_list.append(processed_df)
        
        # 取第一次测试的数据用于比较
        data = processed_data_list[0]
        
        # 创建两个Y轴
        ax2 = ax.twinx()
        
        # 绘制力曲线
        line1, = ax.plot(data['displacement'], data['force'], 
                 color='blue', 
                 label='力 (N)',
                 linewidth=2)
        
        # 绘制压力曲线
        line2, = ax2.plot(data['displacement'], data['pressure'], 
                 color='red', 
                 label='压力 (kPa)',
                 linewidth=2)
        
        # 添加图例
        lines = [line1, line2]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, fontsize=8, loc='upper left')
        
        # 设置子图属性
        ax.set_title(f'样本 {sample_number} 力和压力比较', fontsize=12)
        ax.set_xlabel('位移 (mm)', fontsize=10)
        ax.set_ylabel('力 (N)', fontsize=10, color='blue')
        ax2.set_ylabel('压力 (kPa)', fontsize=10, color='red')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 设置坐标轴范围
        ax.set_xlim(0, max(data['displacement']))
        ax.set_ylim(0, max(data['force']) * 1.1)
        
        valid_pressure = data['pressure'][np.isfinite(data['pressure'])]
        if len(valid_pressure) > 0:
            max_pressure = max(valid_pressure)
            ax2.set_ylim(0, min(max_pressure * 1.1, 2000))
        
        # 添加标记线 - 例如800 kPa的疼痛阈值
        ax2.axhline(y=800, color='red', linestyle='--', alpha=0.5)
        ax2.text(max(data['displacement'])*0.05, 820, '疼痛阈值 (800 kPa)', 
                fontsize=8, color='red')
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    # 保存图形
    plt.savefig('Force_vs_Pressure_Comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def fit_pressure_curve(data_dict):
    """拟合压力曲线并显示拟合方程
    Args:
        data_dict: 包含所有试样数据的字典
    """
    # 创建图形
    fig = plt.figure(figsize=(20, 15))
    
    # 计算子图的行数和列数
    n_samples = len(data_dict)
    n_cols = 3
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    # 存储拟合方程
    equations = {}
    
    # 绘制每个试样的压力拟合曲线
    for i, (sample_number, data_list) in enumerate(data_dict.items()):
        # 创建子图
        ax = plt.subplot(n_rows, n_cols, i + 1)
        
        # 收集所有数据点
        all_displacements = []
        all_pressures = []
        
        # 转换数据
        for df in data_list:
            processed_df = convert_force_to_pressure(df)
            valid_data = processed_df[processed_df['displacement'] > 0]  # 只使用正位移值
            all_displacements.extend(valid_data['displacement'].values)
            all_pressures.extend(valid_data['pressure'].values)
        
        # 转换为numpy数组并排序
        all_data = np.array(list(zip(all_displacements, all_pressures)))
        all_data = all_data[np.isfinite(all_data[:, 1])]  # 移除无穷大或NaN值
        sorted_data = all_data[all_data[:, 0].argsort()]
        
        # 如果有足够数据点才进行拟合
        if len(sorted_data) > 10:
            # 拟合四次多项式
            popt = np.polyfit(sorted_data[:, 0], sorted_data[:, 1], 4)
            
            # 生成拟合方程
            equation = f"P = {popt[0]:.4f}x⁴ + {popt[1]:.4f}x³ + {popt[2]:.4f}x² + {popt[3]:.4f}x + {popt[4]:.4f}"
            equations[sample_number] = equation
            
            # 计算拟合精度
            y_fit = np.polyval(popt, sorted_data[:, 0])
            residuals = sorted_data[:, 1] - y_fit
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((sorted_data[:, 1] - np.mean(sorted_data[:, 1]))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # 绘制原始数据点
            ax.scatter(sorted_data[:, 0], sorted_data[:, 1], 
                      color='red', s=10, alpha=0.5, label='测试数据')
            
            # 绘制拟合曲线
            x_fit = np.linspace(0, max(sorted_data[:, 0]), 200)
            y_fit = np.polyval(popt, x_fit)
            ax.plot(x_fit, y_fit, 'b-', linewidth=2,
                   label=f'拟合曲线\n{equation}\nR² = {r_squared:.4f}')
        else:
            ax.text(0.5, 0.5, '数据点不足以进行拟合',
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes)
        
        # 设置子图属性
        ax.set_title(f'样本 {sample_number} 压力拟合曲线', fontsize=12)
        ax.set_xlabel('位移 (mm)', fontsize=10)
        ax.set_ylabel('压力 (kPa)', fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        if len(sorted_data) > 10:
            ax.legend(fontsize=8)
        
        # 设置坐标轴范围
        if len(sorted_data) > 0:
            ax.set_xlim(0, max(sorted_data[:, 0]))
            max_pressure = max(sorted_data[:, 1])
            ax.set_ylim(0, min(max_pressure * 1.1, 2000))
        
        # 添加标记线 - 例如800 kPa的疼痛阈值
        ax.axhline(y=800, color='red', linestyle='--', alpha=0.7)
        ax.text(max(sorted_data[:, 0])*0.05 if len(sorted_data) > 0 else 0.5, 
                820, '疼痛阈值 (800 kPa)', fontsize=8, color='red')
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    # 保存图形
    plt.savefig('Pressure_Fitted_Curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 打印所有拟合方程
    print("\n压力拟合方程(kPa):")
    for sample_number, equation in equations.items():
        print(f"样本 {sample_number}: {equation}")
    
    return equations

def main():
    # 设置基础路径
    base_path = r"C:\Users\wufan\Documents\MyFiles\3 研究院文件\项目文件\足鞋工效2024\2项目研发相关材料\3虚拟适配相关材料\材料测试\修正位移"
    
    # 要测试的试样编号
    sample_numbers = [1, 2, 3, 4, 5, 6, 7]
    
    # 存储所有试样数据
    data_dict = {}
    
    # 处理每个试样
    for sample_number in sample_numbers:
        print(f"\n处理样本 {sample_number}...")
        
        # 加载数据
        data_list = load_test_data(base_path, sample_number)
        
        if len(data_list) > 0:
            # 存储数据
            data_dict[sample_number] = data_list
            print(f"样本 {sample_number} 数据已加载")
        else:
            print(f"样本 {sample_number} 没有有效数据")
    
    # 分析转换结果
    print("\n压力转换分析:")
    print("* 球半径(R) = 12.5mm")
    print("* 压力单位: kPa (千帕)")
    print("* 转换公式: 压力(kPa) = 力(N) / (π·R·d) × 1000")
    print("  其中d是压入深度(mm)")
    
    # 绘制压力曲线
    plot_pressure_curves(data_dict)
    print("\n已生成所有试样的压力曲线图：All_Samples_Pressure_Curves.png")
    
    # 绘制力和压力比较图
    plot_force_and_pressure(data_dict)
    print("已生成力与压力比较图：Force_vs_Pressure_Comparison.png")
    
    # 拟合压力曲线
    fit_pressure_curve(data_dict)
    print("已生成压力拟合曲线图：Pressure_Fitted_Curves.png")
    
    print("\n转换完成! 所有结果已保存为图像文件。")

if __name__ == "__main__":
    main() 