import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Any

# 添加项目根目录到路径以便导入
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model_old.teacher_model import CTRTeacherModel
from model_old.student_model import LightweightCTRModel, FeatureExtractor

class ModelComparator:
    """模型比较器：比较教师模型和学生模型的性能和特征表示"""
    
    def __init__(
        self, 
        teacher_model_path: str,
        student_model_path: str,
        data_path: str,
        sparse_features: List[str] = None,
        device: str = None
    ):
        """
        初始化比较器
        
        参数:
            teacher_model_path: 教师模型路径
            student_model_path: 学生模型路径
            data_path: 数据路径
            sparse_features: 稀疏特征列表
            device: 设备类型
        """
        self.teacher_model_path = teacher_model_path
        self.student_model_path = student_model_path
        self.data_path = data_path
        
        # 设置默认特征
        if sparse_features is None:
            self.sparse_features = ['hour', 'C1', 'banner_pos', 'site_id', 
                                   'app_id', 'device_type', 'device_conn_type']
        else:
            self.sparse_features = sparse_features
            
        # 设置设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"使用设备: {self.device}")
        
        # 加载模型和数据
        self.load_models_and_data()
        
    def load_models_and_data(self):
        """加载模型和数据"""
        print("加载数据...")
        self.data = pd.read_csv(self.data_path)
        
        # 数据预处理
        self.data['click'] = pd.to_numeric(self.data['click'], errors='coerce')
        self.data = self.data.dropna(subset=['click'])
        self.data['click'] = self.data['click'].astype(int)
        
        # 划分训练集和测试集
        print("划分训练集与测试集...")
        _, self.test_data = train_test_split(self.data, test_size=0.2, random_state=42)
        
        # 加载教师模型
        print(f"加载教师模型: {self.teacher_model_path}")
        self.teacher_model = CTRTeacherModel(sparse_features=self.sparse_features)
        try:
            self.teacher_model.load_model(self.teacher_model_path, self.data)
        except Exception as e:
            print(f"警告: 加载教师模型出错: {e}")
            print("尝试使用非严格模式加载模型...")
            checkpoint = torch.load(self.teacher_model_path)
            self.teacher_model.sparse_features = checkpoint['sparse_features']
            self.teacher_model.embedding_dim = checkpoint['embedding_dim']
            self.teacher_model.task = checkpoint['task']
            self.teacher_model.encoders = checkpoint['encoders']
            self.teacher_model.build_model(self.data)
            self.teacher_model.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        # 加载学生模型
        print(f"加载学生模型: {self.student_model_path}")
        student_checkpoint = torch.load(self.student_model_path)
        self.student_model = LightweightCTRModel(
            input_dim=student_checkpoint['input_dim'],
            hidden_dims=student_checkpoint['hidden_dims'],
            dropout_rate=student_checkpoint['dropout_rate']
        ).to(self.device)
        self.student_model.load_state_dict(student_checkpoint['model_state_dict'])
        self.student_model.eval()
        
        # 创建特征提取器
        self.feature_extractor = FeatureExtractor(self.teacher_model)
        
        # 预处理测试数据
        self.processed_test = self.teacher_model.preprocess_data(self.test_data)
        self.test_input = self.teacher_model.prepare_model_input(self.processed_test)
        self.y_test = self.processed_test['click'].values
        
    def compare_model_size(self) -> Dict[str, int]:
        """比较模型大小和参数量"""
        print("\n=== 模型大小比较 ===")
        
        # 计算教师模型参数量
        teacher_params = sum(p.numel() for p in self.teacher_model.model.parameters())
        
        # 计算学生模型参数量
        student_params = sum(p.numel() for p in self.student_model.parameters())
        
        # 计算模型大小（MB）
        teacher_size_mb = os.path.getsize(self.teacher_model_path) / (1024 * 1024)
        student_size_mb = os.path.getsize(self.student_model_path) / (1024 * 1024)
        
        # 输出比较结果
        print(f"教师模型参数量: {teacher_params:,}")
        print(f"学生模型参数量: {student_params:,}")
        print(f"参数量减少比例: {(1 - student_params/teacher_params)*100:.2f}%")
        print(f"教师模型大小: {teacher_size_mb:.2f} MB")
        print(f"学生模型大小: {student_size_mb:.2f} MB")
        print(f"模型大小减少比例: {(1 - student_size_mb/teacher_size_mb)*100:.2f}%")
        
        return {
            'teacher_params': teacher_params,
            'student_params': student_params,
            'teacher_size_mb': teacher_size_mb,
            'student_size_mb': student_size_mb,
            'param_reduction': (1 - student_params/teacher_params)*100,
            'size_reduction': (1 - student_size_mb/teacher_size_mb)*100
        }
    
    def compare_inference_speed(self, num_runs: int = 10, batch_size: int = 1000) -> Dict[str, float]:
        """比较模型推理速度"""
        print("\n=== 推理速度比较 ===")
        
        # 准备数据 - 取测试集的子集而不是提取输入字典
        if len(self.processed_test) > batch_size:
            test_sample_df = self.processed_test.iloc[:batch_size].copy()
        else:
            test_sample_df = self.processed_test.copy()
            
        # 从DataFrame中准备模型输入
        test_sample_input = self.teacher_model.prepare_model_input(test_sample_df)
            
        # 提取特征
        with torch.no_grad():
            test_features = self.feature_extractor.extract_features(test_sample_input)
            if isinstance(test_features, torch.Tensor):
                test_features = test_features.float()
                if test_features.device != self.device:
                    test_features = test_features.to(self.device)
        
        # 预热
        with torch.no_grad():
            _ = self.teacher_model.predict(test_sample_df)  # 使用DataFrame而非字典
            _ = self.student_model.predict(test_features)
        
        # 测量教师模型推理时间
        teacher_times = []
        print(f"测量教师模型推理时间 ({num_runs}次)...")
        for _ in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                _ = self.teacher_model.predict(test_sample_df)  # 使用DataFrame而非字典
            teacher_times.append(time.time() - start_time)
        
        # 测量学生模型推理时间
        student_times = []
        print(f"测量学生模型推理时间 ({num_runs}次)...")
        for _ in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                _ = self.student_model.predict(test_features)
            student_times.append(time.time() - start_time)
        
        # 计算平均推理时间
        teacher_avg_time = sum(teacher_times) / len(teacher_times)
        student_avg_time = sum(student_times) / len(student_times)
        
        # 输出结果
        print(f"教师模型平均推理时间: {teacher_avg_time*1000:.2f} ms")
        print(f"学生模型平均推理时间: {student_avg_time*1000:.2f} ms")
        print(f"推理速度提升: {(teacher_avg_time/student_avg_time):.2f}x")
        
        return {
            'teacher_avg_time_ms': teacher_avg_time*1000,
            'student_avg_time_ms': student_avg_time*1000,
            'speedup_ratio': teacher_avg_time/student_avg_time,
            'teacher_times': teacher_times,
            'student_times': student_times
        }     
    
    def compare_prediction_performance(self) -> Dict[str, float]:
        """比较模型预测性能"""
        print("\n=== 预测性能比较 ===")
        
        # 教师模型预测 - 使用DataFrame而非字典
        teacher_preds = self.teacher_model.predict(self.processed_test)
        
        # 学生模型预测
        with torch.no_grad():
            test_features = self.feature_extractor.extract_features(self.test_input)
            if isinstance(test_features, torch.Tensor):
                test_features = test_features.float().to(self.device)
            student_preds = self.student_model.predict(test_features)
            if isinstance(student_preds, torch.Tensor):
                student_preds = student_preds.cpu().numpy()
        
        # 确保预测结果是正确的形状
        if len(student_preds.shape) > 1 and student_preds.shape[1] == 1:
            student_preds = student_preds.flatten()
        
        # 计算评估指标
        teacher_loss = log_loss(self.y_test, teacher_preds)
        teacher_auc = roc_auc_score(self.y_test, teacher_preds)
        
        student_loss = log_loss(self.y_test, student_preds)
        student_auc = roc_auc_score(self.y_test, student_preds)
        
        # 输出结果
        print(f"教师模型 - Log Loss: {teacher_loss:.4f}, AUC: {teacher_auc:.4f}")
        print(f"学生模型 - Log Loss: {student_loss:.4f}, AUC: {student_auc:.4f}")
        print(f"性能差距 - Log Loss: {student_loss-teacher_loss:.4f}, AUC: {student_auc-teacher_auc:.4f}")
        
        return {
            'teacher_loss': teacher_loss,
            'teacher_auc': teacher_auc,
            'student_loss': student_loss,
            'student_auc': student_auc,
            'loss_difference': student_loss - teacher_loss,
            'auc_difference': student_auc - teacher_auc,
            'teacher_preds': teacher_preds,
            'student_preds': student_preds
        }

    def compare_feature_representations(self, num_samples: int = 1000) -> Dict[str, Any]:
        """比较模型的特征表示"""
        print("\n=== 特征表示比较 ===")
        
        # 准备数据 - 使用DataFrame子集
        if len(self.processed_test) > num_samples:
            test_sample_df = self.processed_test.iloc[:num_samples].copy()
        else:
            test_sample_df = self.processed_test.copy()
            num_samples = len(test_sample_df)
        
        # 准备模型输入
        test_sample_input = self.teacher_model.prepare_model_input(test_sample_df)
            
        # 获取教师模型中间层输出
        teacher_outputs = self.teacher_model.get_hidden_outputs(test_sample_input)
        
        # 打印可用的教师模型输出键
        print("教师模型输出包含以下键:")
        for key, value in teacher_outputs.items():
            if isinstance(value, torch.Tensor):
                shape_info = f"形状={value.shape}" if hasattr(value, 'shape') else "非张量"
                print(f" - {key}: {shape_info}")
        
        # 获取学生模型特征
        test_features = self.feature_extractor.extract_features(test_sample_input)
        if isinstance(test_features, torch.Tensor):
            test_features = test_features.float().to(self.device)
            
        # 获取学生模型中间层输出
        with torch.no_grad():
            _, student_intermediate = self.student_model(test_features)
            
        # 打印可用的学生模型中间层输出
        print("学生模型中间层包含以下键:")
        for key, value in student_intermediate.items():
            if isinstance(value, torch.Tensor):
                print(f" - {key}: 形状={value.shape}")
        
        # 添加特征表示的分析 - 计算各层输出的统计特性
        teacher_stats = {}
        student_stats = {}
        
        # 分析教师模型输出
        for key, value in teacher_outputs.items():
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                # 确保张量是浮点类型，以便进行统计计算
                if value.dtype not in [torch.float, torch.float16, torch.float32, torch.float64]:
                    value = value.float()
                
                try:
                    stats = {
                        'mean': float(value.mean().item()),
                        'std': float(value.std().item()) if value.std().item() == value.std().item() else 0.0,  # 避免NaN
                        'min': float(value.min().item()),
                        'max': float(value.max().item()),
                        'shape': list(value.shape)
                    }
                    teacher_stats[key] = stats
                except RuntimeError as e:
                    print(f"警告: 计算教师模型'{key}'统计信息时出错: {e}")
        
        # 分析学生模型中间层
        for key, value in student_intermediate.items():
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                # 确保张量是浮点类型
                if value.dtype not in [torch.float, torch.float16, torch.float32, torch.float64]:
                    value = value.float()
                
                try:
                    stats = {
                        'mean': float(value.mean().item()),
                        'std': float(value.std().item()) if value.std().item() == value.std().item() else 0.0,  # 避免NaN
                        'min': float(value.min().item()),
                        'max': float(value.max().item()),
                        'shape': list(value.shape)
                    }
                    student_stats[key] = stats
                except RuntimeError as e:
                    print(f"警告: 计算学生模型'{key}'统计信息时出错: {e}")
        
        # 打印统计数据摘要
        print("\n教师模型特征统计:")
        for key, stats in teacher_stats.items():
            print(f" - {key}: 均值={stats['mean']:.4f}, 标准差={stats['std']:.4f}")
        
        print("\n学生模型特征统计:")
        for key, stats in student_stats.items():
            print(f" - {key}: 均值={stats['mean']:.4f}, 标准差={stats['std']:.4f}")
        
        return {
            'teacher_outputs': teacher_outputs,
            'student_features': test_features,
            'student_intermediate': student_intermediate,
            'teacher_stats': teacher_stats,
            'student_stats': student_stats
        }

    def plot_prediction_comparison(self, results: Dict[str, Any] = None):
        """可视化预测结果比较"""
        if results is None:
            results = self.compare_prediction_performance()
            
        teacher_preds = results['teacher_preds']
        student_preds = results['student_preds']
        
        # 创建ROC曲线
        plt.figure(figsize=(12, 5))
        
        # 绘制ROC曲线
        plt.subplot(1, 2, 1)
        fpr_teacher, tpr_teacher, _ = roc_curve(self.y_test, teacher_preds)
        fpr_student, tpr_student, _ = roc_curve(self.y_test, student_preds)
        
        plt.plot(fpr_teacher, tpr_teacher, label=f'教师模型 (AUC = {results["teacher_auc"]:.4f})')
        plt.plot(fpr_student, tpr_student, label=f'学生模型 (AUC = {results["student_auc"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('假正率 (FPR)')
        plt.ylabel('真正率 (TPR)')
        plt.title('ROC曲线比较')
        plt.legend()
        
        # 绘制预测分布
        plt.subplot(1, 2, 2)
        plt.hist(teacher_preds, alpha=0.5, bins=50, label='教师模型')
        plt.hist(student_preds, alpha=0.5, bins=50, label='学生模型')
        plt.xlabel('预测概率')
        plt.ylabel('频率')
        plt.title('预测分布比较')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('prediction_comparison.png')
        print("预测比较图已保存为: prediction_comparison.png")
        plt.close()
        
    def visualize_model_comparison(self):
        """可视化模型比较"""
        # 获取比较数据
        size_results = self.compare_model_size()
        speed_results = self.compare_inference_speed()
        perf_results = self.compare_prediction_performance()
        
        # 创建比较图表
        plt.figure(figsize=(15, 12))
        
        # 1. 模型大小比较
        plt.subplot(2, 2, 1)
        bars = plt.bar(['teacher_params', 'student_params'], 
                      [size_results['teacher_params'], size_results['student_params']])
        plt.title('parameter count comparison')
        plt.ylabel('parameter count')
        plt.yscale('log')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:,}',
                    ha='center', va='bottom', rotation=0)
        
        # 2. 推理时间比较
        plt.subplot(2, 2, 2)
        bars = plt.bar(['Teacher', 'Student'], 
                      [speed_results['teacher_avg_time_ms'], speed_results['student_avg_time_ms']])
        plt.title(f'Average Inference Speedup (提速 {speed_results["speedup_ratio"]:.2f}x)')
        plt.ylabel('Time (miniseconds)')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f} ms',
                    ha='center', va='bottom', rotation=0)
        
        # 3. AUC比较
        plt.subplot(2, 2, 3)
        bars = plt.bar(['Teacher', 'Student'], 
                      [perf_results['teacher_auc'], perf_results['student_auc']])
        plt.title('AUC COmparison')
        plt.ylabel('AUC')
        plt.ylim(0.5, 1.0)
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', rotation=0)
        
        # 4. Log Loss比较
        plt.subplot(2, 2, 4)
        bars = plt.bar(['Teacher', 'Student'], 
                      [perf_results['teacher_loss'], perf_results['student_loss']])
        plt.title('Log Loss比较')
        plt.ylabel('Log Loss')
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', rotation=0)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        print("模型比较图已保存为: model_comparison.png")
        plt.close()
        
    def run_comprehensive_comparison(self):
        """运行全面比较并生成报告"""
        print("\n开始全面模型比较...")
        
        # 1. 比较模型大小
        size_results = self.compare_model_size()
        
        # 2. 比较推理速度
        speed_results = self.compare_inference_speed()
        
        # 3. 比较预测性能
        perf_results = self.compare_prediction_performance()
        
        # 4. 比较特征表示
        feat_results = self.compare_feature_representations()
        
        # 5. 可视化比较结果
        self.plot_prediction_comparison(perf_results)
        self.visualize_model_comparison()
        
        # 6. 生成摘要报告
        print("\n=== 模型比较摘要 ===")
        print(f"参数量减少: {size_results['param_reduction']:.2f}%")
        print(f"模型大小减少: {size_results['size_reduction']:.2f}%")
        print(f"推理加速: {speed_results['speedup_ratio']:.2f}x")
        print(f"性能差距(AUC): {perf_results['auc_difference']:.4f}")
        
        return {
            'size_results': size_results,
            'speed_results': speed_results,
            'perf_results': perf_results,
            'feat_results': feat_results
        }


if __name__ == "__main__":
    # 获取当前脚本所在目录的上级目录
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 构建路径
    data_path = os.path.join(base_dir, 'data', 'avazu', 'avazu_sample.csv')
    teacher_model_path = os.path.join(base_dir, 'models', 'teacher_model.pth')
    student_model_path = os.path.join(base_dir, 'models', 'student_model.pth')
    
    # 检查路径是否存在
    paths_exist = True
    for path in [data_path, teacher_model_path, student_model_path]:
        if not os.path.exists(path):
            print(f"错误: 文件不存在: {path}")
            paths_exist = False
    
    if not paths_exist:
        print("请确保所有必要文件存在后再运行此脚本")
        sys.exit(1)
    
    print(f"使用以下路径:")
    print(f"- 数据文件: {data_path}")
    print(f"- 教师模型: {teacher_model_path}")
    print(f"- 学生模型: {student_model_path}")
    
    # 创建比较器并运行比较
    comparator = ModelComparator(
        teacher_model_path=teacher_model_path,
        student_model_path=student_model_path,
        data_path=data_path
    )
    
    # 运行全面比较
    results = comparator.run_comprehensive_comparison()