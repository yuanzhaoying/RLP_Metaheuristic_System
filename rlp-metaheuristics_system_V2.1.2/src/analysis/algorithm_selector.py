"""
算法选择模块
包含四个关键阶段：数据构建、模型训练、性能评估、解释性分析
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False


class AlgorithmSelector:
    """算法选择器"""
    
    def __init__(self, model_type='decision_tree', random_state=42):
        """
        初始化算法选择器
        
        参数:
            model_type: 机器学习模型类型
            random_state: 随机种子
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.algorithm_names = None
        self.X_test = None
        self.y_test = None
        
    def _create_model(self):
        """创建机器学习模型"""
        models = {
            'decision_tree': DecisionTreeClassifier(max_depth=5, random_state=self.random_state),
            'random_forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=self.random_state),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=self.random_state),
            'svm': SVC(kernel='rbf', random_state=self.random_state),
            'knn': KNeighborsClassifier(n_neighbors=5)
        }
        
        return models.get(self.model_type, DecisionTreeClassifier(max_depth=5, random_state=self.random_state))
    
    def build_dataset(self, perf_df, feature_df, instance_col='instance_id', 
                     algo_col='algorithm_name', perf_col='best_objective'):
        """
        第一阶段：数据构建
        
        参数:
            perf_df: 性能数据DataFrame
            feature_df: 特征数据DataFrame
            instance_col: 实例列名
            algo_col: 算法列名
            perf_col: 性能列名
        
        返回:
            perf_matrix: 性能矩阵
            X: 特征矩阵
            y: 标签（最优算法）
        """
        # 如果有重复的(instance, algorithm)组合，取最优值
        if perf_df.duplicated(subset=[instance_col, algo_col]).any():
            perf_df_agg = perf_df.groupby([instance_col, algo_col])[perf_col].min().reset_index()
        else:
            perf_df_agg = perf_df[[instance_col, algo_col, perf_col]]
        
        # 构造性能矩阵
        perf_matrix = perf_df_agg.pivot(
            index=instance_col,
            columns=algo_col,
            values=perf_col
        )
        
        # 过滤掉不可行解
        perf_matrix = perf_matrix[perf_matrix < 1e9].dropna()
        
        # 构造标签（最优算法）
        best_algo = perf_matrix.idxmin(axis=1)
        
        # 特征对齐
        feature_df_indexed = feature_df.set_index(instance_col)
        common_instances = perf_matrix.index.intersection(feature_df_indexed.index)
        
        perf_matrix = perf_matrix.loc[common_instances]
        X = feature_df_indexed.loc[common_instances]
        y = best_algo.loc[common_instances]
        
        # 保存特征名和算法名
        self.feature_names = X.columns.tolist()
        self.algorithm_names = perf_matrix.columns.tolist()
        
        return perf_matrix, X, y
    
    def train_model(self, X, y, test_size=0.3):
        """
        第二阶段：模型训练与预测
        
        参数:
            X: 特征矩阵
            y: 标签
            test_size: 测试集比例
        
        返回:
            X_train, X_test, y_train, y_test: 训练集和测试集
        """
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
        
        # 训练/测试划分
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=self.random_state
        )
        
        # 创建并训练模型
        self.model = self._create_model()
        self.model.fit(X_train, y_train)
        
        # 保存测试集用于特征重要性分析
        self.X_test = X_test
        self.y_test = y_test
        
        return X_train, X_test, y_train, y_test
    
    def evaluate_performance(self, perf_matrix, X_test, y_test):
        """
        第三阶段：性能评估
        
        参数:
            perf_matrix: 性能矩阵
            X_test: 测试集特征
            y_test: 测试集标签
        
        返回:
            results: 评估结果字典
        """
        # 1. VBS（虚拟最佳算法）
        vbs_perf = perf_matrix.loc[X_test.index].min(axis=1)
        vbs_score = vbs_perf.mean()
        
        # 2. SBS（单一最佳算法）
        sbs_algo = perf_matrix.mean(axis=0).idxmin()
        sbs_score = perf_matrix[sbs_algo].mean()
        
        # 3. Selector性能
        y_pred = self.model.predict(X_test)
        
        # Hit Rate
        hit_rate = accuracy_score(y_test, y_pred)
        
        # Selector性能和regret
        selector_perf = []
        regret = []
        
        for inst in X_test.index:
            pred_algo = self.model.predict([X_test.loc[inst]])[0]
            
            # 如果预测的算法不在性能矩阵中，选择SBS
            if pred_algo not in perf_matrix.columns:
                pred_algo = sbs_algo
            
            actual_best = perf_matrix.loc[inst].min()
            perf = perf_matrix.loc[inst, pred_algo]
            
            selector_perf.append(perf)
            regret.append(perf - actual_best)
        
        selector_score = np.mean(selector_perf)
        
        # P90 penalty
        p90_penalty = np.percentile(regret, 90)
        
        # 平均regret
        avg_regret = np.mean(regret)
        
        # 性能提升
        improvement_over_sbs = (sbs_score - selector_score) / sbs_score * 100
        gap_to_vbs = (selector_score - vbs_score) / vbs_score * 100
        
        results = {
            'SBS': {
                'algorithm': sbs_algo,
                'score': sbs_score
            },
            'VBS': {
                'score': vbs_score
            },
            'Selector': {
                'score': selector_score,
                'hit_rate': hit_rate,
                'p90_penalty': p90_penalty,
                'avg_regret': avg_regret,
                'improvement_over_sbs': improvement_over_sbs,
                'gap_to_vbs': gap_to_vbs
            },
            'predictions': y_pred,
            'actual': y_test,
            'regret': regret
        }
        
        return results
    
    def analyze_feature_importance(self):
        """
        第四阶段：特征重要性分析
        
        返回:
            feature_importance: 特征重要性DataFrame
        """
        if self.model is None:
            return None
        
        importance = None
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_).mean(axis=0) if self.model.coef_.ndim > 1 else np.abs(self.model.coef_)
        else:
            from sklearn.inspection import permutation_importance
            try:
                result = permutation_importance(self.model, self.X_test, self.y_test, n_repeats=10, random_state=self.random_state)
                importance = result.importances_mean
            except:
                importance = np.ones(len(self.feature_names)) / len(self.feature_names)
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values(by='importance', ascending=False)
        
        return feature_importance
    
    def instance_space_analysis(self, X, y, method='pca'):
        """
        第四阶段：实例空间分析
        
        参数:
            X: 特征矩阵
            y: 标签
            method: 降维方法 ('pca' 或 'tsne')
        
        返回:
            embedding_df: 降维后的数据
            fig: 可视化图形
        """
        # 数据标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 降维
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=self.random_state, perplexity=min(30, len(X)-1))
        else:
            reducer = PCA(n_components=2, random_state=self.random_state)
        
        embedding = reducer.fit_transform(X_scaled)
        
        # 构造结果DataFrame
        embedding_df = pd.DataFrame({
            'x': embedding[:, 0],
            'y': embedding[:, 1],
            'best_algo': y.values,
            'instance_id': y.index
        })
        
        # 可视化
        fig, ax = plt.subplots(figsize=(10, 8))
        
        algorithms = embedding_df['best_algo'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(algorithms)))
        
        for algo, color in zip(algorithms, colors):
            subset = embedding_df[embedding_df['best_algo'] == algo]
            ax.scatter(subset['x'], subset['y'], label=algo, color=color, alpha=0.6, s=50)
        
        ax.set_xlabel('Dimension 1', fontsize=12)
        ax.set_ylabel('Dimension 2', fontsize=12)
        ax.set_title(f'Instance Space Analysis ({method.upper()})', fontsize=14)
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return embedding_df, fig
    
    def plot_feature_importance(self, top_n=20):
        """
        绘制特征重要性图
        
        参数:
            top_n: 显示前N个重要特征
        
        返回:
            fig: matplotlib图形对象
        """
        feature_importance = self.analyze_feature_importance()
        
        if feature_importance is None:
            return None
        
        # 选择前N个重要特征
        top_features = feature_importance.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(top_features)))
        
        bars = ax.barh(range(len(top_features)), top_features['importance'].values, color=colors)
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'].values, fontsize=10)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importance', fontsize=14)
        ax.grid(True, axis='x', alpha=0.3)
        
        for bar, importance in zip(bars, top_features['importance'].values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{importance:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        
        return fig
    
    def plot_performance_comparison(self, results):
        """
        绘制性能比较图
        
        参数:
            results: 评估结果字典
        
        返回:
            fig: matplotlib图形对象
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 左图：性能比较
        ax1 = axes[0]
        methods = ['SBS', 'Selector', 'VBS']
        scores = [results['SBS']['score'], results['Selector']['score'], results['VBS']['score']]
        colors = ['#ff7f7f', '#7fbf7f', '#7f7fff']
        
        bars = ax1.bar(methods, scores, color=colors, alpha=0.8)
        ax1.set_ylabel('Average Objective Value', fontsize=12)
        ax1.set_title('Performance Comparison', fontsize=14)
        ax1.grid(True, axis='y', alpha=0.3)
        
        for bar, score in zip(bars, scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{score:.2f}', ha='center', va='bottom', fontsize=11)
        
        # 右图：性能指标
        ax2 = axes[1]
        metrics = ['Hit Rate', 'Improvement\nover SBS (%)', 'Gap to\nVBS (%)']
        values = [
            results['Selector']['hit_rate'] * 100,
            results['Selector']['improvement_over_sbs'],
            results['Selector']['gap_to_vbs']
        ]
        colors = ['#7fbf7f', '#7f7fff', '#ff7f7f']
        
        bars = ax2.bar(metrics, values, color=colors, alpha=0.8)
        ax2.set_ylabel('Value', fontsize=12)
        ax2.set_title('Selector Performance Metrics', fontsize=14)
        ax2.grid(True, axis='y', alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        for bar, value in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{value:.2f}', ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        
        return fig


def analyze_selector(perf_df, feature_df, model_type='decision_tree', 
                    instance_col='instance_id', algo_col='algorithm_name', 
                    perf_col='best_objective', test_size=0.3, 
                    random_state=42, use_tsne=False,
                    calc_sbs=True, calc_vbs=True, calc_selector=True,
                    calc_winner_hit=True, calc_penalty=True, calc_risk_lambda=False,
                    feature_importance=True, shap=True, perm_importance=True, isa=True):
    """
    完整的算法选择分析流程
    
    参数:
        perf_df: 性能数据DataFrame
        feature_df: 特征数据DataFrame
        model_type: 机器学习模型类型
        instance_col: 实例列名
        algo_col: 算法列名
        perf_col: 性能列名
        test_size: 测试集比例
        random_state: 随机种子
        use_tsne: 是否使用t-SNE降维
        calc_sbs: 是否计算SBS
        calc_vbs: 是否计算VBS
        calc_selector: 是否计算Selector
        calc_winner_hit: 是否计算Winner Hit Rate
        calc_penalty: 是否计算Penalty
        calc_risk_lambda: 是否计算Risk Lambda Ablation
        feature_importance: 是否计算特征重要性
        shap: 是否进行SHAP分析
        perm_importance: 是否进行Permutation Importance分析
        isa: 是否进行实例空间分析
    
    返回:
        results: 分析结果字典
    """
    selector = AlgorithmSelector(model_type=model_type, random_state=random_state)
    
    perf_matrix, X, y = selector.build_dataset(perf_df, feature_df, instance_col, algo_col, perf_col)
    
    X_train, X_test, y_train, y_test = selector.train_model(X, y, test_size)
    
    results = selector.evaluate_performance(perf_matrix, X_test, y_test)
    
    results['feature_importance'] = selector.analyze_feature_importance() if feature_importance else None
    
    if shap:
        try:
            import shap
            explainer = shap.TreeExplainer(selector.model) if hasattr(selector.model, 'estimators_') else shap.KernelExplainer(selector.model.predict_proba, X_train[:100])
            shap_values = explainer.shap_values(selector.X_test)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            results['shap_values'] = pd.DataFrame({
                'feature': selector.feature_names,
                'mean_abs_shap': mean_abs_shap
            }).sort_values(by='mean_abs_shap', ascending=False)
        except Exception as e:
            print(f"SHAP分析错误: {e}")
            results['shap_values'] = None
    else:
        results['shap_values'] = None
    
    if perm_importance:
        try:
            from sklearn.inspection import permutation_importance as sk_perm_importance
            perm_result = sk_perm_importance(selector.model, selector.X_test, selector.y_test, n_repeats=10, random_state=random_state)
            results['permutation_importance'] = pd.DataFrame({
                'feature': selector.feature_names,
                'importance_mean': perm_result.importances_mean,
                'importance_std': perm_result.importances_std
            }).sort_values(by='importance_mean', ascending=False)
        except Exception as e:
            print(f"Permutation Importance分析错误: {e}")
            results['permutation_importance'] = None
    else:
        results['permutation_importance'] = None
    
    embedding_df, isa_fig = selector.instance_space_analysis(X, y, method='tsne' if use_tsne else 'pca') if isa else (None, None)
    
    if isa and embedding_df is not None:
        results['isa_results'] = {
            'embedding': embedding_df,
            'method': 't-SNE' if use_tsne else 'PCA',
            'total_instances': len(embedding_df)
        }
    else:
        results['isa_results'] = None
    
    fi_fig = selector.plot_feature_importance(top_n=20) if feature_importance else None
    
    pc_fig = selector.plot_performance_comparison(results)
    
    results['embedding'] = embedding_df
    results['selector'] = selector
    results['figures'] = {
        'instance_space': isa_fig,
        'feature_importance': fi_fig,
        'performance_comparison': pc_fig
    }
    
    return results
