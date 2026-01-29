import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.calibration import calibration_curve
from sklearn.metrics import log_loss, brier_score_loss
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, truncnorm, nbinom, gamma
from scipy.special import logit, expit
import warnings
warnings.filterwarnings('ignore')

def show_figure_nonblocking():
    """非阻塞显示图形"""
    plt.show(block=False)
    plt.pause(3)

def setup_plot_style():
        """设置绘图样式和中文字体"""
        import platform
        
        # 设置中文字体
        system = platform.system()
        if system == 'Windows':
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
            print("ok")
        elif system == 'Darwin':  # macOS
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC']
        else:  # Linux
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
        
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置其他样式
        plt.rcParams['figure.figsize'] = [10, 6]
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
        
        print("绘图样式设置完成")


# 设置中文显示和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

setup_plot_style()

# 1. 数据加载和预处理
print("步骤1: 加载和预处理数据...")
# 注意：实际使用时需要指定正确的文件路径
# df = pd.read_csv('2024_Wimbledon_featured_matches.csv')
# 这里我们使用提供的示例数据，假设已经加载到变量df中
# 由于数据量较大，我们只取部分数据进行演示

# 选择一场比赛进行分析（示例：Alcaraz vs Jarry）
df = pd.read_csv('2024_Wimbledon_featured_matches.csv')
df =df[(df['player1'] == 'Andrey Rublev') & (df['player2'] == 'David Goffin')].copy()

# 2. 数据清洗和特征工程
print("步骤2: 特征工程...")

# 2.1 处理比分变量
def map_score(score_str):
    """将网球比分映射为数值"""
    if pd.isna(score_str):
        return 0
    if str(score_str).upper() == 'AD':
        return 4  # Advantage
    try:
        return int(score_str)
    except:
        # 处理特殊情况
        if score_str == '0':
            return 0
        elif score_str == '15':
            return 1
        elif score_str == '30':
            return 2
        elif score_str == '40':
            return 3
        else:
            return 0

# 应用映射
df['p1_score_num'] = df['p1_score'].apply(map_score)
df['p2_score_num'] = df['p2_score'].apply(map_score)

# 2.2 计算局内点数差
df['point_diff_in_game'] = df['p1_score_num'] - df['p2_score_num']

# 2.3 识别关键分状态
def get_game_point(row, player='p1'):
    """判断是否为局点"""
    if player == 'p1':
        my_score = row['p1_score_num']
        opp_score = row['p2_score_num']
    else:
        my_score = row['p2_score_num']
        opp_score = row['p1_score_num']
    
    # 在40分制中
    if my_score >= 3 and my_score > opp_score:
        return 1
    return 0

df['game_point_p1'] = df.apply(lambda x: get_game_point(x, 'p1'), axis=1)
df['game_point_p2'] = df.apply(lambda x: get_game_point(x, 'p2'), axis=1)

# 判断是否平分局
df['deuce_flag'] = ((df['p1_score_num'] >= 3) & (df['p2_score_num'] >= 3) & 
                    (df['p1_score_num'] == df['p2_score_num'])).astype(int)

# 2.4 发球特征
df['is_p1_serving'] = (df['server'] == 1).astype(int)
df['second_serve'] = (df['serve_no'] == 2).astype(int)

# 2.5 标准化跑动距离
if 'p1_distance_run' in df.columns and 'p2_distance_run' in df.columns:
    df['total_distance'] = df['p1_distance_run'] + df['p2_distance_run']
    df['p1_distance_norm'] = df['p1_distance_run'] / df['total_distance'].clip(lower=0.1)
    df['p2_distance_norm'] = df['p2_distance_run'] / df['total_distance'].clip(lower=0.1)
    df['distance_diff'] = df['p1_distance_norm'] - df['p2_distance_norm']

# 2.6 计算总分差
df['total_points_won'] = df['p1_points_won'] + df['p2_points_won']
df['p1_point_share'] = df['p1_points_won'] / df['total_points_won'].clip(lower=1)

# 2.7 发球速度特征
if 'speed_mph' in df.columns:
    df['speed_mph'] = pd.to_numeric(df['speed_mph'], errors='coerce')
    # 填补缺失值
    df['speed_mph'] = df['speed_mph'].fillna(df['speed_mph'].median())

# 2.8 回合长度特征
if 'rally_count' in df.columns:
    df['log_rally_count'] = np.log1p(df['rally_count'])

# 3. 构建滞后特征（短期状态）
print("步骤3: 构建滞后特征...")

def add_lag_features(df, window_sizes=[3, 5]):
    """添加滞后窗口特征"""
    df = df.sort_values(['set_no', 'game_no', 'point_no']).reset_index(drop=True)
    
    # 创建副本用于计算滞后特征
    df_lag = df.copy()
    
    # 计算赢分标志（用于滞后统计）
    df_lag['p1_won_point'] = (df_lag['point_victor'] == 1).astype(int)
    
    for window in window_sizes:
        # 过去k分的赢球率
        df_lag[f'p1_win_rate_last_{window}'] = df_lag['p1_won_point'].rolling(
            window=window, min_periods=1).mean().shift(1).fillna(0.5)
        
        # 过去k分的平均回合长度
        if 'rally_count' in df.columns:
            df_lag[f'avg_rally_last_{window}'] = df_lag['rally_count'].rolling(
                window=window, min_periods=1).mean().shift(1).fillna(df_lag['rally_count'].mean())
        
        # 过去k分的平均跑动差
        if 'distance_diff' in df.columns:
            df_lag[f'avg_distance_diff_last_{window}'] = df_lag['distance_diff'].rolling(
                window=window, min_periods=1).mean().shift(1).fillna(0)
    
    return df_lag.drop(columns=['p1_won_point'])

df = add_lag_features(df)

# 4. 准备模型特征和目标变量
print("步骤4: 准备模型训练数据...")

# 定义特征列
feature_cols = [
    # 比分状态
    'p1_sets', 'p2_sets', 'p1_games', 'p2_games',
    'point_diff_in_game', 'game_point_p1', 'game_point_p2', 'deuce_flag',
    
    # 发球状态
    'is_p1_serving', 'second_serve',
    
    # 累计统计数据
    'p1_point_share',
    
    # 回合特征
    'rally_count',
    
    # 滞后特征
    'p1_win_rate_last_3', 'p1_win_rate_last_5',
    'avg_rally_last_3', 'avg_distance_diff_last_3'
]

# 只保留存在的特征
feature_cols = [col for col in feature_cols if col in df.columns]

# 添加发球速度和跑动特征（如果存在）
if 'speed_mph' in df.columns:
    feature_cols.append('speed_mph')
if 'distance_diff' in df.columns:
    feature_cols.append('distance_diff')

# 目标变量：当前分是否由P1赢得
df['target'] = (df['point_victor'] == 1).astype(int)

# 处理缺失值
df_features = df[feature_cols].copy()
df_features = df_features.fillna(df_features.median())

# 5. 训练随机森林模型（使用分组交叉验证）
print("步骤5: 训练随机森林模型...")

# 创建比赛ID分组（这里只有一场比赛，所以使用时间序列交叉验证）
# 在实际应用中，如果有多个比赛，应该按比赛ID分组
if len(df) > 100:
    # 使用时间序列划分：前70%训练，后30%测试
    split_idx = int(len(df) * 0.7)
    X_train = df_features.iloc[:split_idx]
    y_train = df['target'].iloc[:split_idx]
    X_test = df_features.iloc[split_idx:]
    y_test = df['target'].iloc[split_idx:]
    
    # 训练模型
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    
    # 预测整个序列（为了保持连续性，我们用训练好的模型预测整个序列）
    df['pred_prob'] = rf_model.predict_proba(df_features)[:, 1]
    
else:
    # 如果数据量小，直接训练并预测
    rf_model = RandomForestClassifier(
        n_estimators=50,
        max_depth=8,
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(df_features, df['target'])
    df['pred_prob'] = rf_model.predict_proba(df_features)[:, 1]

# 6. 计算比赛流指标
print("步骤6: 计算比赛流指标...")

# 计算残差（实际结果 - 预测概率）
df['residual'] = df['target'] - df['pred_prob']

# 计算EWMA（指数加权移动平均）
def compute_ewma(series, df, alpha_normal=0.15, alpha_critical=0.3, critical_columns=['is_break_point', 'is_set_point', 'is_match_point']):
    ewma = np.zeros_like(series, dtype=float)
    ewma[0] = series[0]
    
    for i in range(1, len(series)):
        # 检查是否为关键分
        is_critical = False
        for col in critical_columns:
            if col in df.columns and df.iloc[i].get(col, 0) == 1:
                is_critical = True
                break
        
        # 根据是否为关键分选择alpha
        alpha = alpha_critical if is_critical else alpha_normal
        
        ewma[i] = (1 - alpha) * ewma[i-1] + alpha * series[i]
    
    return ewma

df['momentum'] = compute_ewma(df['residual'].values, df, alpha_normal=0.15,alpha_critical=0.3,critical_columns=['is_break_point', 'is_set_point','is_match_point'])

# 7. 变量分布拟合
print("步骤7: 拟合变量分布...")

# 7.1 发球速度分布（截断正态）
if 'speed_mph' in df.columns:
    speed_data = df['speed_mph'].dropna()
    if len(speed_data) > 10:
        # 拟合截断正态分布
        a, b = 70, 150  # 合理速度范围
        loc, scale = norm.fit(speed_data)
        # 调整参数使在截断范围内
        a_norm, b_norm = (a - loc) / scale, (b - loc) / scale
        speed_dist_params = {
            'type': 'truncnorm',
            'params': (a_norm, b_norm, loc, scale)
        }
    else:
        speed_dist_params = None

# 7.2 回合长度分布（负二项分布）
if 'rally_count' in df.columns:
    rally_data = df['rally_count'].dropna().astype(int)
    if len(rally_data) > 10:
        # 负二项分布参数估计
        mean_rally = rally_data.mean()
        var_rally = rally_data.var()
        
        if var_rally > mean_rally:
            p_nb = mean_rally / var_rally
            n_nb = mean_rally * p_nb / (1 - p_nb)
            rally_dist_params = {
                'type': 'nbinom',
                'params': (n_nb, p_nb)
            }
        else:
            # 如果方差不大，使用泊松分布
            rally_dist_params = {
                'type': 'poisson',
                'params': (mean_rally,)
            }
    else:
        rally_dist_params = None

# 7.3 跑动距离分布（Gamma分布）
if 'p1_distance_run' in df.columns:
    distance_data = df['p1_distance_run'].dropna()
    if len(distance_data) > 10:
        # Gamma分布参数估计
        a_gamma, loc_gamma, scale_gamma = gamma.fit(distance_data)
        distance_dist_params = {
            'type': 'gamma',
            'params': (a_gamma, loc_gamma, scale_gamma)
        }
    else:
        distance_dist_params = None

# 8. 模拟比赛流（技术基准模型版本）
print("步骤8: 基于技术基准模型模拟比赛流...")

def create_baseline_model():
    """创建技术基准模型（不包含滞后特征）"""
    # 定义技术特征（不包含任何滞后/状态特征）
    technical_features = [
        # 比分状态
        'is_p1_serving',
        'second_serve',
        'point_diff_in_game',
        'game_point_p1',
        'game_point_p2',
        'deuce_flag',
        'p1_sets', 'p2_sets',
        'p1_games', 'p2_games',
        'p1_point_share',  # 累计统计，但通常是技术能力的反映
    ]
    
    # 添加物理特征（如果可用）
    if 'speed_mph' in df.columns:
        technical_features.append('speed_mph')
    if 'distance_diff' in df.columns:
        technical_features.append('distance_diff')
    if 'rally_count' in df.columns:
        technical_features.append('rally_count')
    
    # 只选择存在的特征
    technical_features = [f for f in technical_features if f in df.columns]
    
    # 准备数据
    X = df[technical_features].copy()
    X = X.fillna(X.median())
    y = df['target']
    
    # 使用随机森林作为基准模型（与完整模型相同类型，但不含滞后特征）
    baseline_rf = RandomForestClassifier(
        n_estimators=50,  # 比完整模型少一些树，避免过拟合
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    baseline_rf.fit(X, y)
    
    # 计算基准预测概率
    df['baseline_prob'] = baseline_rf.predict_proba(X)[:, 1]
    
    print(f"基准模型特征 ({len(technical_features)}个): {technical_features}")
    print(f"基准模型准确率: {baseline_rf.score(X, y):.3f}")
    print(f"基准模型Brier Score: {brier_score_loss(y, df['baseline_prob']):.4f}")
    
    # 对比完整模型
    print(f"完整模型准确率: {rf_model.score(df_features, y):.3f}")
    print(f"完整模型Brier Score: {brier_score_loss(y, df['pred_prob']):.4f}")
    
    # 分析特征重要性
    baseline_importance = pd.DataFrame({
        'feature': technical_features,
        'importance': baseline_rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n基准模型特征重要性 (前5):")
    print(baseline_importance.head(5))
    
    return baseline_rf, technical_features, X 

baseline_model, tech_features, X_baseline = create_baseline_model()

def simulate_match_baseline(num_simulations=50):
    """基于技术基准模型模拟比赛，生成比赛流置信带"""
    n_points = len(df)
    momentum_sims = np.zeros((num_simulations, n_points))
    
    # 准备技术特征数据
    baseline_features = X_baseline.copy()
    
    for sim in range(num_simulations):
        # 初始化模拟序列
        sim_residuals = np.zeros(n_points)
        
        for i in range(n_points):
            # 获取当前局面的技术特征
            current_tech_features = baseline_features.iloc[i].copy()
            
            # 使用技术基准模型预测概率
            tech_pred_prob = baseline_model.predict_proba(
                current_tech_features.values.reshape(1, -1)
            )[0, 1]
            
            # 添加不确定性（模拟比赛的随机性）
            # 发球局和接发球局的不确定性不同
            if df['is_p1_serving'].iloc[i] == 1:
                uncertainty = 0.08  # 发球局相对确定
            else:
                uncertainty = 0.12  # 接发球局更不确定
            
            # 关键分不确定性更大
            is_critical = False
            for col in ['is_break_point', 'is_set_point', 'is_match_point']:
                if col in df.columns and df.iloc[i].get(col, 0) == 1:
                    uncertainty += 0.06
                    is_critical = True
                    break
            
            # 应用随机扰动
            noise = np.random.normal(0, uncertainty)
            simulated_prob = tech_pred_prob + noise
            simulated_prob = np.clip(simulated_prob, 0.15, 0.85)
            
            # 模拟本分结果
            sim_result = np.random.binomial(1, simulated_prob)
            
            # 计算模拟残差（与完整模型预测比较）
            # 注意：这里使用完整模型的预测概率作为基准
            full_model_pred_prob = df['pred_prob'].iloc[i]
            sim_residual = sim_result - full_model_pred_prob
            sim_residuals[i] = sim_residual
        
        # 计算模拟的momentum
        momentum_sims[sim, :] = compute_ewma(
            sim_residuals, df, 
            alpha_normal=0.15, 
            alpha_critical=0.3,
            critical_columns=['is_break_point', 'is_set_point', 'is_match_point']
        )
        
        if (sim + 1) % 10 == 0:
            print(f"模拟进度：{sim+1}/{num_simulations}")
    
    # 计算置信区间
    lower_bound = np.percentile(momentum_sims, 2.5, axis=0)
    upper_bound = np.percentile(momentum_sims, 97.5, axis=0)
    median = np.percentile(momentum_sims, 50, axis=0)
    
    return {
        'momentum_sims': momentum_sims,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'median': median
    }

# 运行基准模拟（增加模拟次数以提高统计可靠性）
print("\n开始基准模型模拟...")
sim_results_baseline = simulate_match_baseline(num_simulations=50)

def calculate_exceedance(real_momentum, sim_results, percentile=95):
    """计算真实序列超出模拟置信带的比例"""
    alpha = (100 - percentile) / 2
    lower = np.percentile(sim_results['momentum_sims'], alpha, axis=0)
    upper = np.percentile(sim_results['momentum_sims'], 100 - alpha, axis=0)
    
    # 找出超出置信带的点
    exceed_lower = real_momentum < lower
    exceed_upper = real_momentum > upper
    exceed_total = np.sum(exceed_lower | exceed_upper)
    
    # 计算比例
    exceed_percent = exceed_total / len(real_momentum) * 100
    
    return exceed_percent, lower, upper

real_momentum = df['momentum'].values
exceed_percent, lower_95, upper_95 = calculate_exceedance(real_momentum, sim_results_baseline, percentile=95)

print(f"\n基准模型模拟结果:")
print(f"真实momentum超出95%置信带的比例: {exceed_percent:.1f}%")

# 统计显著性检验
from scipy.stats import binomtest
n_points = len(df)
n_exceed = int(exceed_percent / 100 * n_points)
expected_exceed = int(0.05 * n_points)  # 期望5%超出

# 二项检验
p_value = binomtest(n_exceed, n_points, p=0.05).pvalue
print(f"超出点数: {n_exceed}/{n_points} (期望: {expected_exceed})")
print(f"二项检验p值: {p_value:.6f}")

if p_value < 0.05:
    print("→ 结果统计显著 (p < 0.05): 拒绝零假设，momentum存在")
    significance_level = "高度显著" if p_value < 0.01 else "显著"
    print(f"→ {significance_level}: 比赛中的表现波动不是随机的")
else:
    print("→ 结果不显著: 无法拒绝零假设，可能只是随机波动")

# 9. 可视化结果
print("步骤9: 生成可视化...")

# 创建时间轴（使用点序号）
time_points = np.arange(len(df))
# 9.1 创建主图：比赛流分析（添加基准模型对比）
fig, axes = plt.subplots(4, 1, figsize=(14, 16), height_ratios=[1, 1, 1, 1])

# 子图1：点胜率预测与实际结果（保持不变）
ax1 = axes[0]
ax1.plot(time_points, df['pred_prob'], label='完整模型预测概率', alpha=0.7, linewidth=2)
ax1.plot(time_points, df['baseline_prob'], '--', label='基准模型预测概率', alpha=0.7, linewidth=1.5, color='green')
ax1.scatter(time_points, df['target'], alpha=0.5, s=20, label='实际结果', color='red', marker='x')
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax1.set_ylabel('赢分概率')
ax1.set_title('模型预测对比: 完整模型 vs 技术基准模型')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# 子图2：比赛流指标（动量）- 完整模型模拟
ax2 = axes[1]
# 真实动量
ax2.plot(time_points, df['momentum'], label='真实比赛流', color='darkred', linewidth=2.5)
# 基准模型模拟置信带
ax2.fill_between(time_points, sim_results_baseline['lower_bound'], sim_results_baseline['upper_bound'], 
                  alpha=0.4, color='green', label='技术基准模型95%置信区间')
ax2.plot(time_points, sim_results_baseline['median'], '--', color='green', alpha=0.7, 
         label='基准模型模拟中位数', linewidth=1.5)

ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.set_ylabel('比赛流指标 (M_t)')
ax2.set_title(f'比赛流分析: 真实 vs 技术基准模型模拟 (超出: {exceed_percent:.1f}%, p={p_value:.4f})')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# 标注超出置信带的区域
exceed_mask = (real_momentum < lower_95) | (real_momentum > upper_95)
ax2.scatter(time_points[exceed_mask], real_momentum[exceed_mask], 
           s=25, c='red', alpha=0.6, zorder=5, label=f'超出置信带点 ({exceed_percent:.1f}%)')

# 标注关键事件
game_wins = df[df['game_victor'] != 0].index
set_wins = df[df['set_victor'] != 0].index

for idx in game_wins:
    ax2.axvline(x=idx, color='green', linestyle=':', alpha=0.3, linewidth=0.8)
for idx in set_wins:
    ax2.axvline(x=idx, color='red', linestyle='--', alpha=0.5, linewidth=1.2)

# 子图3：残差分布对比
ax3 = axes[2]
# 计算两种模型的残差
df['residual_full'] = df['target'] - df['pred_prob']
df['residual_baseline'] = df['target'] - df['baseline_prob']

# 绘制残差分布
ax3.hist(df['residual_full'], bins=30, alpha=0.5, label='完整模型残差', color='blue', density=True)
ax3.hist(df['residual_baseline'], bins=30, alpha=0.5, label='基准模型残差', color='green', density=True)
ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5)
ax3.set_xlabel('残差 (实际 - 预测)')
ax3.set_ylabel('频率')
ax3.set_title('残差分布对比')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 添加统计信息
full_residual_mean = df['residual_full'].mean()
baseline_residual_mean = df['residual_baseline'].mean()
full_residual_std = df['residual_full'].std()
baseline_residual_std = df['residual_baseline'].std()

ax3.text(0.02, 0.95, f'完整模型: 均值={full_residual_mean:.3f}, 标准差={full_residual_std:.3f}', 
         transform=ax3.transAxes, fontsize=9, verticalalignment='top')
ax3.text(0.02, 0.88, f'基准模型: 均值={baseline_residual_mean:.3f}, 标准差={baseline_residual_std:.3f}', 
         transform=ax3.transAxes, fontsize=9, verticalalignment='top')

# 子图4：累计残差对比
ax4 = axes[3]
# 计算累计残差
cumulative_full = np.cumsum(df['residual_full'])
cumulative_baseline = np.cumsum(df['residual_baseline'])

ax4.plot(time_points, cumulative_full, color='blue', linewidth=2, label='完整模型累计残差')
ax4.plot(time_points, cumulative_baseline, color='green', linewidth=2, label='基准模型累计残差')
ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
ax4.set_xlabel('点数序号')
ax4.set_ylabel('累计残差')
ax4.set_title('累计残差对比: 趋势分析')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 添加显著性标注
if p_value < 0.05:
    significance_text = f'统计显著 (p={p_value:.4f})'
    color = 'red'
else:
    significance_text = f'不显著 (p={p_value:.4f})'
    color = 'gray'

fig.suptitle(f'比赛流分析: {df["player1"].iloc[0]} vs {df["player2"].iloc[0]}\n' +
             f'技术基准模型检验 - {significance_text}', fontsize=14, y=0.98)

plt.tight_layout()
plt.savefig('match_flow_analysis_with_baseline.png', dpi=300, bbox_inches='tight')
show_figure_nonblocking()

# 10. 比赛流解读
print("\n" + "="*60)
print("比赛流分析结果解读")
print("="*60)

# 10.1 模型性能
print(f"\n1. 模型性能指标:")
print(f"   - Brier Score: {brier_score_loss(df['target'], df['pred_prob']):.4f}")
print(f"   - Log Loss: {log_loss(df['target'], df['pred_prob']):.4f}")
print(f"   - 平均预测概率: {df['pred_prob'].mean():.3f}")

# 10.2 比赛流统计
print(f"\n2. 比赛流统计:")
print(f"   - 平均比赛流指标: {df['momentum'].mean():.4f}")
print(f"   - 比赛流标准差: {df['momentum'].std():.4f}")
print(f"   - 最大正向比赛流: {df['momentum'].max():.4f} (发生在第{df['momentum'].idxmax()}分)")
print(f"   - 最大负向比赛流: {df['momentum'].min():.4f} (发生在第{df['momentum'].idxmin()}分)")

# 10.3 关键阶段分析
print(f"\n3. 关键阶段分析:")

# 找出比赛流持续为正的阶段
positive_momentum_streaks = []
current_streak = []
for i, val in enumerate(df['momentum']):
    if val > 0:
        current_streak.append(i)
    elif current_streak:
        if len(current_streak) >= 5:  # 只考虑至少持续5分的阶段
            positive_momentum_streaks.append(current_streak.copy())
        current_streak = []

if positive_momentum_streaks:
    print(f"   - P1有{len(positive_momentum_streaks)}次持续的比赛流优势阶段")
    for i, streak in enumerate(positive_momentum_streaks[:3]):  # 只显示前3个
        avg_momentum = df['momentum'].iloc[streak].mean()
        print(f"     阶段{i+1}: 第{streak[0]}-{streak[-1]}分，平均比赛流={avg_momentum:.3f}")

# 找出比赛流持续为负的阶段
negative_momentum_streaks = []
current_streak = []
for i, val in enumerate(df['momentum']):
    if val < 0:
        current_streak.append(i)
    elif current_streak:
        if len(current_streak) >= 5:
            negative_momentum_streaks.append(current_streak.copy())
        current_streak = []

if negative_momentum_streaks:
    print(f"   - P2有{len(negative_momentum_streaks)}次持续的比赛流优势阶段")
    for i, streak in enumerate(negative_momentum_streaks[:3]):
        avg_momentum = df['momentum'].iloc[streak].mean()
        print(f"     阶段{i+1}: 第{streak[0]}-{streak[-1]}分，平均比赛流={avg_momentum:.3f}")

# 10.4 发球优势分析
print(f"\n4. 发球优势分析:")
p1_serving_mask = df['is_p1_serving'] == 1
p1_serving_win_rate = df.loc[p1_serving_mask, 'target'].mean()
p2_serving_win_rate = 1 - df.loc[~p1_serving_mask, 'target'].mean()

print(f"   - P1发球时赢分率: {p1_serving_win_rate:.3f}")
print(f"   - P2发球时赢分率: {p2_serving_win_rate:.3f}")
print(f"   - 发球优势差值: {abs(p1_serving_win_rate - p2_serving_win_rate):.3f}")

# 10.5 模拟结果分析 - 添加基准模型分析
print(f"\n5. 基准模型模拟分析:")
print(f"   - 真实比赛流在 {exceed_percent:.1f}% 的点数上超出技术基准模型95%置信带")
print(f"   - 二项检验p值: {p_value:.6f}")

if p_value < 0.05:
    if p_value < 0.01:
        significance = "高度显著"
    elif p_value < 0.05:
        significance = "显著"
    
    print(f"   → {significance}: 拒绝零假设，比赛中的表现波动不是随机的")
    print(f"   → 这表明存在系统性表现优势（momentum），不能仅由技术因素解释")
    
    # 进一步分析超出区域
    if exceed_percent > 15:
        print(f"   → 强证据: 超出比例高 ({exceed_percent:.1f}%)，momentum效应明显")
    elif exceed_percent > 10:
        print(f"   → 中等证据: 超出比例中等 ({exceed_percent:.1f}%)")
    else:
        print(f"   → 弱证据: 超出比例较低但有统计显著性")
else:
    print(f"   → 无法拒绝零假设: 比赛波动可能只是随机因素和技术轮转的结果")
    print(f"   → 对于这场比赛，没有足够证据表明存在momentum效应")

print(f"\n6. 模型对比:")
print(f"   - 完整模型准确率: {rf_model.score(df_features, df['target']):.4f}")
print(f"   - 基准模型准确率: {baseline_model.score(X_baseline, df['target']):.4f}")
print(f"   - 准确率提升: {(rf_model.score(df_features, df['target']) - baseline_model.score(X_baseline, df['target'])):.4f}")
print(f"   - 完整模型Brier Score: {brier_score_loss(df['target'], df['pred_prob']):.4f}")
print(f"   - 基准模型Brier Score: {brier_score_loss(df['target'], df['baseline_prob']):.4f}")

# 保存详细结果到CSV
output_df = df[['set_no', 'game_no', 'point_no', 'target', 'pred_prob', 
                'baseline_prob', 'residual', 'momentum']].copy()
output_df.to_csv('match_flow_results_with_baseline.csv', index=False)
print("\n详细结果已保存到 match_flow_results_with_baseline.csv")