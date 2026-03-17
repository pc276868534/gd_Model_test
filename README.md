# PM Risk Prediction Model - Streamlit Version

这是一个结直肠癌患者PM风险预测模型的Streamlit应用，从R Shiny应用转换而来。

## 📁 项目结构

```
app/
├── app.py                    # Streamlit主应用文件
├── requirements.txt          # Python依赖包列表
├── .streamlit/
│   └── config.toml          # Streamlit配置文件
├── image_ChooseModel.RData  # R模型数据文件（原始）
└── README.md                # 本说明文件
```

## 🚀 快速开始

### 1. 本地测试

#### 安装依赖
```bash
pip install -r requirements.txt
```

#### 运行应用
```bash
streamlit run app.py
```

应用将在浏览器中自动打开，默认地址是 `http://localhost:8501`

### 2. 部署到Streamlit Cloud

#### 步骤一：推送到GitHub
确保你已经将所有文件推送到你的GitHub仓库：
```bash
git add .
git commit -m "Add Streamlit application"
git push origin main
```

#### 步骤二：部署到Streamlit Cloud
1. 访问 [Streamlit Cloud](https://share.streamlit.io)
2. 登录你的GitHub账号
3. 点击 "New app" 按钮
4. 填写以下信息：
   - **Repository**: 选择 `pc276868534/gd_Model_test`
   - **Branch**: 选择 `main`（或其他默认分支）
   - **Main file path**: 输入 `app/app.py`
5. 点击 "Deploy" 按钮

Streamlit Cloud会自动：
- 从GitHub仓库拉取代码
- 安装 `requirements.txt` 中的依赖
- 启动Streamlit应用

#### 步骤三：等待部署完成
部署通常需要1-3分钟，完成后你会看到应用URL，格式类似：
```
https://app-name.streamlit.app
```

## 📊 功能特性

- ✅ 患者风险预测
- ✅ 实时交互界面
- ✅ 风险等级可视化（低/中/高风险）
- ✅ 全局SHAP特征重要性分析
- ✅ 个体SHAP解释
- ✅ 响应式设计
- ✅ 美观的用户界面

## 🔧 配置说明

### 输入特征
应用支持以下输入特征：

| 特征名称 | 类型 | 默认值 | 说明 |
|---------|------|--------|------|
| AST | 数值型 | 20 U/L | 谷草转氨酶 |
| PLT | 数值型 | 239 ×10⁹/L | 血小板计数 |
| Gender | 分类 | Female | 性别（Male/Female） |
| Number of metastatic organs | 数值型 | 1 | 转移器官数量 |
| Other site metastasis | 数值型 | 0 | 其他部位转移数量 |
| Primary tumor site | 分类 | left colon cancer | 原发肿瘤部位 |

### 风险等级划分
- **低风险**: 概率 ≤ 0.3（绿色）
- **中风险**: 0.3 < 概率 ≤ 0.5（黄色）
- **高风险**: 概率 > 0.5（红色）

## ⚠️ 重要提示

### 关于模型集成

当前版本使用**模拟数据**进行演示，你需要根据实际情况集成你的机器学习模型：

#### 选项1：使用scikit-learn模型
```python
from sklearn.ensemble import RandomForestClassifier
import joblib

# 训练你的模型（示例）
# model = RandomForestClassifier()
# model.fit(X_train, y_train)

# 保存模型
# joblib.dump(model, 'app/model.pkl')

# 在app.py中加载
model = joblib.load('app/model.pkl')

def predict_risk(input_data):
    df = pd.DataFrame([input_data])
    prediction = model.predict_proba(df)[:, 1]
    return round(float(prediction[0]), 3)
```

#### 选项2：使用R模型（通过rpy2）
如果必须使用R模型，可以安装rpy2库：
```bash
pip install rpy2
```

然后在Python中调用R模型：
```python
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

pandas2ri.activate()

# 加载R模型
r_script = """
library(mlr3)
load("app/image_ChooseModel.RData")
"""

ro.r(r_script)

def predict_risk(input_data):
    # 转换数据并调用R模型
    # ... 具体实现
    pass
```

#### 选项3：重新训练Python模型
推荐使用Python重新训练模型，性能和兼容性更好：
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd

# 加载训练数据
data = pd.read_csv('training_data.csv')
X = data.drop('target', axis=1)
y = data['target']

# 训练模型
model = GradientBoostingClassifier()
model.fit(X, y)

# 保存模型
joblib.dump(model, 'app/model.pkl')
```

## 📦 依赖包

- `streamlit` - Web应用框架
- `pandas` - 数据处理
- `numpy` - 数值计算
- `plotly` - 交互式图表
- `scikit-learn` - 机器学习（用于模型集成）
- `joblib` - 模型序列化

## 🎨 自定义主题

主题配置在 `.streamlit/config.toml` 文件中，你可以修改：
- `primaryColor` - 主色调
- `backgroundColor` - 背景色
- `textColor` - 文字颜色
- 等等...

## 📝 更新日志

### v1.0.0 (2024-03-17)
- 从R Shiny应用转换到Streamlit
- 保留原有UI设计和功能
- 添加交互式图表
- 实现风险预测可视化
- 支持本地测试和云端部署

## 🆘 常见问题

### Q1: 部署后应用无法启动？
A: 检查以下几点：
1. `requirements.txt` 是否正确
2. `app.py` 文件路径是否正确（应该是 `app/app.py`）
3. 查看Streamlit Cloud的日志获取错误信息

### Q2: 如何查看部署日志？
A: 在Streamlit Cloud的app页面，点击右上角的 "..." 菜单，选择 "Logs"

### Q3: 如何更新应用？
A:
1. 修改代码后推送到GitHub
2. Streamlit Cloud会自动检测并重新部署

### Q4: 本地和云端环境不一致？
A: 确保依赖包版本一致，可以在 `requirements.txt` 中固定版本号：
```
streamlit==1.28.0
pandas==2.0.3
```

## 📞 联系方式

如有问题，请通过以下方式联系：
- GitHub Issues: https://github.com/pc276868534/gd_Model_test/issues
- Email: [your-email@example.com]

## 📄 许可证

[你的许可证信息]

---

**注意**: 请将此README中的示例代码和模型路径根据你的实际情况进行修改。
