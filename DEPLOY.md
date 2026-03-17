# Streamlit 应用部署指南

## 📁 项目结构

```
app/
├── app.py                    # Streamlit主应用文件
├── app.R                     # 原始R Shiny应用（参考）
├── requirements.txt          # Python依赖包
├── .streamlit/
│   └── config.toml          # Streamlit配置
├── README.md               # 详细说明文档
└── DEPLOY.md              # 本部署指南
```

## 🚀 快速部署步骤

### 1. 本地测试

#### 安装依赖
```bash
cd app
pip install -r requirements.txt
```

#### 运行应用
```bash
streamlit run app.py
```

### 2. 推送到GitHub

```bash
git add app/
git commit -m "Add Streamlit application"
git push origin main
```

### 3. 部署到Streamlit Cloud

1. 访问 [Streamlit Cloud](https://share.streamlit.io)
2. 登录你的GitHub账号
3. 点击 "New app" 按钮
4. 填写以下信息：
   - **Repository**: `pc276868534/gd_Model_test`
   - **Branch**: `main`
   - **Main file path**: `app/app.py`
5. 点击 "Deploy"

## ⚠️ 重要提示

### 关于模型集成

当前版本使用**模拟数据**进行演示。要集成你的实际模型：

#### 选项1：使用scikit-learn模型（推荐）

```python
import joblib

# 加载模型
model = joblib.load('model.pkl')

def predict_risk(input_data):
    df = pd.DataFrame([input_data])
    prediction = model.predict_proba(df)[:, 1]
    return round(float(prediction[0]), 3)
```

#### 选项2：使用R模型（通过rpy2）

```python
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

pandas2ri.activate()

# 调用R模型
```

## 📊 功能特性

- ✅ 完全按照原R Shiny应用的界面设计
- ✅ 4:8的左右布局
- ✅ 全局SHAP分析（条形图和蜂群图）
- ✅ 个体SHAP分析（Waterfall和Force图）
- ✅ 风险预测和可视化
- ✅ 响应式设计

## 🎨 界面说明

- **左侧（4列）**: 输入特征（两列布局）
- **右侧（8列）**:
  - 全局SHAP分析
  - 预测结果
  - 个体SHAP分析

## 📝 更新日志

- v1.0.0 (2024-03-17): 初始版本，完全按照app.R样式设计

## 🆘 常见问题

### Q: 如何查看部署日志？
A: 在Streamlit Cloud的app页面，点击右上角的 "..." 菜单，选择 "Logs"

### Q: 如何更新应用？
A: 修改代码后推送到GitHub，Streamlit Cloud会自动重新部署

### Q: 本地和云端环境不一致？
A: 检查依赖包版本，可以在requirements.txt中固定版本号

---

**部署完成后，你的应用URL将是**: `https://app-name.streamlit.app`
