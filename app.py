import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import os

# 页面配置
st.set_page_config(
    page_title="PM Risk Prediction Model",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    body {
        background-color: #E6F7FF;
        font-family: 'Segoe UI', Arial, sans-serif;
    }
    .main-header {
        background-color: #2c77b4;
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgb(44 119 180);
        margin-bottom: 20px;
    }
    .section-title {
        color: #2c77b4;
        font-weight: bold;
        border-left: 4px solid #2c77b4;
        padding-left: 10px;
        margin-bottom: 15px;
        font-size: 18px;
    }
    .risk-container {
        position: relative;
        width: 100%;
        height: 16px;
        background: linear-gradient(to right,
            #5cb85c 0%, #5cb85c 30%,
            #f0ad4e 30%, #f0ad4e 50%,
            #d9534f 50%, #d9534f 100%);
        border-radius: 8px;
        margin: 15px 0;
    }
    .risk-label-badge {
        display: inline-block;
        padding: 6px 20px;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        font-size: 14px;
        margin-top: 10px;
    }
    .footer {
        text-align: left;
        color: #333;
        padding: 15px 0;
        font-size: 12px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# 标题
st.markdown("""
<div class="main-header">
    <h2 style="margin:0; font-size: 24px;">PM Risk Prediction Model for Colorectal Cancer Patients</h2>
</div>
""", unsafe_allow_html=True)

# 特征定义
FEATURES = {
    'AST': {'type': 'numeric', 'default': 20, 'label': 'AST (U/L)'},
    'PLT': {'type': 'numeric', 'default': 239, 'label': 'PLT (×10⁹/L)'},
    'gender': {'type': 'categorical', 'default': 'Female', 'label': 'Gender',
               'choices': ['Male', 'Female']},
    'number.of.metastatic.organs': {'type': 'numeric', 'default': 1, 'label': 'Number of metastatic organs (n)'},
    'other.site.metastasis': {'type': 'numeric', 'default': 0, 'label': 'Other site metastasis (n)'},
    'primary.tumor.sites': {'type': 'categorical', 'default': 'left colon cancer', 'label': 'Primary tumor site',
                            'choices': ['left colon cancer', 'right colon cancer', 'rectal cancer']}
}

# 模拟模型预测函数（需要替换为实际模型）
def predict_risk(input_data):
    """
    风险预测函数
    注意：这里使用模拟数据，你需要根据实际情况加载你的模型
    """
    # 这里是模拟预测逻辑，你需要替换为实际的模型预测
    # 例如: model.predict(input_data)
    
    # 模拟计算（基于特征的简单加权）
    ast_score = (input_data['AST'] - 50) / 100
    plt_score = (input_data['PLT'] - 250) / 500
    gender_score = 0.1 if input_data['gender'] == 'Male' else 0
    organ_score = input_data['number.of.metastatic.organs'] * 0.15
    site_score = {'left colon cancer': 0.05, 'right colon cancer': 0.1, 'rectal cancer': 0.08}[input_data['primary.tumor.sites']]
    other_score = input_data['other.site.metastasis'] * 0.1
    
    prob = 0.3 + ast_score + plt_score + gender_score + organ_score + site_score + other_score
    prob = max(0, min(1, prob))  # 限制在0-1之间
    
    return round(prob, 3)

# 侧边栏 - 输入特征
st.sidebar.markdown("### 📊 Input Features")

# 创建输入字段
input_data = {}

for feature_name, feature_info in FEATURES.items():
    if feature_info['type'] == 'numeric':
        input_data[feature_name] = st.sidebar.number_input(
            feature_info['label'],
            value=float(feature_info['default']),
            step=1.0,
            format="%.1f"
        )
    elif feature_info['type'] == 'categorical':
        input_data[feature_name] = st.sidebar.selectbox(
            feature_info['label'],
            feature_info['choices'],
            index=feature_info['choices'].index(feature_info['default'])
        )

# 预测按钮
predict_button = st.sidebar.button("🔮 Predict Now", use_container_width=True, type="primary")

# 主内容区域
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 Input Summary</div>', unsafe_allow_html=True)
    
    # 显示输入值摘要
    for feature_name, feature_info in FEATURES.items():
        st.markdown(f"**{feature_info['label']}**: {input_data[feature_name]}")
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # 全局SHAP分析（模拟）
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Global SHAP Analysis</div>', unsafe_allow_html=True)
    
    # 创建SHAP重要性图表（模拟数据）
    shap_features = list(FEATURES.keys())
    shap_values = [0.15, 0.12, 0.10, 0.08, 0.06, 0.04]  # 模拟数据
    shap_labels = [FEATURES[f]['label'] for f in shap_features]
    
    fig_shap_bar = go.Figure(go.Bar(
        x=shap_values,
        y=shap_labels,
        orientation='h',
        marker_color='#FFA726'
    ))
    
    fig_shap_bar.update_layout(
        title="Global SHAP Feature Importance",
        xaxis_title="mean(|SHAP value|)",
        yaxis_title="Feature",
        height=300,
        margin=dict(l=150, r=20, t=40, b=40)
    )
    
    st.plotly_chart(fig_shap_bar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# 预测结果区域
if predict_button or 'last_prediction' in st.session_state:
    # 执行预测
    prob = predict_risk(input_data)
    st.session_state.last_prediction = prob
    
    col3, col4 = st.columns([1, 2])
    
    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🎯 Prediction Result</div>', unsafe_allow_html=True)
        
        # 概率文本
        st.markdown(f"""
        <div style="color: #2c77b4; font-weight: 600; font-size: 20px; margin-bottom: 10px;">
            The probability that this patient has the disease is {prob}
        </div>
        """, unsafe_allow_html=True)
        
        # 风险指示器
        if prob <= 0.3:
            pos = (prob / 0.3) * 30
            risk_color = "#5cb85c"
            risk_label = "Low Risk"
        elif prob <= 0.5:
            pos = 30 + ((prob - 0.3) / (0.5 - 0.3)) * 20
            risk_color = "#f0ad4e"
            risk_label = "Medium Risk"
        else:
            pos = 50 + ((prob - 0.5) / (1 - 0.5)) * 50
            risk_color = "#d9534f"
            risk_label = "High Risk"
        
        st.markdown(f"""
        <div class="risk-container">
            <div style="position: absolute; top: -4px; width: 2px; height: 24px; 
                        background-color: #1a1a1a; border-radius: 1px; left: {pos}%;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; color: #666; font-size: 12px; margin-top: 5px;">
            <span>0 Low Risk</span>
            <span>0.3</span>
            <span>0.5</span>
            <span>High Risk 1</span>
        </div>
        <div class="risk-label-badge" style="background-color: {risk_color}; margin-top: 15px;">
            {risk_label}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Individual SHAP Analysis</div>', unsafe_allow_html=True)
        
        # 创建个体SHAP瀑布图（模拟）
        shap_individual_values = [-0.05, 0.03, 0.02, 0.08, -0.02, 0.01]
        
        fig_waterfall = go.Figure(go.Bar(
            x=shap_individual_values,
            y=shap_labels,
            orientation='h',
            marker_color=['#b2182b' if x > 0 else '#2166ac' for x in shap_individual_values]
        ))
        
        fig_waterfall.update_layout(
            title="SHAP Waterfall Plot",
            xaxis_title="SHAP Value",
            yaxis_title="",
            height=300,
            margin=dict(l=150, r=20, t=40, b=40)
        )
        
        st.plotly_chart(fig_waterfall, use_container_width=True)
        
        # 创建力导向图（模拟）
        fig_force = go.Figure()
        fig_force.add_trace(go.Bar(
            x=[prob],
            y=['Prediction'],
            orientation='h',
            marker_color='#2c77b4'
        ))
        
        fig_force.update_layout(
            title="Individual SHAP Force Plot",
            xaxis_title="Prediction Value",
            yaxis_title="",
            height=200,
            margin=dict(l=20, r=20, t=40, b=40)
        )
        
        st.plotly_chart(fig_force, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# 页脚
st.markdown("""
<div class="footer">
    <p><strong>Web address:</strong> https://chgdpnk1.shinyapps.io/gdzjnkzxyy_Model/</p>
    <p><strong>GitHub:</strong> https://github.com/pc276868534/gd_Model_test</p>
</div>
""", unsafe_allow_html=True)

# 说明信息
st.markdown("---")
st.markdown("""
### 📌 部署说明

1. **本地测试**:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

2. **部署到Streamlit Cloud**:
   - 确保所有文件已推送到GitHub仓库
   - 访问 https://share.streamlit.io
   - 点击"New app"
   - 选择你的仓库和分支
   - 指定文件路径为 `app/app.py`
   - 点击"Deploy"

3. **重要提示**:
   - 当前版本使用模拟数据进行预测，你需要集成实际的模型
   - 如果你有训练好的模型，请将其保存为`.pkl`或`.joblib`格式
   - 将模型文件放在`app/`目录下
   - 在代码中加载模型并替换`predict_risk`函数中的模拟逻辑
""")

# 模型集成示例代码（注释状态）
"""
# 如果你有实际模型，取消以下注释并修改：

import joblib

# 加载模型
# model = joblib.load('app/your_model.pkl')

def predict_risk(input_data):
    # 将输入数据转换为DataFrame
    df = pd.DataFrame([input_data])
    
    # 进行预测
    prediction = model.predict_proba(df)[:, 1]
    
    return round(float(prediction[0]), 3)
"""
