import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# 页面配置
st.set_page_config(
    page_title="PM Risk Prediction Model",
    page_icon="🏥❤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 自定义CSS样式 - 完全按照app.R
st.markdown("""
<style>
    body { 
        background-color: #E6F7FF;
        font-family: 'Segoe UI', Arial, sans-serif; 
        margin: 0;
        padding: 0;
    }
    
    .navbar-custom { 
        background-color: #2c77b4; 
        color: white; 
        padding: 12px 20px; 
        border-radius: 0 0 10px 10px; 
        margin-bottom: 20px; 
        box-shadow: 0 2px 8px rgba(0,0,0,0.1); 
    }
    
    .card { 
        background: white; 
        padding: 15px;
        border-radius: 12px; 
        box-shadow: 0 2px 8px rgb(44 119 180); 
        margin-bottom: 15px;
        text-align: left; 
    }
    
    .section-title { 
        color: #2c77b4; 
        font-weight: bold; 
        border-left: 4px solid #2c77b4; 
        padding-left: 10px; 
        margin-bottom: 12px;
        font-size: 16px;
    }
    
    .risk-container { 
        position: relative; 
        width: 95%;
        height: 16px;
        background: linear-gradient(to right, 
            #5cb85c 0%, #5cb85c 30%, 
            #f0ad4e 30%, #f0ad4e 50%, 
            #d9534f 50%, #d9534f 100%); 
        border-radius: 8px; 
        margin: 12px auto 5px auto;
    }
    
    .risk-indicator {
        position: absolute;
        top: -4px;
        width: 2px;
        height: 24px;
        background-color: #1a1a1a;
        border-radius: 1px;
        transform: translateX(-50%);
        transition: left 0.5s ease-out;
    }

    .scale-wrapper { 
        position: relative; 
        width: 95%;
        height: 18px;
        color: #666; 
        font-size: 12px;
        font-weight: 500; 
        margin-bottom: 12px;
        margin-left: auto;
        margin-right: auto;
    }
    
    .scale-label { 
        position: absolute; 
        transform: translateX(-50%); 
    }
    
    .risk-label-badge { 
        display: inline-block; 
        padding: 4px 16px;
        border-radius: 20px; 
        color: white; 
        font-weight: bold; 
        font-size: 14px;
        margin-top: 5px;
    }
    
    .prob-text-style { 
        color: #2c77b4; 
        font-weight: 600; 
        font-size: 20px;
        margin-bottom: 8px;
    }
    
    footer { 
        text-align: left; 
        color: #333; 
        padding: 15px 0;
        font-size: 12px;
        margin-top: 10px;
    }
    
    .form-group { 
        margin-bottom: 6px !important;
    }
    
    .form-group label { 
        margin-bottom: 3px !important;
        font-size: 13px;
        color: #333;
    }
    
    .form-control { 
        height: 36px !important;
        padding: 5px 10px !important;
        font-size: 13px;
    }
    
    /* SHAP图表样式 */
    .shap-plot-container {
        margin-bottom: 20px;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 10px;
        background-color: #f9f9f9;
    }
    
    .shap-title {
        text-align: center;
        font-weight: bold;
        color: #2c77b4;
        margin-bottom: 10px;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# 标题
st.markdown("""
<div class="navbar-custom">
    <h2 style="margin:0; font-size: 20px;">PM Risk Prediction Model for Colorectal Cancer Patients</h2>
</div>
""", unsafe_allow_html=True)

# 特征定义
FEATURES = [
    {'name': 'AST', 'type': 'numeric', 'default': 20, 'label': 'AST (U/L)'},
    {'name': 'PLT', 'type': 'numeric', 'default': 239, 'label': 'PLT (×10⁹/L)'},
    {'name': 'gender', 'type': 'categorical', 'default': 'Female', 'label': 'Gender',
     'choices': ['Male', 'Female']},
    {'name': 'number.of.metastatic.organs', 'type': 'numeric', 'default': 1,
     'label': 'Number of metastatic organs (n)'},
    {'name': 'other.site.metastasis', 'type': 'numeric', 'default': 0,
     'label': 'Other site metastasis (n)'},
    {'name': 'primary.tumor.sites', 'type': 'categorical', 'default': 'left colon cancer',
     'label': 'Primary tumor site',
     'choices': ['left colon cancer', 'right colon cancer', 'rectal cancer']}
]

# 模拟预测函数
def predict_risk(input_data):
    ast_score = (input_data['AST'] - 50) / 100
    plt_score = (input_data['PLT'] - 250) / 500
    gender_score = 0.1 if input_data['gender'] == 'Male' else 0
    organ_score = input_data['number.of.metastatic.organs'] * 0.15
    site_score = {'left colon cancer': 0.05, 'right colon cancer': 0.1, 'rectal cancer': 0.08}[
        input_data['primary.tumor.sites']
    ]
    other_score = input_data['other.site.metastasis'] * 0.1
    
    prob = 0.3 + ast_score + plt_score + gender_score + organ_score + site_score + other_score
    prob = max(0, min(1, prob))
    
    return round(prob, 3)

# 主容器
st.markdown('<div style="margin: 10px;">', unsafe_allow_html=True)

col_left, col_right = st.columns([4, 8])

# 左侧 - 输入特征
with col_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Input Features</div>', unsafe_allow_html=True)
    
    input_data = {}
    
    for i in range(0, len(FEATURES), 2):
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown('<div class="form-group">', unsafe_allow_html=True)
            f = FEATURES[i]
            if f['type'] == 'numeric':
                input_data[f['name']] = st.number_input(
                    f['label'],
                    value=float(f['default']),
                    step=1.0,
                    format="%.1f",
                    key=f"input_{f['name']}"
                )
            else:
                input_data[f['name']] = st.selectbox(
                    f['label'],
                    f['choices'],
                    index=f['choices'].index(f['default']),
                    key=f"input_{f['name']}"
                )
            st.markdown('</div>', unsafe_allow_html=True)
        
        if i + 1 < len(FEATURES):
            with c2:
                st.markdown('<div class="form-group">', unsafe_allow_html=True)
                f = FEATURES[i + 1]
                if f['type'] == 'numeric':
                    input_data[f['name']] = st.number_input(
                        f['label'],
                        value=float(f['default']),
                        step=1.0,
                        format="%.1f",
                        key=f"input_{f['name']}"
                    )
                else:
                    input_data[f['name']] = st.selectbox(
                        f['label'],
                        f['choices'],
                        index=f['choices'].index(f['default']),
                        key=f"input_{f['name']}"
                    )
                st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div style="margin-top: 5px;">', unsafe_allow_html=True)
    predict_button = st.button("Predict Now", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# 右侧 - 结果
with col_right:
    # 添加外边距容器
    st.markdown('<div style="margin: 10px;">', unsafe_allow_html=True)
    # 全局SHAP分析
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Global SHAP Analysis</div>', unsafe_allow_html=True)
    
    # 条形图
    st.markdown('<div class="shap-plot-container">', unsafe_allow_html=True)
    st.markdown('<div class="shap-title">Global SHAP Importance Plot</div>', unsafe_allow_html=True)
    
    shap_labels = [f['label'] for f in FEATURES]
    shap_values = [0.15, 0.12, 0.10, 0.08, 0.06, 0.04]
    
    fig_bar = go.Figure(go.Bar(
        x=shap_values,
        y=shap_labels,
        orientation='h',
        marker_color='#FFA726'
    ))
    
    fig_bar.update_layout(
        title="Global SHAP Feature Importance",
        xaxis_title="mean(|SHAP value|)",
        yaxis_title="Feature",
        height=300,
        margin=dict(l=150, r=20, t=40, b=40),
        font=dict(size=12, color='black', family='Segoe UI, Arial, sans-serif'),
        title=dict(
            font=dict(size=16, color='#2c77b4')
        ),
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            linecolor='black',
            linewidth=0.5,
            showgrid=True,
            gridcolor='#e0e0e0',
            gridwidth=0.5,
            tickfont=dict(size=12, color='black'),
            title_font=dict(size=14, weight='bold', color='black')
        ),
        yaxis=dict(
            showgrid=False,
            linecolor='black',
            linewidth=0.5,
            tickfont=dict(size=12, color='black'),
            title_font=dict(size=14, weight='bold', color='black')
        )
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 蜂群图
    st.markdown('<div class="shap-plot-container">', unsafe_allow_html=True)
    st.markdown('<div class="shap-title">Global SHAP Beeswarm Plot</div>', unsafe_allow_html=True)
    
    beeswarm_data = []
    for idx, f in enumerate(FEATURES):
        for _ in range(50):
            beeswarm_data.append({
                'Feature': f['label'],
                'SHAP Value': np.random.normal(shap_values[idx], 0.02),
                'Feature Value': np.random.uniform(-3, 3)
            })
    
    beeswarm_df = pd.DataFrame(beeswarm_data)
    
    fig_swarm = px.scatter(
        beeswarm_df,
        x='SHAP Value',
        y='Feature',
        color='Feature Value',
        color_continuous_scale='RdBu',
        range_color=[-3, 3],
        color_continuous_midpoint=0,
        size_max=8,
        opacity=0.8
    )
    
    fig_swarm.update_layout(
        title="Global SHAP Beeswarm Plot",
        xaxis_title="SHAP Value",
        yaxis_title="Feature",
        height=400,
        margin=dict(l=150, r=60, t=40, b=40),
        font=dict(size=12, color='black', family='Segoe UI, Arial, sans-serif'),
        title=dict(
            font=dict(size=16, color='#2c77b4')
        ),
        plot_bgcolor='#f5f5f5',
        paper_bgcolor='#f5f5f5',
        xaxis=dict(
            linecolor='black',
            linewidth=0.5,
            showgrid=True,
            gridcolor='#d0d0d0',
            gridwidth=0.5,
            tickfont=dict(size=12, color='black'),
            title_font=dict(size=14, weight='bold', color='black')
        ),
        yaxis=dict(
            showgrid=False,
            linecolor='black',
            linewidth=0.5,
            tickfont=dict(size=12, color='black'),
            title_font=dict(size=14, weight='bold', color='black')
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="right",
            x=1.02,
            title=dict(
                text="Feature Value",
                font=dict(size=12, weight='bold')
            ),
            font=dict(size=11)
        )
    )
    fig_swarm.update_traces(marker=dict(size=6))
    
    st.plotly_chart(fig_swarm, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 预测结果
    if predict_button or 'last_prob' in st.session_state:
        prob = predict_risk(input_data)
        st.session_state.last_prob = prob
        
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="prob-text-style">
            The probability that this patient has disease is {prob}
        </div>
        """, unsafe_allow_html=True)
        
        if prob <= 0.3:
            pos = (prob / 0.3) * 30
        elif prob <= 0.5:
            pos = 30 + ((prob - 0.3) / (0.5 - 0.3)) * 20
        else:
            pos = 50 + ((prob - 0.5) / (1 - 0.5)) * 50
        
        pos = max(0, min(100, pos))
        
        if prob > 0.5:
            risk_color = "#d9534f"
            risk_label = "High Risk"
        elif prob >= 0.3:
            risk_color = "#f0ad4e"
            risk_label = "Medium Risk"
        else:
            risk_color = "#5cb85c"
            risk_label = "Low Risk"
        
        st.markdown(f"""
        <div class="risk-container">
            <div class="risk-indicator" style="left: {pos}%;"></div>
        </div>
        <div class="scale-wrapper">
            <span class="scale-label" style="left: 0%; transform: none;">0 Low Risk</span>
            <span class="scale-label" style="left: 30%;">0.3</span>
            <span class="scale-label" style="left: 50%;">0.5</span>
            <span class="scale-label" style="right: 0%; transform: none;">High Risk 1</span>
        </div>
        <div class="risk-label-badge" style="background-color: {risk_color};">
            {risk_label}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 个体SHAP分析 - 完全按照R shapviz包的标准
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Individual SHAP Analysis</div>', unsafe_allow_html=True)

        # Waterfall图 - 按照shapviz的sv_waterfall函数样式
        waterfall_data = [
            {'Feature': 'Number of metastatic organs', 'Value': 0.0754},
            {'Feature': 'Gender', 'Value': -0.0276},
            {'Feature': 'PLT', 'Value': 0.0247},
            {'Feature': 'Primary tumor site', 'Value': 0.0227},
            {'Feature': 'AST', 'Value': -0.02},
            {'Feature': 'Other site metastasis', 'Value': -0.00768}
        ]

        fig_waterfall = go.Figure()

        # shapviz的waterfall从底部开始,正值向右,负值向左
        # 从baseline开始累积
        baseline = 0.276
        current_x = baseline

        # 按照shapviz的顺序:从顶部特征到底部特征
        for idx, item in enumerate(reversed(waterfall_data)):
            color = '#FFC107' if item['Value'] > 0 else '#9C27B0'
            
            if item['Value'] >= 0:
                # 正值:从current_x开始向右
                fig_waterfall.add_trace(go.Bar(
                    y=[item['Feature']],
                    x=[item['Value']],
                    orientation='h',
                    marker_color=color,
                    text=[f"{item['Value']:.4f}"],
                    textposition='outside',
                    textfont=dict(size=9, color='black'),
                    showlegend=False,
                    hoverinfo='x+y',
                    base=[current_x]
                ))
                current_x += item['Value']
            else:
                # 负值:从current_x开始向左
                fig_waterfall.add_trace(go.Bar(
                    y=[item['Feature']],
                    x=[abs(item['Value'])],
                    orientation='h',
                    marker_color=color,
                    text=[f"{item['Value']:.4f}"],
                    textposition='outside',
                    textfont=dict(size=9, color='black'),
                    showlegend=False,
                    hoverinfo='x+y',
                    base=[current_x]
                ))
                current_x += item['Value']  # 当前值减小

        # 添加baseline竖线
        fig_waterfall.add_vline(x=baseline, line_dash='dash', line_color='gray', line_width=1)
        
        # 添加baseline标签
        fig_waterfall.add_annotation(
            x=baseline,
            y=0.5,
            text='E[f(x)]',
            showarrow=False,
            font=dict(size=9, color='gray'),
            yshift=10
        )

        fig_waterfall.update_layout(
            title="SHAP Waterfall Plot",
            xaxis_title="SHAP Value",
            yaxis_title="",
            height=310,
            margin=dict(l=180, r=50, t=40, b=40),
            font=dict(size=10, color='black', family='Segoe UI, Arial, sans-serif'),
            title=dict(
                font=dict(size=16, color='#2c77b4')
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False,
            xaxis=dict(
                linecolor='black',
                linewidth=1,
                showgrid=False,
                zeroline=False,
                tickfont=dict(size=12, color='black'),
                title_font=dict(size=14, weight='bold', color='black')
            ),
            yaxis=dict(
                showgrid=False,
                linecolor='black',
                linewidth=1,
                autorange='reversed',
                zeroline=False,
                tickfont=dict(size=10, color='black'),
                title_font=dict(size=10, color='black')
            ),
            hovermode='closest'
        )

        st.plotly_chart(fig_waterfall, use_container_width=True)

        # Force Plot - 按照shapviz的sv_force函数样式
        st.markdown('<div style="padding-left: 150px;">', unsafe_allow_html=True)

        fig_force = go.Figure()

        # Force plot:所有特征贡献从baseline开始累积,显示最终预测值
        baseline = 0.276
        current_x = baseline

        # 添加所有条形,按绝对值降序排列
        sorted_data = sorted(waterfall_data, key=lambda x: abs(x['Value']), reverse=True)

        for item in sorted_data:
            color = '#FFC107' if item['Value'] > 0 else '#9C27B0'
            
            if item['Value'] >= 0:
                fig_force.add_trace(go.Bar(
                    y=[0],
                    x=[item['Value']],
                    orientation='h',
                    marker_color=color,
                    text=[f"{item['Value']:.4f}"],
                    textposition='inside',
                    textfont=dict(size=9, color='black'),
                    showlegend=False,
                    hoverinfo='text',
                    hovertemplate=f"{item['Feature']}<br>{item['Value']:.4f}<extra></extra>",
                    base=[current_x]
                ))
                current_x += item['Value']
            else:
                fig_force.add_trace(go.Bar(
                    y=[0],
                    x=[abs(item['Value'])],
                    orientation='h',
                    marker_color=color,
                    text=[f"{item['Value']:.4f}"],
                    textposition='inside',
                    textfont=dict(size=9, color='black'),
                    showlegend=False,
                    hoverinfo='text',
                    hovertemplate=f"{item['Feature']}<br>{item['Value']:.4f}<extra></extra>",
                    base=[current_x]
                ))
                current_x += item['Value']

        # 添加baseline竖线
        fig_force.add_vline(x=baseline, line_dash='dash', line_color='gray', line_width=1)

        # 添加最终预测值标签
        final_pred = baseline + sum([item['Value'] for item in sorted_data])
        fig_force.add_annotation(
            x=final_pred,
            y=0,
            text=f"{final_pred:.4f}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='black',
            ax=-20,
            ay=0,
            font=dict(size=11, color='black', weight='bold')
        )

        fig_force.update_layout(
            title="Individual SHAP Force Plot",
            xaxis_title="Prediction Value",
            yaxis_title="",
            height=220,
            margin=dict(l=20, r=30, t=40, b=40),
            font=dict(size=10, color='black', family='Segoe UI, Arial, sans-serif'),
            title=dict(
                font=dict(size=16, color='#2c77b4')
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=False,
            barmode='stack',
            xaxis=dict(
                linecolor='black',
                linewidth=1,
                showgrid=False,
                zeroline=False,
                range=[0.2, 0.4],
                tickfont=dict(size=12, color='black'),
                title_font=dict(size=14, weight='bold', color='black')
            ),
            yaxis=dict(
                showgrid=False,
                showticklabels=False,
                linecolor='black',
                linewidth=0.5,
                zeroline=False,
                range=[-0.5, 0.5],
                title_font=dict(size=10, color='black')
            ),
            hovermode='closest'
        )

        st.plotly_chart(fig_force, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# 结束右侧容器
    st.markdown('</div>', unsafe_allow_html=True)

# 页脚
st.markdown("""
<footer>
    web address: https://chgdpnk1.shinyapps.io/gdzjnkzxyy_Model/
</footer>
""", unsafe_allow_html=True)
