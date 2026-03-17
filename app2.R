try(library(shiny))
try(library(mlr3))
try(library(mlr3learners))
try(library(mlr3pipelines))
try(library(mlr3extralearners))
try(library(ggplot2))
try(library(shapviz))
try(library(kernelshap))

# --- 1. 配置名称映射与清洗函数 ---
display_names_map <- c(
  "AST" = "AST (U/L)",
  "PLT" = "PLT (×10⁹/L)", 
  "gender" = "Gender",
  "number.of.metastatic.organs" = "Number of metastatic organs (n)",
  "other.site.metastasis" = "Other site metastasis (n)",
  "primary.tumor.sites" = "Primary tumor site"
)

clean_name <- function(x) {
  if (x %in% names(display_names_map)) {
    return(display_names_map[[x]])
  }
  return(gsub("\\.", " ", x))
}

clean_name_for_plot <- function(x) {
  name <- clean_name(x)
  return(gsub("\\s*\\(.*?\\)", "", name)) 
}

gender_choices <- list("Male" = 0, "Female" = 1)
site_choices <- list("left colon cancer" = 1, "right colon cancer" = 2, "rectal cancer" = 3)

# --- 2. 加载模型逻辑 ---
try(load("image_ChooseModel.RData"))
try(graph_pipeline_ChooseModel$param_set$set_values(.values = best_ChooseModel_param_vals))
try(graph_pipeline_ChooseModel$train(task_train))

task_model <- model_ChooseModel_aftertune$state$train_task
variables <- setNames(as.list(task_model$feature_types$type), task_model$feature_types$id)
train_data <- as.data.frame(task_train$data())

# 定义自定义默认值
custom_defaults <- list(
  "AST" = 20,  # AST默认值设为20
  "PLT" = 239,  # PLT默认值设为239
  "gender" = 1,  # 性别默认设为女性 (1)
  "number.of.metastatic.organs" = 1,  # 转移器官数量默认设为1
  "other.site.metastasis" = 0,  # 其他部位转移默认设为0
  "primary.tumor.sites" = 1  # 原发肿瘤部位默认设为左结肠癌
)

# 计算默认值的函数
get_default_value <- function(feature) {
  if (feature %in% names(custom_defaults)) {
    return(custom_defaults[[feature]])
  } else if (variables[[feature]] %in% c("numeric", "integer")) {
    return(median(train_data[[feature]], na.rm = TRUE))
  } else {
    return(task_model$levels(feature)[[1]][1])
  }
}

# 检查是否有全局SHAP对象，如果没有则尝试创建
if (!exists("SHAP_sv_ChooseModel")) {
  tryCatch({
    # 如果SHAP对象不存在，尝试从数据计算
    library(shapviz)
    library(kernelshap)
    
    # 创建预测函数
    pred_fun <- function(object, newdata) {
      pred <- object$predict_newdata(newdata)
      if ("prob.1" %in% names(as.data.table(pred))) {
        return(as.numeric(as.data.table(pred)$prob.1))
      } else {
        return(as.numeric(as.data.table(pred)$response))
      }
    }
    
    # 使用部分数据计算全局SHAP值（减少计算时间）
    feature_names <- task_model$feature_names
    bg_data <- train_data[sample(nrow(train_data), min(100, nrow(train_data))), feature_names, drop = FALSE]
    
    # 计算kernelshap
    shap_values <- kernelshap(
      object = model_ChooseModel_aftertune,
      X = train_data[sample(nrow(train_data), min(500, nrow(train_data))), feature_names, drop = FALSE],
      bg_X = bg_data,
      pred_fun = pred_fun
    )
    
    # 创建shapviz对象
    SHAP_sv_ChooseModel <- shapviz(shap_values)
    
    print("全局SHAP对象已成功计算并创建")
  }, error = function(e) {
    print(paste("无法计算全局SHAP对象:", e$message))
  })
}

# --- 3. UI 界面布局（包含全局SHAP分析）---
ui <- fluidPage(
  tags$head(
    tags$style(HTML("
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
        width: 95%;  /* 改为95% */
        height: 16px;
        background: linear-gradient(to right, 
          #5cb85c 0%, #5cb85c 30%, 
          #f0ad4e 30%, #f0ad4e 50%, 
          #d9534f 50%, #d9534f 100%); 
        border-radius: 8px; 
        margin: 12px auto 5px auto;  /* 增加auto使其居中 */
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
        width: 95%;  /* 改为95% */
        height: 18px;
        color: #666; 
        font-size: 12px;
        font-weight: 500; 
        margin-bottom: 12px;
        margin-left: auto;  /* 增加自动外边距使其居中 */
        margin-right: auto;  /* 增加自动外边距使其居中 */
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
    "))
  ),
  
  # 添加页边距容器
  div(style = "margin: 10px;",  
    div(class = "navbar-custom", 
        h2("PM Risk Prediction Model for Colorectal Cancer Patients", 
           style = "margin:0; font-size: 20px;")
    ),
    
    fluidRow(
      column(width = 4,
             div(class = "card",
                 div(class = "section-title", "Input Features"),
                 fluidRow(
                   lapply(names(variables), function(feature) {
                     # 获取默认值
                     default_val <- get_default_value(feature)
                     
                     # 创建输入控件
                     if (feature == "gender") {
                       input_control <- selectInput(feature, clean_name(feature), 
                                                   choices = gender_choices,
                                                   selected = as.character(default_val))
                     } else if (feature == "primary.tumor.sites") {
                       input_control <- selectInput(feature, clean_name(feature), 
                                                   choices = site_choices,
                                                   selected = as.character(default_val))
                     } else if (variables[[feature]] %in% c("numeric", "integer")) {
                       input_control <- numericInput(feature, clean_name(feature), 
                                                    value = default_val)
                     } else {
                       # 其他分类变量
                       input_control <- selectInput(feature, clean_name(feature), 
                                                   choices = task_model$levels(feature)[[1]],
                                                   selected = as.character(default_val))
                     }
                     
                     # 返回列
                     column(width = 6, input_control)
                   })
                 ),
                 div(style = "margin-top: 5px;",
                     actionButton("predict", "Predict Now", 
                                  class = "btn-primary", 
                                  style = "width:100%; height: 40px; font-size: 16px; border-radius: 8px;")
                 )
             )
      ),
      
      column(width = 8,
             # 全局SHAP分析卡片
             div(class = "card",
                 div(class = "section-title", "Global SHAP Analysis"),
                 div(class = "shap-plot-container",
                     div(class = "shap-title", "Global SHAP Importance Plot"),
                     plotOutput("global_shap_bar", width = "100%", height = "300px")
                 ),
                 div(class = "shap-plot-container",
                     div(class = "shap-title", "Global SHAP Beeswarm Plot"),
                     plotOutput("global_shap_beeswarm", width = "100%", height = "400px")
                 )
             ),
             
             div(class = "card",
                 div(class = "section-title", "Prediction Result"),
                 uiOutput("prob_text"),
                 div(class = "risk-container", uiOutput("dynamic_indicator")),
                 div(class = "scale-wrapper",
                     span(class = "scale-label", style = "left: 0%; transform: none;", "0 Low Risk"),
                     span(class = "scale-label", style = "left: 30%;", "0.3"),
                     span(class = "scale-label", style = "left: 50%;", "0.5"),
                     span(class = "scale-label", style = "right: 0%; transform: none;", "High Risk 1")
                 ),
                 uiOutput("risk_badge")
             ),
             
             div(class = "card",
                 div(class = "section-title", "Individual SHAP Analysis"),
                 plotOutput("waterfall", height = "310px"),
                 div(style = "padding-left: 150px;", plotOutput("force_plot", height = "220px"))
             )
      )
    ),
    

  )
)

# --- 4. Server 逻辑（包含全局SHAP分析）---
server <- function(input, output) {
  
  # 渲染全局SHAP条形图（重要性图）
  output$global_shap_bar <- renderPlot({
    if (exists("SHAP_sv_ChooseModel")) {
      # 创建自定义主题
      custom_theme <- theme_minimal() +
        theme(
          panel.background = element_rect(fill = "white", color = NA),
          panel.grid.major = element_line(color = "#e0e0e0", linewidth = 0.5),
          panel.grid.minor = element_blank(),
          axis.line = element_line(color = "black", linewidth = 0.5),
          axis.text = element_text(size = 12, color = "black"),
          axis.title = element_text(size = 14, face = "bold", color = "black"),
          plot.title = element_text(size = 16, face = "bold", hjust = 0.5, color = "#2c77b4"),
          legend.position = "none"
        )
      
      # 创建条形图
      sv_importance(SHAP_sv_ChooseModel) + 
        custom_theme +
        labs(
          title = "Global SHAP Feature Importance",
          x = "mean(|SHAP value|)",
          y = "Feature"
        ) +
        scale_fill_manual(values = "#FFA726")  # 橙黄色
    } else {
      # 如果没有全局SHAP对象，显示提示
      plot(1, type = "n", xlab = "", ylab = "", xlim = c(0, 1), ylim = c(0, 1),
           main = "Global SHAP Importance Plot Not Available")
      text(0.5, 0.5, "Global SHAP model not loaded or could not be calculated", 
           cex = 1.2, col = "darkred")
    }
  })
  
  # 渲染全局SHAP蜂群图（散点图）
  output$global_shap_beeswarm <- renderPlot({
    if (exists("SHAP_sv_ChooseModel")) {
      # 创建自定义主题
      custom_theme <- theme_minimal() +
        theme(
          panel.background = element_rect(fill = "#f5f5f5", color = NA),  # 灰色背景
          panel.grid.major = element_line(color = "#d0d0d0", linewidth = 0.5),
          panel.grid.minor = element_blank(),
          axis.line = element_line(color = "black", linewidth = 0.5),
          axis.text = element_text(size = 12, color = "black"),
          axis.title = element_text(size = 14, face = "bold", color = "black"),
          plot.title = element_text(size = 16, face = "bold", hjust = 0.5, color = "#2c77b4"),
          legend.position = "right",
          legend.title = element_text(size = 12, face = "bold"),
          legend.text = element_text(size = 11)
        )
      
      # 创建蜂群图
      sv_importance(SHAP_sv_ChooseModel, kind = "beeswarm", show_numbers = FALSE) + 
        custom_theme +
        labs(
          title = "Global SHAP Beeswarm Plot",
          x = "SHAP Value",
          y = "Feature"
        ) +
        scale_color_gradient2(
          low = "#2166ac",  # 蓝色代表低值
          mid = "#f7f7f7",  # 白色代表中间值
          high = "#b2182b", # 红色代表高值
          midpoint = 0,
          name = "Feature Value",
          labels = c("Low", "", "High"),
          breaks = c(-3, 0, 3)  # 根据您的数据调整
        )
    } else {
      # 如果没有全局SHAP对象，显示提示
      plot(1, type = "n", xlab = "", ylab = "", xlim = c(0, 1), ylim = c(0, 1),
           main = "Global SHAP Beeswarm Plot Not Available")
      text(0.5, 0.5, "Global SHAP model not loaded or could not be calculated", 
           cex = 1.2, col = "darkred")
    }
  })
  
  observeEvent(input$predict, {
    # 准备输入数据
    input_list <- lapply(names(variables), function(f) {
      val <- input[[f]]
      if (variables[[f]] %in% c("numeric", "integer")) {
        return(as.numeric(val))
      }
      return(val)
    })
    
    input_df <- as.data.frame(input_list)
    colnames(input_df) <- names(variables)
    
    # 处理因子变量
    for (f in names(variables)) {
      if (variables[[f]] == "factor") {
        input_df[[f]] <- factor(input_df[[f]], levels = task_model$levels(f)[[1]])
      }
    }
    
    # 进行预测
    pred <- model_ChooseModel_aftertune$predict_newdata(input_df)
    prob <- round(as.numeric(as.data.table(pred)$prob.1), 3)
    
    # 更新概率文本
    output$prob_text <- renderUI({
      div(class = "prob-text-style", paste("The probability that this patient has the disease is", prob))
    })
    
    # 更新风险指示器
    output$dynamic_indicator <- renderUI({
      pos <- if (prob <= 0.3) {
        (prob / 0.3) * 30
      } else if (prob <= 0.5) {
        30 + ((prob - 0.3) / (0.5 - 0.3)) * 20
      } else {
        50 + ((prob - 0.5) / (1 - 0.5)) * 50
      }
      tags$div(class = "risk-indicator", style = paste0("left: ", max(0, min(100, pos)), "%;"))
    })
    
    # 更新风险标签
    output$risk_badge <- renderUI({
      res <- if(prob > 0.5) {
        list("#d9534f", "High Risk")
      } else if(prob >= 0.3) {
        list("#f0ad4e", "Medium Risk")
      } else {
        list("#5cb85c", "Low Risk")
      }
      div(span(class = "risk-label-badge", style = paste0("background-color:", res[[1]]), res[[2]]))
    })
    
    # 计算和绘制个体SHAP图
    try({
      pred_fun <- function(obj, newdata) {
        as.numeric(as.data.table(obj$predict_newdata(newdata))$prob.1)
      }
      
      shap_vals <- kernelshap(model_ChooseModel_aftertune, input_df, 
                              bg_X = train_data[sample(nrow(train_data), 50), task_model$feature_names, drop = FALSE], 
                              pred_fun = pred_fun)
      
      colnames(shap_vals$S) <- sapply(colnames(shap_vals$S), clean_name_for_plot)
      colnames(shap_vals$X) <- sapply(colnames(shap_vals$X), clean_name_for_plot)
      
      sv_obj <- shapviz(shap_vals)
      
      theme_clean <- theme_minimal() + 
                     theme(panel.grid = element_blank(), 
                           panel.border = element_blank(),
                           axis.line.x = element_line(color = "black"),
                           axis.line.y = element_blank(),
                           axis.text = element_text(size = 10),
                           axis.title = element_text(size = 10))
      
      output$waterfall <- renderPlot({
        sv_waterfall(sv_obj) + theme_clean + labs(title = "SHAP Waterfall Plot", x = "SHAP Value", y = "")
      })
      
      output$force_plot <- renderPlot({
        sv_force(sv_obj) + theme_clean + 
          theme(axis.line.x = element_line(color = "black"),
                axis.line.y = element_blank(), 
                axis.text.y = element_blank()) +
          labs(title = "Individual SHAP Force Plot", x = "Prediction Value", y = "")
      })
    })
  })
}

shinyApp(ui, server)