try(library(shiny))
try(library(mlr3))
try(library(mlr3learners))
try(library(mlr3pipelines))
try(library(mlr3extralearners))
try(library(ggplot2))
try(library(RWeka))
try(library(lightgbm))
try(library(extraTrees))
try(library(kknn))
try(library(ranger))
#try(library(catboost))
try(library(gbm))
try(library(ada))
try(library(bnlearn))
try(library(xgboost))
try(library(e1071)) 
try(library(shapviz))
try(library(kernelshap))
library(smotefamily)

# 1. 加载模型
try(load("image_ChooseModel.RData"))
try(graph_pipeline_ChooseModel$param_set$set_values(.values = best_ChooseModel_param_vals))
try(graph_pipeline_ChooseModel$train(task_train))

# 2. 获取模型的输入特征信息
task_model <- model_ChooseModel_aftertune$state$train_task
feature_types <- task_model$feature_types
variables <- setNames(as.list(feature_types$type), feature_types$id)

# 获取训练数据
train_data <- as.data.frame(task_train$data())

# 计算每个特征的中位数（连续变量）和水平（分类变量）
default_values <- lapply(names(variables), function(feature) {
  type <- variables[[feature]]
  if (type == "numeric"|type == "integer") {
    median(train_data[[feature]], na.rm = TRUE)
  } else if (type == "factor") {
    levels <- task_model$levels(feature)[[1]]
    levels[1]
  } else if (type == "logical") {
    FALSE
  } else {
    ""
  }
})
names(default_values) <- names(variables)

# 3. 动态生成 Shiny UI
ui <- fluidPage(
  titlePanel("Machine Learning Prediction Model"),
  sidebarLayout(
    sidebarPanel(
      # 动态生成输入组件
      lapply(names(variables), function(feature) {
        type <- variables[[feature]]
        if (type == "numeric"|type == "integer") {
          numericInput(
            inputId = feature, 
            label = feature, 
            value = default_values[[feature]]
          )
        } else if (type == "factor") {
          levels <- task_model$levels(feature)[[1]]
          selectInput(
            inputId = feature, 
            label = feature, 
            choices = levels,
            selected = default_values[[feature]]
          )
        } else if (type == "logical") {
          checkboxInput(
            inputId = feature, 
            label = feature, 
            value = default_values[[feature]]
          )
        } else {
          textInput(
            inputId = feature, 
            label = feature, 
            value = default_values[[feature]]
          )
        }
      }),
      actionButton("predict", "Predict")
    ),
    mainPanel(
      h3("Model prediction results:"),
      verbatimTextOutput("prediction"),

      # 静态全局SHAP图表
      h3("Global SHAP Analysis:"),
      plotOutput("warm",width = "100%",height = "450px"),
      plotOutput("warmm",width = "100%",height = "450px"),
     
  
      HTML("<b><span style='font-size: 32px;'>Individual prediction result</span></b>"), tags$br(),
      htmlOutput("text_output"),tags$br(),tags$br(),
       
      # 动态个体SHAP图表
      h3("Individual SHAP Analysis for Current Input:"),
      plotOutput("individual_waterfall",width = "100%",height = "450px"),
      plotOutput("individual_force",width = "100%",height = "450px")
    )
  )
)

# 4. Shiny Server 逻辑
server <- function(input, output) {
  
  # 渲染全局SHAP图表（只需要渲染一次）
  output$warm <- renderPlot({
    sv_importance(SHAP_sv_ChooseModel) + 
      theme(axis.text = element_text(size = 16), 
            axis.title = element_text(size = 16))
  })
  
  output$warmm <- renderPlot({
    sv_importance(SHAP_sv_ChooseModel, kind = "beeswarm") + 
      theme(axis.text = element_text(size = 16), 
            axis.title = element_text(size = 16))
  })
  
  observeEvent(input$predict, {
    # 收集用户输入
    input_data <- lapply(names(variables), function(feature) {
      input[[feature]]
    })
    names(input_data) <- names(variables)
    
    # 将输入数据转换为 data.frame
    input_df <- as.data.frame(input_data)
    print("Input data:")
    print(input_df)
    
    # 确保数据类型正确
    for (feature in names(variables)) {
      if (variables[[feature]] == "factor") {
        input_df[[feature]] <- factor(input_df[[feature]], levels = task_model$levels(feature)[[1]])
      } else if (variables[[feature]] == "numeric") {
        input_df[[feature]] <- as.numeric(input_df[[feature]])
      } else if (variables[[feature]] == "logical") {
        input_df[[feature]] <- as.logical(input_df[[feature]])
      }
    }
    
    # 进行预测
    prediction <- model_ChooseModel_aftertune$predict_newdata(input_df)
    
    # 显示预测结果
    output$prediction <- renderPrint({
      prediction
    })
    
    output$text_output <- renderUI({
      HTML(paste("<span style='color: DarkCyan; font-size:25px;'>",
                 "The probability that this patient has the disease is :",
                 round(as.numeric(as.data.table(prediction)$prob.1),3), 
                 "</span>"))
    })
    
    # 计算个体SHAP值
    compute_individual_shap(input_df, output)
  })
}

# 独立的SHAP计算函数
compute_individual_shap <- function(input_df, output) {
  # 获取特征名称
  feature_names <- task_model$feature_names
  background_data <- train_data[, feature_names, drop = FALSE]
  input_features <- input_df[, feature_names, drop = FALSE]
   
 
  individual_shap <- NULL
  shap_method_used <- "none"
   
  
  # 使用kernelshap
  if (is.null(individual_shap)) {
    tryCatch({
      print("Trying kernel SHAP method...")
      # 安装kernelshap包如果需要
      if (!requireNamespace("kernelshap", quietly = TRUE)) {
        print("kernelshap package not available")
      } else {
        
        # 定义预测函数
        pred_fun <- function(object, newdata) {
          pred <- object$predict_newdata(newdata)
          # 返回概率或预测值
          if ("prob.1" %in% names(as.data.table(pred))) {
            as.numeric(as.data.table(pred)$prob.1)
          } else {
            as.numeric(as.data.table(pred)$response)
          }
        }
        
        # 使用kernelshap计算
        shap_values <- kernelshap(
          object = model_ChooseModel_aftertune,
          X = input_features,
          bg_X = background_data[sample(nrow(background_data), min(50, nrow(background_data))), ],
          pred_fun = pred_fun
        )
        
        individual_shap <- shapviz(shap_values)
        shap_method_used <- "kernelshap"
        print("Kernel SHAP successful!")
      }
    }, error = function(e) {
      print(paste("Kernel SHAP failed:", e$message))
    })
  }
   
  # 如果成功计算了SHAP值
  if (!is.null(individual_shap)) {
    print(paste("SHAP calculation successful using method:", shap_method_used))
    
    # 渲染个体SHAP waterfall图
    output$individual_waterfall <- renderPlot({
      sv_waterfall(individual_shap, row_id = 1) + 
        theme(axis.text = element_text(size = 14), 
              axis.title = element_text(size = 14)) +
        ggtitle(paste("SHAP Waterfall Plot for Current Input"))
    })
    
    # 渲染个体SHAP force图
    output$individual_force <- renderPlot({
      sv_force(individual_shap, row_id = 1) + 
        theme(axis.text = element_text(size = 14), 
              axis.title = element_text(size = 14)) +
        ggtitle(paste("SHAP Force Plot for Current Input"))
    })
    
  } else {
    # 所有方法都失败了
    print("All SHAP methods failed, showing input visualization")
    show_input_visualization(input_df, output, "No suitable SHAP method found for this model type")
  }
}


# 5. 运行 Shiny App
shinyApp(ui = ui, server = server)
    

