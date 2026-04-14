try(library(shiny))
try(library(mlr3))
try(library(mlr3learners))
try(library(mlr3pipelines))
try(library(mlr3extralearners))
try(library(ggplot2))
try(library(shapviz))
try(library(kernelshap))

# =========================================================
# 0) Load model objects (keep your original logic)
# =========================================================
try(load("image_ChooseModel.RData"))
try(graph_pipeline_ChooseModel$param_set$set_values(.values = best_ChooseModel_param_vals))
try(graph_pipeline_ChooseModel$train(task_train))

task_model <- model_ChooseModel_aftertune$state$train_task
variables <- setNames(as.list(task_model$feature_types$type), task_model$feature_types$id)
train_data <- as.data.frame(task_train$data())

# =========================================================
# 1) Feature display names (original -> UI display)
#    original features: age, bmi, gender, glucose, race, sbp
# =========================================================
display_names_map <- c(
  "age" = "Age, year",
  "bmi" = "BMI, kg/m^2",
  "gender" = "Gender",
  "glucose" = "FPG, mmol/L",
  "race" = "Race",
  "sbp" = "SBP, mmHg"
)

clean_name <- function(x) {
  if (x %in% names(display_names_map)) return(display_names_map[[x]])
  return(gsub("\\.", " ", x))
}

# For SHAP plot title cleanup (remove parenthesis in display name if any)
clean_name_for_plot <- function(x) {
  name <- clean_name(x)
  return(gsub("\\s*\\(.*?\\)", "", name))
}

# =========================================================
# 2) UI select options with display mapping
#    but pass original numeric codes to the model
# =========================================================

# Sex: original 1=Male, 2=Female
gender_display_choices <- c(
  "Male" = 1,
  "Female" = 2
)

# Race: original 1=Han, 2=Yi, 3=Meng, 4=Dai, 5=TuJia
race_display_choices <- c(
  "Han" = 1,
  "Yi" = 2,
  "Meng" = 3,
  "Dai" = 4,
  "TuJia" = 5
)

# =========================================================
# 3) Default values (custom defaults)
#    (If you want other defaults, modify here)
# =========================================================
custom_defaults <- list(
  "age" = 50,
  "bmi" = 28,
  "gender" = 1,   # 1=Male
  "glucose" = 7.0,
  "race" = 1,     # 1=Han
  "sbp" = 120
)

get_default_value <- function(feature) {
  if (feature %in% names(custom_defaults)) {
    return(custom_defaults[[feature]])
  } else if (feature %in% names(train_data) && is.numeric(train_data[[feature]])) {
    return(median(train_data[[feature]], na.rm = TRUE))
  } else {
    # fallback
    if (!is.null(task_model$feature_names) && feature %in% task_model$feature_names) {
      levs <- task_model$levels(feature)
      if (!is.null(levs) && length(levs[[1]]) > 0) return(levs[[1]][1])
    }
    return(NULL)
  }
}

# =========================================================
# 4) UI layout requirements
#    - Input split into 3 columns, each 3 inputs
#    - Remove duplicate prediction parts
#    - Remove Global SHAP
#    - Background淡蓝色
#    - Font bigger
#    - Prediction panel model name in top-left
#    - Risk colors: >0.5 red (High), 0.3-0.5 yellow (Medium), <0.3 green (Low)
# =========================================================

# Determine features order exactly as requested
feature_order <- c("age", "bmi", "gender", "glucose", "race", "sbp")

ui <- fluidPage(
  tags$head(
    tags$style(HTML("
      body {
        background-color: #E6F7FF; /*淡蓝色*/
        font-family: 'Segoe UI', Arial, sans-serif;
        margin: 0;
        padding: 0;
      }

      .navbar-custom {
        background-color: #2c77b4;
        color: white;
        padding: 14px 20px;
        border-radius: 0 0 12px 12px;
        margin-bottom: 18px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      }

      .card {
        background: white;
        padding: 18px;
        border-radius: 14px;
        box-shadow: 0 2px 10px rgba(44,119,180,0.25);
        margin-bottom: 16px;
      }

      .section-title {
        color: #2c77b4;
        font-weight: 800;
        border-left: 4px solid #2c77b4;
        padding-left: 10px;
        margin-bottom: 14px;
        font-size: 18px;
      }

      .form-control {
        height: 42px !important;
        padding: 8px 10px !important;
        font-size: 15px !important;
      }

      .form-group label {
        margin-bottom: 6px !important;
        font-size: 14px;
        color: #333;
        font-weight: 700;
      }

      .btn-primary {
        font-size: 16px;
        font-weight: 700;
        border-radius: 10px;
        height: 44px !important;
      }

      /* Probability text */
      .prob-text-style {
        color: #2c77b4;
        font-weight: 800;
        font-size: 22px;
        margin-bottom: 10px;
      }

      /* Risk indicator bar (single, with thresholds) */
      .risk-container {
        position: relative;
        width: 100%;
        height: 16px;
        border-radius: 10px;
        margin-top: 10px;
        overflow: hidden;
        background: linear-gradient(to right,
          #5cb85c 0%, #5cb85c 30%,
          #f0ad4e 30%, #f0ad4e 50%,
          #d9534f 50%, #d9534f 100%);
      }

      .risk-indicator {
        position: absolute;
        top: -6px;
        width: 3px;
        height: 28px;
        background-color: #1a1a1a;
        border-radius: 2px;
        transform: translateX(-50%);
        transition: left 0.5s ease-out;
      }

      .risk-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 999px;
        color: white;
        font-weight: 900;
        font-size: 14px;
        margin-top: 12px;
      }

      .scale-wrapper {
        position: relative;
        width: 100%;
        margin-top: 8px;
        margin-bottom: 6px;
        color: #666;
        font-size: 12px;
        font-weight: 700;
      }
      .scale-label {
        position: absolute;
        transform: translateX(-50%);
      }

      footer {
        text-align: left;
        color: #333;
        padding: 14px 0 8px 0;
        font-size: 12px;
        margin-top: 8px;
        font-weight: 600;
      }
    "))
  ),

  # Header
  div(style = "margin: 10px;",
      div(class = "navbar-custom",
          h2("Lung Prediction Model for Diabetic Patients",
             style = "margin:0; font-size: 20px; font-weight: 900;")
      ),

      fluidRow(
        # Left column (inputs)
        column(width = 4,
               div(class = "card",
                   div(class = "section-title", "Input Features"),

                   # Inputs split into 3 columns (each column has 2-1? -> requirement: 3 columns each 3 inputs)
                   # 6 features total，所以实际无法做到“每列3个输入框”（那会是9个）。
                   # 我按最合理方式：3列布局，分别放 2个/2个/2个，使整体对齐且满足“分成3列”。
                   # 如果你必须严格“每列3个”，你需要再提供3个额外特征名。
                   fluidRow(
                     column(width = 4,
                            style = "padding-right:10px;",
                            div(class="form-group",
                                label("age", display_names_map["age"], class="control-label"),
                                numericInput("age", NULL, value = get_default_value("age"))
                            ),
                            div(class="form-group",
                                label("gender", display_names_map["gender"], class="control-label"),
                                selectInput("gender", NULL,
                                             choices = gender_display_choices,
                                             selected = as.character(get_default_value("gender")))
                            )
                     ),
                     column(width = 4,
                            style = "padding-left:10px; padding-right:10px;",
                            div(class="form-group",
                                label("bmi", display_names_map["bmi"], class="control-label"),
                                numericInput("bmi", NULL, value = get_default_value("bmi"))
                            ),
                            div(class="form-group",
                                label("glucose", display_names_map["glucose"], class="control-label"),
                                numericInput("glucose", NULL, value = get_default_value("glucose"))
                            )
                     ),
                     column(width = 4,
                            style = "padding-left:10px;",
                            div(class="form-group",
                                label("race", display_names_map["race"], class="control-label"),
                                selectInput("race", NULL,
                                             choices = race_display_choices,
                                             selected = as.character(get_default_value("race")))
                            ),
                            div(class="form-group",
                                label("sbp", display_names_map["sbp"], class="control-label"),
                                numericInput("sbp", NULL, value = get_default_value("sbp"))
                            )
                     )
                   ),

                   div(style = "margin-top: 14px;",
                       actionButton("predict", "Predict Now",
                                    class = "btn-primary",
                                    style = "width:100%; height: 44px;")
                   )
               )
        ),

        # Right column (results)
        column(width = 8,
               div(class = "card",
                   div(class = "section-title", "Prediction Result"),

                   uiOutput("prob_text"),

                   div(class = "risk-container",
                       uiOutput("dynamic_indicator")  # a vertical marker
                   ),

                   # risk badge (red/yellow/green)
                   uiOutput("risk_badge"),

                   # Optional scale labels (English)
                   div(class="scale-wrapper",
                       span(class="scale-label", style="left: 0%; transform: none;", "0 Low Risk"),
                       span(class="scale-label", style="left: 30%;", "0.3"),
                       span(class="scale-label", style="left: 50%;", "0.5"),
                       span(class="scale-label", style="right: 0%; transform: none;", "High Risk 1")
                   )
               ),

               div(class = "card",
                   div(class = "section-title", "Individual SHAP Analysis"),
                   plotOutput("waterfall", height = "310px"),
                   div(style = "padding-left: 80px;",
                       plotOutput("force_plot", height = "220px")
                   )
               ),

               # footer at bottom
               footer("Power by FreeStatistics, Contact email: freestatistics@163.com")
        )
      )
  )
)

# =========================================================
# 5) Server logic
#    - gender/race: UI shows labels, but values sent to model are original codes
#    - risk threshold & English labels
# =========================================================
server <- function(input, output) {

  observeEvent(input$predict, {

    # Build input df with original feature IDs (age,bmi,gender,glucose,race,sbp)
    input_df <- data.frame(
      age = as.numeric(input$age),
      bmi = as.numeric(input$bmi),
      gender = as.numeric(input$gender),   # 1 Male, 2 Female (original codes preserved)
      glucose = as.numeric(input$glucose),
      race = as.numeric(input$race),       # 1..5 original codes preserved
      sbp = as.numeric(input$sbp)
    )

    # Ensure factor if needed by model (your original code supported factor conversion)
    # Here we try to align with task_model feature types.
    for (f in names(input_df)) {
      ft <- variables[[f]]
      if (!is.null(ft) && ft == "factor") {
        input_df[[f]] <- factor(input_df[[f]], levels = task_model$levels(f)[[1]])
      }
    }

    # Predict probability
    pred <- model_ChooseModel_aftertune$predict_newdata(input_df)
    prob <- round(as.numeric(as.data.table(pred)$prob.1), 3)

    output$prob_text <- renderUI({
      div(class = "prob-text-style",
          paste0("The probability that this patient has the disease is ", prob)
      )
    })

    # Risk marker position on bar:
    # map prob -> percent using thresholds:
    # 0.0~0.3 => 0~30, 0.3~0.5 => 30~50, 0.5~1.0 => 50~100
    output$dynamic_indicator <- renderUI({
      p <- prob
      pos <- if (p <= 0.3) {
        (p / 0.3) * 30
      } else if (p <= 0.5) {
        30 + ((p - 0.3) / (0.5 - 0.3)) * 20
      } else {
        50 + ((p - 0.5) / (1 - 0.5)) * 50
      }
      tags$div(
        class = "risk-indicator",
        style = paste0("left:", max(0, min(100, pos)), "%;")
      )
    })

    # Risk badge colors + labels (English)
    output$risk_badge <- renderUI({
      if (prob > 0.5) {
        div(class="risk-badge", style = "background-color:#d9534f;", "High Risk ( > 0.5 )")
      } else if (prob >= 0.3) {
        div(class="risk-badge", style = "background-color:#f0ad4e; color:#333;", "Medium Risk ( 0.3 - 0.5 )")
      } else {
        div(class="risk-badge", style = "background-color:#5cb85c;", "Low Risk ( < 0.3 )")
      }
    })

    # Individual SHAP (kernelshap)
    # background samples: we take random rows for bg_X
    try({
      pred_fun <- function(obj, newdata) {
        as.numeric(as.data.table(obj$predict_newdata(newdata))$prob.1)
      }

      # model expects feature names in task_model$feature_names order
      bg_X <- train_data[sample(nrow(train_data), min(50, nrow(train_data))), task_model$feature_names, drop = FALSE]

      shap_vals <- kernelshap(
        model_ChooseModel_aftertune,
        input_df,
        bg_X = bg_X,
        task_model$feature_names,
        drop = FALSE,
        pred_fun = pred_fun
      )

      colnames(shap_vals$S) <- sapply(colnames(shap_vals$S), clean_name_for_plot)
      colnames(shap_vals$X) <- sapply(colnames(shap_vals$X), clean_name_for_plot)

      sv_obj <- shapviz(shap_vals)

      theme_clean <- theme_minimal() +
        theme(
          panel.grid = element_blank(),
          panel.border = element_blank(),
          axis.line.x = element_line(color = "black"),
          axis.line.y = element_blank(),
          axis.text = element_text(size = 11, face = "bold"),
          axis.title = element_text(size = 12)
        )

      output$waterfall <- renderPlot({
        sv_waterfall(sv_obj) + theme_clean +
          labs(title = "SHAP Waterfall Plot", x = "SHAP Value", y = "")
      })

      output$force_plot <- renderPlot({
        sv_force(sv_obj) + theme_clean +
          theme(
            axis.line.x = element_line(color = "black"),
            axis.line.y = element_blank(),
            axis.text.y = element_blank()
          ) +
          labs(title = "Individual SHAP Force Plot", x = "Prediction Value", y = "")
      })
    }, silent = TRUE)
  })
}

shinyApp(ui, server)
