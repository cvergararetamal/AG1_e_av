#!/usr/bin/env RScript
###############################################################
# Creador: Christian Vergara Retamal
# Usuario VIU: christian.vergararet
# Grado Master en Big Data y Ciencia de Datos
# Asignatura: Estadística Avanzada

# Versión 0.1
# Date: 25/11/2023

#####################Librerías##########################

library(skimr)
library(ggplot2)
library(readr)
library(dplyr)
library(glmnet)
library(pROC)
library(caret)

##### CFG
set.seed(1324)  # Para ejecución de código y mismo resultados

#####################Análisis Exploratorio##########################

data <- read_csv("data_desercion.csv")
# Revsión de los tipos de datos
data_types <- sapply(data, class)
# Revisión de estadísticos básicos
basic_stats <- summary(data)
# Revisión de valores perdidos o nulos
missing_values <- colSums(is.na(data))

str(data)

# Columnas numpericas del dataframe
num_columns <- names(data[sapply(data, is.numeric)])

# Crear histogramas para cada columna numérica
library(ggplot2)
for (col in num_columns) {
  p <- ggplot(data, aes(x = !!sym(col))) +
    geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "blue", color = "black") +
    geom_density(color = "salmon") +
    ggtitle(paste("Distribución de", col)) +
    theme_minimal()
  print(p)
}

num_columns_with_desercion <- c(num_columns, "DESERTA_1ER_ANHO")
correlation_matrix <- cor(data[num_columns_with_desercion], use = "complete.obs")

# Visualizar la matriz de correlación con un mapa de calor
library(corrplot)
corrplot(correlation_matrix, method = "color", type = "upper", order = "hclust", 
         tl.col = "black", tl.srt = 45, addCoef.col = "black", tl.cex = 0.5)



#####################Preprocesamiento##########################

# Seleccion de variables
data_refactorizada <- select(data, "FACULTAD", "GRUPO_ANHO_EGRESO", "UNIDAD_EDUCATIVA", "DEPENDENCIA",
                             "TIPO_EDUCACION", "POSICION_LISTA_POST", "GRUPO_ED_PADRES", "SEXO",
                             "1RA_PREFERENCIA", "PTJE_LENGUAJE", "PTJE_MATEMATICAS", "PTJE_NEM",
                             "GRATUIDAD", "TRABAJA_Y_ESTUDIA", "HIJOS", "DISCAPACIDAD", "SITUACION_DE_POSTULACION",
                             "CERCANIA", "AMBAS_PRUEBAS", "EDAD_EGRESO_COLEGIO", "EDAD_INGRESO_UNIV",
                             "DESERTA_1ER_ANHO")

head(data_refactorizada)

# Cambio nombre de variable por potencial error en la aplicación de algoritmos (parte numérica)
names(data_refactorizada)[names(data_refactorizada) == "1RA_PREFERENCIA"] <- "PRIMERA_PREFRENCIA"

# transformacion variables categórica ('OHE')
categorical_vars <- c("FACULTAD", "GRUPO_ANHO_EGRESO", "UNIDAD_EDUCATIVA", "DEPENDENCIA",
                      "TIPO_EDUCACION", "GRUPO_ED_PADRES", "SEXO", "PRIMERA_PREFRENCIA", "GRATUIDAD",
                      "TRABAJA_Y_ESTUDIA", "HIJOS", "DISCAPACIDAD", "SITUACION_DE_POSTULACION",
                      "CERCANIA", "AMBAS_PRUEBAS")
data_refactorizada[categorical_vars] <- lapply(data_refactorizada[categorical_vars], factor)

num_vars <- c("POSICION_LISTA_POST", "PTJE_LENGUAJE", "PTJE_MATEMATICAS", "PTJE_NEM", "EDAD_EGRESO_COLEGIO",
              "EDAD_INGRESO_UNIV")
data_refactorizada[num_vars] <- scale(data_refactorizada[num_vars])

# OHE
data_refactorizada_ohe <- model.matrix(~ . - 1, data_refactorizada)

# Convertir la matriz a un dataframe
data_refactorizada_ohe_df <- as.data.frame(data_refactorizada_ohe)
colnames(data_refactorizada_ohe_df) <- colnames(data_refactorizada_ohe)

head(data_refactorizada_ohe)

head(data_refactorizada_ohe_df)


######### Ridge y Lasso #########

# Separar las variables independientes (X) y la variable dependiente (y)
X <-as.matrix(data_refactorizada_ohe_df[, -which(names(data_refactorizada_ohe_df) == "DESERTA_1ER_ANHO")])
y <-data_refactorizada_ohe_df$DESERTA_1ER_ANHO

# Dividir los datos en conjunto de entrenamiento y prueba
indices <- sample(1:nrow(X), size=floor(0.67*nrow(X))) # Dicidimos los datos en 0.67 train y 0.33 test
X_train<-X[indices, ]
X_test <- X[-indices, ]

y_train <-y[indices]
y_test <-y[-indices]

# Aplicación Modelo Ridge
modelo_ridge <- glmnet(X_train, y_train,alpha=0)
predicciones_ridge<- predict(modelo_ridge, newx = X_test, s = 1)
mse_ridge <- mean((y_test - predicciones_ridge)^2)

# Aplicación Modelo Lasso
modelo_lasso <- glmnet(X_train, y_train, alpha = 1)
predicciones_lasso<- predict(modelo_lasso, newx = X_test, s = 0.1)
mse_lasso <- mean((y_test - predicciones_lasso)^2)

mse_ridge
mse_lasso


##### Ajuste de variables dado los resultados (coeficientes obtenidos) ######
##### LASSO #####

X_lasso <- as.matrix(data_refactorizada_ohe_df[, -which(names(data_refactorizada_ohe_df) == "DESERTA_1ER_ANHO")])
y_lasso <- data_refactorizada_ohe_df$DESERTA_1ER_ANHO

set.seed(1324) 
indices_lasso <- sample(1:nrow(X_lasso), size=floor(0.67*nrow(X_lasso)))
X_train_lasso <-X_lasso[indices_lasso, ]
X_test_lasso <-X_lasso[-indices_lasso, ]
y_train_lasso <-y_lasso[indices_lasso]
y_test_lasso <-y_lasso[-indices_lasso]

# Ajuste del modelo Lasso
lasso_model <- glmnet(X_train_lasso, y_train_lasso, alpha = 1,lambda = 0.01) # Alpha 1 dictamina lasso

lasso_coefficients <- predict(lasso_model, type = "coefficients", s = 0.01)[,1]

# Filtrar variables significativas (coeficientes distintos de 0)
significant_variables_lasso <- lasso_coefficients[lasso_coefficients != 0]
print(significant_variables_lasso)

########################################################
##### Ajuste de variables dado los resultados (coeficientes obtenidos) ######
##### Ridge #####

X_ridge<- as.matrix(data_refactorizada_ohe_df[, -which(names(data_refactorizada_ohe_df) == "DESERTA_1ER_ANHO")])
y_ridge<- data_refactorizada_ohe_df$DESERTA_1ER_ANHO
set.seed(1324)
indices_ridge<- sample(1:nrow(X_ridge), size=floor(0.67*nrow(X_ridge)))
X_train_ridge<- X_ridge[indices_ridge, ]
X_test_ridge<- X_ridge[-indices_ridge, ]
y_train_ridge<- y_ridge[indices_ridge]
y_test_ridge<- y_ridge[-indices_ridge]

# Ajuste del modelo Ridge
ridge_model <- glmnet(X_train_ridge, y_train_ridge, alpha = 0) #0 risge

#coeficientes del modelo
ridge_coeficientes <- predict(ridge_model, type="coefficients", s = 0.01)[,1]
print(ridge_coeficientes)



################## Modelamiento Predictivo ########################


### Regresión logística estándar 
set.seed(1324)

X <-as.matrix(data_refactorizada_ohe_df[, -which(names(data_refactorizada_ohe_df) == "DESERTA_1ER_ANHO")])
y <-data_refactorizada_ohe_df$DESERTA_1ER_ANHO

# Dividir los datos en conjunto de entrenamiento y prueba
indices <- sample(1:nrow(X), size=floor(0.67*nrow(X))) # Dicidimos los datos en 0.67 train y 0.33 test
X_train<-X[indices, ]
X_test <- X[-indices, ]

y_train <-y[indices]
y_test <-y[-indices]
set.seed(1324)
data_train <- data.frame(X_train, DESERTA_1ER_ANHO = y_train)
data_test <- data.frame(X_test, DESERTA_1ER_ANHO = y_test)
modelo_logit<- glm(DESERTA_1ER_ANHO ~ ., data = data_train)

summary(modelo_logit)

### Regresión logística variables lasso
set.seed(1324)
data_train_lasso <- data.frame(X_train_lasso, DESERTA_1ER_ANHO = y_train_lasso)
data_test_lasso <- data.frame(X_test_lasso, DESERTA_1ER_ANHO = y_test_lasso)

#Variables por Lasso
variables_lasso <- c("FACULTADFACULTAD.DE.ARQUITECTURA..MUSICA.Y.DISENO", 
                     "FACULTADFACULTAD.DE.ECONOMIA.Y.NEGOCIOS", 
                     "FACULTADFACULTAD.DE.INGENIERIA", 
                     "GRUPO_ANHO_EGRESOULTIMO.ANO", 
                     "UNIDAD_EDUCATIVA2", 
                     "TIPO_EDUCACIONTECNICO.PROFESIONAL", 
                     "PRIMERA_PREFRENCIA1", 
                     "PTJE_LENGUAJE", 
                     "PTJE_MATEMATICAS", 
                     "PTJE_NEM", 
                     "GRATUIDADSI", 
                     "CERCANIALEJOS", 
                     "EDAD_EGRESO_COLEGIO")

formula_lasso <- as.formula(paste("DESERTA_1ER_ANHO ~", paste(variables_lasso, collapse = " + ")))
modelo_logit_lasso <- glm(formula_lasso, data = data_train_lasso)
summary(modelo_logit_lasso)



### Comparativas

# Función summarize para evaluar los modelos

calcualr_metricas_model <- function(modelo, X_test, y_test) {
  X_test_aux <- X_test[, !(names(X_test) %in% c("DESERTA_1ER_ANHO"))]
  predicciones_prob <- predict(modelo, newdata = X_test_aux, type = "response")
  predicciones <- ifelse(predicciones_prob > 0.5, 1, 0)
  matriz_confusion <- table(Predicted = factor(predicciones, levels=c("0", "1")), Actual = factor(y_test, levels=c("0", "1")))
  accuracy <- sum(diag(matriz_confusion)) / sum(matriz_confusion)  # Precisión (Accuracy)
  precision <- precision(matriz_confusion)
  recall <- sensitivity(matriz_confusion)
  f1_score <-(2 * precision * recall) / (precision + recall)
  auc_score <-pROC::auc(pROC::roc(y_test, predicciones_prob))
  falsos_positivos <-matriz_confusion["1", "0"]

  # Return
  list(Accuracy = accuracy,
       precision = precision, 
       recall = recall, 
       F1_score = f1_score, 
       AUC_score = auc_score, 
       falsos_positivos = falsos_positivos)}



#Aplicar a vanila
metricas_logitvanila <- calcualr_metricas_model(modelo_logit, data_test, y_test)

print(metricas_logitvanila)

#Aplicar a lasso
X_test_lasso <- data_test_lasso[variables_lasso]
y_test_lasso <- data_test_lasso$DESERTA_1ER_ANHO
metricas_lasso <- calcualr_metricas_model(modelo_logit_lasso, X_test_lasso, y_test_lasso)
print(metricas_lasso)


#Aplicar a ridge
predicciones_prob_ridge <- predict(ridge_model, newx= X_test_ridge,s= 0.01)
predicciones_ridge <- ifelse(predicciones_prob_ridge > 0.5,1,0)
confusion_matriz_ridge <- confusionMatrix(factor(predicciones_ridge, levels = c("0", "1")), 
                                          factor(y_test_ridge, levels = c("0", "1")))
recall_ridge<- confusion_matriz_ridge$byClass["Sensitivity"]
precision_ridge<-confusion_matriz_ridge$byClass["Positive Predictive Value"]
f1_score_ridge<- 2*(precision_ridge * recall_ridge) / (precision_ridge + recall_ridge)
roc_curve_ridge <- roc(y_test_ridge, predicciones_prob_ridge)
auc_ridge <- auc(roc_curve_ridge)
falsos_positivos_ridge <- confusion_matriz_ridge$table["0", "1"]
# Homologo calcular metrics
list(
  Accuracy = confusion_matriz_ridge$overall["Accuracy"],
  Recall = recall_ridge,
  Precision = precision_ridge,
  F1_Score = f1_score_ridge,
  AUC = auc_ridge,
  falsos_positivos = falsos_positivos_ridge
)










