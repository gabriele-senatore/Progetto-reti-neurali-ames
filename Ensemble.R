library(modeldata)
library(tidyverse)
library(tidymodels)
library(torch)
library(brulee)
library(future)
library(tictoc)
library(corrplot)
library(ggplot2)
library(furrr)

tidymodels_prefer()

dat <- ames

#NOTE SPECIALI: (dalla documentazione dataset)
# Ci sono 5 osservazioni che un docente potrebbe voler rimuovere dal set di dati 
# prima di fornirlo agli studenti (un grafico del PREZZO DI VENDITA rispetto 
# alla SUPERFICIE ABITABILE GRAVE le indicherà rapidamente). Tre di queste sono
# veri e propri outlier (vendite parziali che probabilmente non rappresentano 
# i valori di mercato effettivi) e due sono semplicemente vendite insolite 
# (case molto grandi con prezzi relativamente appropriati). 
# Consiglierei di rimuovere dal set di dati tutte le case con più di 4000 piedi
# quadrati (il che elimina queste 5 osservazioni insolite) prima di assegnarlo 
# agli studenti.

dat <- dat %>% filter(Gr_Liv_Area <= 4000)

## La variabile target del progetto è la seguente

hist(dat$Sale_Price)
boxplot(dat$Sale_Price)

## è una distribuzione asimmetrica positiva. 
## Possiamo risolvere usando il logaritmo per il prezzo. 

dat$Sale_Price <- log(dat$Sale_Price)
boxplot(dat$Sale_Price)

## Vediamo adesso il grafico

hist(dat$Sale_Price)
boxplot(dat$Sale_Price)

## Siccome neighborhood presenta pochi valori in alcuni livelli, li accorpo in
## un'altro livello:

dat <- dat %>%
  mutate(Neighborhood = fct_lump(Neighborhood, n = 25, other_level = "Other"))

# Ci sono alcune variabili che hanno un elevato numero di 0.
colSums(dat==0)

## Lot_frontage presenta 490 valori pari a 0. Questo sembra essere impossibile
## in quanto ogni casa per essere raggiunta deve essere collegata ad una strada.

## Ci potrebbero essere una serie di variabili che possono essere collegate a
## Lot_Frontage. Ad esempio Lot_area che è espressa in quadrati, quindi 
## potrebbe entrare in una regressione lineare attraverso una trasformazione
## come la radice quadrata.

cor(dat$Lot_Frontage, sqrt(dat$Lot_Area))

## Lot configuration pure potrebbe essere collegata. 
## Neighborhood, questa indica il vicinato.
## Condition 1 e Condition 2 danno un'indicazione di che strada si trova vicino. 
## Garage Type e Garage Area sono utili perchè una casa che ha un garage è 
## sicuramente connessa ad una strada. 
## Ms Zoning
## Alley

recipe <- formula (Lot_Frontage ~ sqrt(Lot_Area) + Neighborhood + Lot_Config+  
                     Lot_Shape + Condition_1 + Condition_2 +Garage_Type + 
                     Garage_Area + MS_Zoning + Alley)

## Vediamo la relazione tra alcune variabili:

plot(dat$Lot_Frontage, sqrt(dat$Lot_Area))

## Nella fase previsiva vorremmo non tener conto di eventuali valori anomali.

boxplot(sqrt(dat$Lot_Area))

## Identifichiamoli con regola di Tukey, utillizziamo 3 range interquartili,
## la distribuzione di lot area è normale sia asimmetrica a destra. 

q1_lot_area <- quantile(sqrt(dat$Lot_Area), 0.25)
q3_lot_area <- quantile(sqrt(dat$Lot_Area), 0.75)

IQR <- q3_lot_area - q1_lot_area
lower_bound_age <- q1_lot_area - 3 * IQR
upper_bound_age <- q3_lot_area + 3 * IQR

obs_out <- which(sqrt(dat$Lot_Area) <= lower_bound_age | 
                   sqrt(dat$Lot_Area) >= upper_bound_age )


obs_out         

nrow(dat)
dat <- dat[-obs_out,] 
nrow(dat)

## Proviamo a fare un'imputazione utilizzando una regressione lineare con le 
## variabili selezionate. 

dat <- dat %>%
  mutate(Lot_Frontage = ifelse(Lot_Frontage == 0, NA, Lot_Frontage))

# Divido i dati per imputare i valori mancanti
train_data <- dat %>% filter(!is.na(Lot_Frontage))
test_data <- dat %>% filter(is.na(Lot_Frontage))

# Creo il modello
model <- lm(Lot_Frontage ~ sqrt(Lot_Area) + Neighborhood + Lot_Config + Lot_Shape, 
            data = train_data)
# Faccio le previsioni
pred <- predict(model, newdata = test_data)

# Assegno nuovi valori
dat$Lot_Frontage[is.na(dat$Lot_Frontage)] <- pred


## Per semplificare alcune variabili senza perdere informazione, creo alcune 
## nuove variabili ed elimino alcune inutili osservazioni poco informative.
## Ad esempio pool area è praticamente 0, tranne che per 14 osservazioni. Lo
## stessor ragionamento lo facciamo per Pool_QC.
## Così come three_season_porch, Screen Porch and Enclosed Porch
## Garage_Cars è un duplicato. 
## Bsmt_Fin_Sf_2 ha 2546 valori pari a 0, il suo apporto significativo
## può essere trascurabile.
## Lo stesso ragionamento per misc_val

dat$Pool_Area <- NULL
dat$Pool_QC <- NULL
dat$Three_season_porch <- NULL
dat$Enclosed_Porch <- NULL
dat$Screen_Porch <- NULL
dat$Garage_Cars <- NULL
dat$BsmtFin_SF_2 <- NULL
dat$Misc_Val <- NULL

## Dato che la posizione ci viene data dal vicinato, è inutile riportare come 
## variabili Langitudine e Lotitudine 
dat$Longitude <- NULL
dat$Latitude <- NULL

## Creiamo la feature Total_Bath

dat <- dat %>%
  mutate(Total_Bath = Bsmt_Full_Bath + Full_Bath + 
           0.5*Bsmt_Half_Bath + 0.5 * Half_Bath)

## Togliamo ora quindi queste variabili

dat <- dat %>% select(- Bsmt_Full_Bath, -Bsmt_Half_Bath,
                      -Full_Bath, -Half_Bath)

## Vediamo un grafico che mostra le correlazioni tra le variabili per vedere 
## anche eventuali problemi di multicollinearità.

dat %>% select_if(is.numeric) %>% cor() %>% corrplot()

## Ensemble delle 3 reti neurali

set.seed(1234)
split_data <-initial_split(dat, prop = 0.8)
dat_train <- training(split_data)
dat_test <- testing(split_data)

all_recipe <- recipe(Sale_Price ~ ., data = dat_train) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors())


## Modello 1

brulee_model<-mlp(
  epochs =700,
  hidden_units=c(25L,25L),
  activation = c("tanh","tanh"),
  penalty=0.001,
  learn_rate = 0.01
)%>%
  set_mode("regression")%>%
  set_engine("brulee",optimizer="LBFGS",verbose=FALSE)

brulee_wf <- workflow() %>%
  add_recipe(all_recipe) %>%
  add_model(brulee_model)

pred <- predict(final_model, dat_test) %>%
  bind_cols(dat_test)

rmse(pred, truth = Sale_Price, estimate = .pred)
mae(pred, truth = Sale_Price, estimate = .pred)
rsq(pred, truth = Sale_Price, estimate = .pred)

pred %>% 
  ggplot(aes(x = Sale_Price, y = .pred)) +
  geom_point(alpha = 0.5) +
  geom_abline(lty = 2, color = "red") +
  labs(title = "Rete neurale (25,25),LBFGS, f = tanh, penalty = 0.001, learn_rate = 0.01, epochs = 700 ", 
       x = "Log Prezzo Reale", y = "Log Prezzo Predetto")


## Modello 2

brulee_model_2 <- mlp(
  hidden_units = 8,
  activation = 'gelu',
  penalty = 0.001,
  learn_rate = 0.01,
  epochs = 500
) %>% set_mode('regression') %>% 
  set_engine("brulee",optimizer="LBFGS",verbose=FALSE)

brulee_wf_2 <- workflow() %>%
  add_recipe(all_recipe) %>%
  add_model(brulee_model_2)

final_model <- fit(brulee_wf_2, data = dat_train)

pred <- predict(final_model, dat_test) %>%
  bind_cols(dat_test)

rmse(pred, truth = Sale_Price, estimate = .pred)
mae(pred, truth = Sale_Price, estimate = .pred)
rsq(pred, truth = Sale_Price, estimate = .pred)

pred %>% 
  ggplot(aes(x = Sale_Price, y = .pred)) +
  geom_point(alpha = 0.5) +
  geom_abline(lty = 2, color = "red") +
  labs(title = "Rete neurale (8),LBFGS, f = gelu, penalty = 0.001, learn_rate = 0.01, epochs = 500 ", 
       x = "Log Prezzo Reale", y = "Log Prezzo Predetto")


## Modello 3

mlp_sgd_model <- mlp(
  hidden_units = 16,        # Fisso 
  penalty = 0.01,           # Fisso 
  epochs = 450,
  learn_rate = 0.00316,
  activation = 'relu'
) %>% 
  set_engine('brulee', 
             optimizer = 'SGD', 
             batch_size = 32, 
             stop_iter = 10) %>% 
  set_mode('regression')


sgd_wf <- workflow() %>%
  add_recipe(all_recipe) %>%
  add_model(mlp_sgd_model)

final_model <- fit(sgd_wf, data = dat_train)

pred <- predict(final_model, dat_test) %>%
  bind_cols(dat_test)

rmse(pred, truth = Sale_Price, estimate = .pred)
mae(pred, truth = Sale_Price, estimate = .pred)
rsq(pred, truth = Sale_Price, estimate = .pred)
pred %>% 
  ggplot(aes(x = Sale_Price, y = .pred)) +
  geom_point(alpha = 0.5) +
  geom_abline(lty = 2, color = "red") +
  labs(title = "Rete neurale (16),SGD, f = relu, penalty = 0.001, learn_rate = 0.00316, epochs = 450 ", 
       x = "Log Prezzo Reale", y = "Log Prezzo Predetto")


## Ensemble dei 3 metodi

set.seed(883)

dat_folds <- vfold_cv(dat, v=10, strata = Sale_Price, repeats = 10)

plan(multisession, workers = parallel::detectCores() - 1)

tic()

cv_results_ensemble <- future_map_dfr(dat_folds$splits, function(split) {
  
  # Carico librerie, mi dava un errore su recipes. Tuttavia
  # sarà un problema dovuto al fatto che i core non rilevano 
  # tutti i pacchetti. In questo modo dovrebbe andare bene. 
  
  library(tidymodels)
  library(recipes)
  
  # Estrazione dati del fold
  df_train <- analysis(split)
  df_test  <- assessment(split)
  
  # 1. Addestramento dei 3 modelli
  fit_1 <- brulee_wf   %>% fit(data = df_train)
  fit_2 <- brulee_wf_2 %>% fit(data = df_train)
  fit_3 <- sgd_wf %>% fit(data = df_train)
  
  # 2. Generazione delle previsioni
  pred_1 <- predict(fit_1, df_test)$.pred
  pred_2 <- predict(fit_2, df_test)$.pred
  pred_3 <- predict(fit_3, df_test)$.pred
  
  # 3. Media dell'Ensemble (Simple Averaging)
  ensemble_avg <- (pred_1 + pred_2 + pred_3) / 3
  
  # 4. Calcolo metriche
  truth_val <- df_test$Sale_Price
  
  tibble(
    rmse = rmse_vec(truth = truth_val, estimate = ensemble_avg),
    mae  = mae_vec(truth = truth_val, estimate = ensemble_avg),
    rsq  = rsq_vec(truth = truth_val, estimate = ensemble_avg)
  )
})

toc()

plan(sequential)

# Risultato finale medio dell'ensemble
ensemble_metrics <- cv_results_ensemble %>% 
  summarise(across(everything(), list(mean = mean, sd = sd)))

print(ensemble_metrics)




## Grafico finale


# 1. Trasformiamo i risultati in formato "long" per ggplot
cv_ensemble_long <- cv_results_ensemble %>%
  pivot_longer(cols = everything(), names_to = "metric", values_to = "value")

# 2. Calcoliamo le medie per le linee verticali
v_lines_ensemble <- cv_ensemble_long %>%
  group_by(metric) %>%
  summarise(mean_val = mean(value))

# 3. Creazione del Grafico
ggplot(cv_ensemble_long, aes(x = value, fill = metric)) +
  geom_histogram(bins = 10, color = "white", alpha = 0.8) +
  geom_vline(data = v_lines_ensemble, aes(xintercept = mean_val), 
             color = "red", linetype = "dashed", size = 1) +
  facet_wrap(~ metric, scales = "free") +
  # Aggiungiamo etichette con il valore medio per chiarezza
  geom_text(data = v_lines_ensemble, 
            aes(x = mean_val, y = Inf, label = round(mean_val, 3)), 
            vjust = 2, color = "red", fontface = "bold") +
  scale_fill_viridis_d(option = "mako", begin = 0.3, end = 0.7) + 
  theme_minimal() +
  labs(
    title = "Performance dell'Ensemble (3 Reti Neurali)",
    subtitle = "Media calcolata su 10-fold Cross-Validation ripetuta 10 volte",
    x = "Valore Metrica", 
    y = "Frequenza (Folds)",
    fill = "Metrica"
  )