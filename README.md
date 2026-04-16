# Neural Network Ensemble for Ames Housing Price Prediction

Questo progetto implementa un approccio ensemble basato su reti neurali per la previsione dei prezzi delle abitazioni utilizzando il dataset Ames Housing. L’obiettivo è migliorare le performance predittive combinando modelli con configurazioni differenti.

L’ensemble è composto da tre reti neurali, ciascuna addestrata con iperparametri distinti, tra cui:

funzioni di attivazione diverse,
learning rate differenti,
numero di epoche variabile.

Questa diversificazione consente di catturare differenti pattern nei dati e ridurre il rischio di overfitting associato a un singolo modello.

Il workflow include:

preprocessing e feature engineering del dataset,
addestramento indipendente dei modelli,
combinazione delle predizioni tramite tecniche di ensemble (es. media o voting),
valutazione delle performance su un test set.
