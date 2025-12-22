Système de Prévision des Prix Alimentaires
(Food Price Forecasting System)

 A/ Contexte et Objectif

Ce projet vise à analyser, prévoir et interpréter l’évolution des prix des denrées alimentaires à partir de :

  --> Données FAO agrégées (indices globaux : FIP, Cereals, Oils, etc.)

  --> Données détaillées par pays, marché et denrée (Bénin, Sénégal, Burkina Faso, Nigeria)

B/ L’application est conçue comme un outil d’aide à la décision pour :

-> décideurs publics

-> institutions économiques

-> analystes data


C/  Fonctionnalités principales
* Exploration des données

- Visualisation interactive des séries temporelles

- Statistiques descriptives

- Indicateurs clés (min, max, moyenne, évolution)

- Affichage tendance

*  Filtres dynamiques

-> Choix de la source de données

-> Sélection du pays

-> Sélection de la denrée / indice

-> Choix du modèle de prévision

-> Choix de la période de prévision

* Modèles de prévision implémentés

- Prophet (Meta) – recommandé pour séries temporelles

- ARIMA – approche économétrique classique

- Random Forest – machine learning supervisé

- LSTM – deep learning (réseaux récurrents)

D/  Résultats

-> Graphes historiques + prévisions

-> Intervalles de confiance (si disponibles)

-> Tableaux détaillés des prévisions

-> Interprétation automatique pour décideurs
(hausse, baisse, stabilité + recommandations concrètes)

* Export

Téléchargement des prévisions en CSV et Excel

E/  Architecture du projet

ECP_Kristen-vinia/
│
├── app.py                     # Application Streamlit principale
│
├── data/
│   ├── fao_data.csv            # Données FAO agrégées
│   └── All_countries.csv       # Données détaillées (pays / marchés)
│
├── utils 
│    ├──data_processing.py          # Préparation des données
│
├── models/
│   ├── prophet_model.py
│   ├── arima_model.py
│   ├── random_forest_model.py
│   └── lstm_model.py
│
├── requirements.txt
└── README.md

F/ Logique des modèles
         FAO (données agrégées)

-> Série temporelle unique

-> Modèles univariés (Prophet, ARIMA, LSTM)

         Données détaillées

-> Séries filtrées par pays + denrée

-> Agrégation temporelle (par date)

-> Random Forest et LSTM adaptés aux données riches

G/ Interprétation automatique (Decision Support)

Pour chaque prévision, l’application génère un texte explicatif :

📈 Hausse attendue → pression inflationniste

📉 Baisse attendue → opportunités de stabilisation

➖ Stabilité → maintien des politiques actuelles

Ces textes sont conçus pour être :

-> compréhensibles

-> exploitables par des non-techniques

-> mais sous réserve d'une version 2

H/  Installation et exécution

   1- Cloner le projet
    git clone 
    cd 

   2- Créer un environnement virtuel
    python3 -m venv venv
    source venv/bin/activate

   3-  Installer les dépendances
    pip install -r requirements.txt

         OU 
    pip install pmdarima statsmodels
    pip install prophet
    pip install tensorflow
    pip install scikit-learn
    pip install plotly
    

   4- Lancer l’application
    streamlit run app.py

I/ Stack technique

-> Python

-> Streamlit

-> Pandas / NumPy

-> Plotly

-> Scikit-learn

-> Statsmodels

-> Prophet

-> TensorFlow / Keras

L/ Cadre académique

Ce projet s’inscrit dans un cadre de :

-> Data Analyse

-> Data Science

-> Économétrie

-> Aide à la décision basée sur les données

👥 Auteurs

Kristen

Vinia

2025

📌 Perspectives d’amélioration

Ajout de variables exogènes (climat, importations)

API pour données en temps réel

Déploiement

Modèles multivariés avancés
