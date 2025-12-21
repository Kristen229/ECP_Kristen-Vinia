import streamlit as st
import pandas as pd
import os
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------
# Vérifie si les modèles sont disponibles
# -----------------------------

# Imports des modèles (avec gestion d'erreur)
try:
    from models.prophet_model import train_prophet, predict_prophet, evaluate_prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    st.warning("⚠️ Prophet non disponible. Installez : pip install prophet")

try:
    from models.arima_model import train_arima, predict_arima, evaluate_arima
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    st.warning("⚠️ ARIMA non disponible. Installez : pip install pmdarima statsmodels")

try:
    from models.lstm_model import train_lstm, predict_lstm, evaluate_lstm
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False
    st.warning("⚠️ LSTM non disponible. Installez : pip install tensorflow")

try:
    from models.random_forest_model import train_rf, predict_rf, evaluate_rf
    RF_AVAILABLE = True
except ImportError:
    RF_AVAILABLE = False
    st.warning("⚠️ Random Forest non disponible. Installez : pip install scikit-learn")



from utils.data_processing import load_and_prepare_data

# -----------------------------
# Configuration de la page
# -----------------------------
st.set_page_config(
    page_title="Prévisions Prix Alimentaires",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Style CSS personnalisé
# -----------------------------
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
        
    }
    .stMetric {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: white;
    }
    .stDownloadButton button {
        background-color: #ff7f0e;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Titre principal
# -----------------------------
st.title("📊 Système de Prévision des Prix Alimentaires")
st.markdown("### Analyse prédictive basée sur les modèles")

# ==================== SIDEBAR ====================
st.sidebar.header("⚙️ Configuration")

# 1. Choix de la source de données
data_source = st.sidebar.radio(
    "Source de données",
    ["Données FAO (Générales)", "Données Détaillées "],
    help="Sélectionnez la base de données à utiliser"
)

# -----------------------------
# 2. Chargement des données
# -----------------------------
@st.cache_data
def load_data(source):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")

    try:
        if source == "Données FAO (Générales)":
            file_path = os.path.join(data_dir, "fao_data.csv")
        else:
            file_path = os.path.join(data_dir, "All_countries.csv")

        df = pd.read_csv(file_path, sep=';')

        if 'index' in df.columns:
            df = df.drop('index', axis=1)

        # S'assurer que la colonne Date existe
        if 'date' in df.columns:
            df['Date'] = pd.to_datetime(df['date'], errors="coerce")
        elif 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors="coerce")

        return df

    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        return None

# Charger les données
df = load_data(data_source)
if df is None or df.empty:
    st.error(" Les données n'ont pas été chargées correctement.")
    st.stop()

df_filtered = df.copy()

# -----------------------------
# 3. Sélection Pays / Denrée si données détaillées
# -----------------------------
if data_source == "Données Détaillées ":
    countries_available = ['Benin', 'Senegal', 'Burkina Faso', 'Nigeria']
    selected_country = st.sidebar.selectbox("Pays", countries_available, index=0)
    df_filtered = df_filtered[df_filtered['country_name'] == selected_country]

    commodities_available = df_filtered['commodity_name'].unique()
    selected_commodity = st.sidebar.selectbox("Denrée", commodities_available, index=0)
    df_filtered = df_filtered[df_filtered['commodity_name'] == selected_commodity]


    df_filtered = df[(df['country_name'] == selected_country) &
    (df['commodity_name'] == selected_commodity)].copy()
    
    last_date = df['Date'].max()
    st.write(f"Dernière date disponible dans la base : {last_date}")

# -----------------------------
# 4. Sélection de l'indice ou prix
# -----------------------------
# Si données FAO générales, utiliser les indices
available_indices = ['FIP', 'Meat', 'Dairy', 'Cereals', 'Oils', 'Sugar']
selected_index = available_indices[0]  # par défaut

if data_source == "Données FAO (Générales)":
    selected_index = st.sidebar.selectbox(
        "Indice à prédire",
        available_indices,
        index=0,
        help="Choisissez l'indicateur de prix"
    )
else:
    # Si données détaillées, on prédit le prix
    selected_index = 'price'

# -----------------------------
# 5. Sélection du modèle
# -----------------------------
models_available = {}
if PROPHET_AVAILABLE:
    models_available["Prophet (Recommandé)"] = "prophet"
if ARIMA_AVAILABLE:
    models_available["ARIMA (Classique)"] = "arima"
if LSTM_AVAILABLE:
    models_available["LSTM (Deep Learning)"] = "lstm"
if RF_AVAILABLE:
    models_available["Random Forest"] = "random_forest"

if not models_available:
    st.error(" Aucun modèle disponible. Installez au moins un modèle.")
    st.stop()

selected_model_name = st.sidebar.selectbox(" Modèle de prévision", list(models_available.keys()))
selected_model = models_available[selected_model_name]

# -----------------------------
# 6. Période de prévision
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.subheader(" Période de prévision")

forecast_options = {
    "6 mois": 6,
    "1 an": 12,
    "2 ans": 24,
    "3 ans": 36,
    "Personnalisé": None
}
forecast_period = st.sidebar.selectbox("Durée", list(forecast_options.keys()))

if forecast_period == "Personnalisé":
    custom_months = st.sidebar.slider("Nombre de mois", 1, 60, 12)
    n_months = custom_months
else:
    n_months = forecast_options[forecast_period]

# Date de début de prévision
last_date = df_filtered['Date'].max()
start_forecast_date = st.sidebar.date_input(
    "Date de début",
    value=last_date + timedelta(days=30),
    min_value=last_date
)

# ==================== ONGLETS PRINCIPAUX ====================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Vue d'ensemble",
    "🔮 Prévisions",
    "📈 Comparaison Modèles",
    "📥 Téléchargements"
])

# ==================== TAB 1 : VUE D'ENSEMBLE ====================
with tab1:
    st.header("📊 Analyse des Données Historiques")
    
    # Métriques clés
    col1, col2, col3, col4 = st.columns(4)
    
    current_value = df_filtered[selected_index].iloc[-1]
    previous_value = df_filtered[selected_index].iloc[-2]
    change = ((current_value - previous_value) / previous_value) * 100
    
    with col1:
        st.metric(
            "Valeur actuelle",
            f"{current_value:.2f}",
            f"{change:+.2f}%"
        )
    
    with col2:
        st.metric(
            "Moyenne historique",
            f"{df_filtered[selected_index].mean():.2f}"
        )
    
    with col3:
        st.metric(
            "Maximum",
            f"{df_filtered[selected_index].max():.2f}"
        )
    
    with col4:
        st.metric(
            "Minimum",
            f"{df_filtered[selected_index].min():.2f}"
        )

    # Graphique historique interactif
    st.subheader("Évolution Historique")
    
    df_plot = df_filtered.sort_values('Date').groupby('Date')[selected_index].mean().reset_index()


    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_plot['Date'],
        y=df_plot[selected_index],
        mode='lines',
        name='Historique',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.2)'
    ))
    
    fig.update_layout(
        title=f"Évolution de {selected_index}",
        xaxis_title="Date",
        yaxis_title=selected_index,
        hovermode='x unified',
        height=500,
        template='plotly_dark'
    )
    
    
    st.plotly_chart(fig, use_container_width=True)
    
# ==================== TAB 2 : PRÉVISIONS ====================
with tab2:
    st.header("🔮 Prévisions avec " + selected_model_name)
    
    # Bouton pour lancer la prévision
    if st.button(" Lancer la Prévision", type="primary", use_container_width=True):
        with st.spinner(f"Entraînement du modèle {selected_model_name}..."):
            
            # Préparer les données
            df_model = load_and_prepare_data(df_filtered, selected_index)
            
            # Entraîner et prédire selon le modèle
            try:
                if selected_model == "prophet" and PROPHET_AVAILABLE:
                    model = train_prophet(df_model, selected_index)
                    forecast_df = predict_prophet(model, n_months)
                    
                elif selected_model == "arima" and ARIMA_AVAILABLE:
                    model = train_arima(df_model, selected_index)
                    forecast_df = predict_arima(model, n_months)
                    
                elif selected_model == "lstm" and LSTM_AVAILABLE:
                    model, scaler = train_lstm(df_model, selected_index)
                    forecast_df = predict_lstm(
                        model,
                        scaler,
                        df_model,
                        selected_index,  # ou target_index
                        n_months
                    )
                    """ forecast_df = predict_lstm(model, scaler, df_model, n_months) """
                    
                elif selected_model == "random_forest" and RF_AVAILABLE:
                    model = train_rf(df_model, selected_index)
                    forecast_df = predict_rf(
                        model,
                        df_model,
                        selected_index,
                        n_months
                    )
                
                st.success(f" Prévision réussie pour {n_months} mois !")
                
                # Sauvegarder dans session state
                st.session_state['forecast_df'] = forecast_df
                st.session_state['model_name'] = selected_model_name
                
            except Exception as e:
                st.error(f" Erreur lors de la prévision : {e}")
                st.exception(e)
                st.stop()
    
    # Afficher les résultats si disponibles
    
    fig = go.Figure()

    if 'forecast_df' in st.session_state:
        forecast_df = st.session_state['forecast_df']
        
        # Graphique de prévision
        fig = go.Figure()
        df_plot = df_filtered.sort_values('Date').groupby('Date')[selected_index].mean().reset_index()

        # Données historiques
        fig.add_trace(go.Scatter(
            x=df_plot['Date'],
            y=df_plot[selected_index],
            mode='lines',
            name='Historique',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Prévisions
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Forecast'],
            mode='lines+markers',
            name='Prévisions',
            line=dict(color='#ff7f0e', width=3, dash='dash'),
            marker=dict(size=8)
        ))
        
        # Intervalle de confiance (si disponible)
        if 'Lower_CI' in forecast_df.columns:
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'],
                y=forecast_df['Upper_CI'],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'],
                y=forecast_df['Lower_CI'],
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(255, 127, 14, 0.2)',
                line=dict(width=0),
                name='Intervalle de confiance 95%'
            ))
        
        fig.update_layout(
            title=f"Prévisions {selected_index} - {st.session_state['model_name']}",
            xaxis_title="Date",
            yaxis_title=selected_index,
            hovermode='x unified',
            height=600,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)

        ####

        start_value = df_filtered[selected_index].iloc[-1]
        end_value = forecast_df['Forecast'].iloc[-1]

        growth_pct = ((end_value - start_value) / start_value) * 100
        volatility = forecast_df['Forecast'].std()

        ###

        if growth_pct > 5:
            trend = "hausse"
        elif growth_pct < -5:
            trend = "baisse"
        else:
            trend = "stabilité"

        ####POUR LES DECIDEURS 
        st.subheader(" Analyse et recommandations")
        if trend == "hausse":
            explanation = f"""
            📈 **Tendance haussière attendue sur {n_months} mois**

            Les projections indiquent une **augmentation d’environ {growth_pct:.1f}%** 
            de l’indice **{selected_index}** sur la période considérée.

            -> Cela suggère une **pression à la hausse des prix alimentaires**, 
            pouvant affecter le pouvoir d’achat des ménages.

             **Recommandations possibles** :
            - Anticiper des mesures de régulation des prix
            - Renforcer les stocks stratégiques
            - Soutenir la production locale
            """

        elif trend == "baisse":
            explanation = f"""
            📉 **Tendance baissière attendue sur {n_months} mois**

            Les prix de **{selected_index}** pourraient diminuer d’environ 
            **{abs(growth_pct):.1f}%**, traduisant une amélioration de l’offre.

             **Opportunités** :
            - Réduction des coûts d’importation
            - Stabilisation des marchés locaux
            - Allègement de la pression inflationniste
            """

        else:
            explanation = f"""
            ➖ **Stabilité relative attendue**

            Les prévisions montrent une **variation limitée ({growth_pct:.1f}%)**
            de l’indice **{selected_index}**, indiquant un marché relativement stable.

             **Actions suggérées** :
            - Maintenir les politiques actuelles
            - Surveiller les chocs externes (climat, géopolitique)
            """
        st.info(explanation)

                
        # Tableau des prévisions
        st.subheader("📋 Détails des Prévisions")
        
        # Formater le tableau
        display_df = forecast_df.copy()
        display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m')
        
        if 'Lower_CI' in display_df.columns:
            display_df = display_df.rename(columns={
                'Forecast': 'Prévision',
                'Lower_CI': 'Limite Inf. (95%)',
                'Upper_CI': 'Limite Sup. (95%)'
            })
        else:
            display_df = display_df.rename(columns={'Forecast': 'Prévision'})
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Statistiques de prévision
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Prévision moyenne", f"{forecast_df['Forecast'].mean():.2f}")
        with col2:
            st.metric("Prévision maximale", f"{forecast_df['Forecast'].max():.2f}")
        with col3:
            st.metric("Prévision minimale", f"{forecast_df['Forecast'].min():.2f}")


# ==================== TAB 3 et TAB 4 ====================
with tab3:
    st.header("📈 Comparaison des Modèles")
    st.info("Cette section permet de comparer les performances de tous les modèles sur les données historiques")
    
    if st.button("⚡ Comparer tous les modèles", use_container_width=True):
        with st.spinner("Entraînement de tous les modèles..."):
            
            results = []
            df_model = load_and_prepare_data(df_filtered, selected_index)
            
            # Entraîner tous les modèles disponibles
            for model_name, model_type in models_available.items():
                try:
                    st.write(f"Entraînement : {model_name}...")
                    
                    if model_type == "prophet" and PROPHET_AVAILABLE:
                        model = train_prophet(df_model, selected_index)
                        metrics = evaluate_prophet(model, df_model, selected_index)
                    elif model_type == "arima" and ARIMA_AVAILABLE:
                        model = train_arima(df_model, selected_index)
                        metrics = evaluate_arima(model, df_model, selected_index)
                    elif model_type == "lstm" and LSTM_AVAILABLE:
                        model, scaler = train_lstm(df_model, selected_index)
                        metrics = evaluate_lstm(model, scaler, df_model, selected_index)
                    elif model_type == "random_forest" and RF_AVAILABLE:
                        model = train_rf(df_model, selected_index)
                        metrics = evaluate_rf(model, df_model, selected_index)
                    
                    results.append({
                        'Modèle': model_name,
                        'MAE': metrics['mae'],
                        'RMSE': metrics['rmse'],
                        'MAPE': metrics['mape']
                    })
                    
                except Exception as e:
                    st.warning(f" Erreur avec {model_name}: {e}")
            
            # Afficher résultats
            if results:
                results_df = pd.DataFrame(results).sort_values('MAPE')
                
                st.subheader("🏆 Classement des Modèles")
                st.dataframe(results_df, use_container_width=True)
                
                # Graphique comparatif
                fig = px.bar(
                    results_df,
                    x='Modèle',
                    y='MAPE',
                    title="MAPE par Modèle (plus bas = meilleur)",
                    color='MAPE',
                    color_continuous_scale='RdYlGn_r'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommandation
                best_model = results_df.iloc[0]
                st.success(f"""
                ### ✨ Recommandation
                Le meilleur modèle pour vos données est **{best_model['Modèle']}** 
                avec un MAPE de **{best_model['MAPE']:.2f}%**
                """)

# ==================== TAB 4 : TÉLÉCHARGEMENTS ====================
with tab4:
    st.header(" Télécharger les Résultats")
    
    if 'forecast_df' in st.session_state:
        forecast_df = st.session_state['forecast_df']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💾 Prévisions (CSV)")
            csv = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📊 Télécharger CSV",
                data=csv,
                file_name=f"previsions_{selected_index}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            st.subheader("📊 Prévisions (Excel)")
            from io import BytesIO
            buffer = BytesIO()
            
            try:
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    forecast_df.to_excel(writer, sheet_name='Prévisions', index=False)
                    df_filtered.to_excel(writer, sheet_name='Données Historiques', index=False)
                
                st.download_button(
                    label="📈 Télécharger Excel",
                    data=buffer.getvalue(),
                    file_name=f"previsions_{selected_index}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                    mime="application/vnd.ms-excel",
                    use_container_width=True
                )
            except ImportError:
                st.warning("⚠️ Installez xlsxwriter pour l'export Excel : pip install xlsxwriter")
    else:
        st.warning("⚠️ Aucune prévision disponible. Lancez d'abord une prévision dans l'onglet 'Prévisions'.")



# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #aaa;'>
        <p>Système de Prévision des Prix Alimentaires | Développé avec Kristen & Vinia | 2025</p>
    </div>
""", unsafe_allow_html=True)
