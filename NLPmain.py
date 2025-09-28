# ============================================
# 🎯 NLP Analysis Suite
# ============================================

import streamlit as st
import pandas as pd
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from textblob import TextBlob

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================
# Page Configuration
# ============================
st.set_page_config(
    page_title="NLP Analyzer Pro",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
# Professional CSS Styling
# ============================
st.markdown("""
<style>
    /* Modern Professional Color Scheme */
    :root {
        --primary: #2563eb;
        --primary-dark: #1d4ed8;
        --secondary: #64748b;
        --accent: #06d6a0;
        --background: #f8fafc;
        --card: #ffffff;
        --text: #1e293b;
        --text-light: #64748b;
        --border: #e2e8f0;
    }
    
    /* Main container */
    .main-container {
        background-color: var(--background);
        padding: 2rem;
    }
    
    /* Cards */
    .professional-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .professional-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    
    /* Metrics */
    .metric-card {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Sections */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text);
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--border);
    }
    
    /* Sidebar */
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--text);
        margin-bottom: 1rem;
    }
    
    /* Buttons */
    .stButton button {
        width: 100%;
        background: var(--primary);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        background: var(--primary-dark);
        transform: translateY(-1px);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: var(--card);
        border-radius: 8px 8px 0px 0px;
        gap: 1rem;
        padding: 1rem 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# Initialize NLP
# ============================
@st.cache_resource
def load_nlp_model():
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        st.error("""
        **SpaCy English model not found.** 
        Please install: `python -m spacy download en_core_web_sm`
        """)
        st.stop()

nlp = load_nlp_model()
stop_words = STOP_WORDS

# ============================
# Feature Engineering Classes
# ============================
class ProfessionalFeatureExtractor:
    @staticmethod
    def extract_lexical_features(texts):
        """Extract lexical features with advanced preprocessing"""
        processed_texts = []
        for text in texts:
            doc = nlp(str(text).lower())
            tokens = [token.lemma_ for token in doc if token.text not in stop_words and token.is_alpha]
            processed_texts.append(" ".join(tokens))
        return TfidfVectorizer(max_features=1000, ngram_range=(1, 2)).fit_transform(processed_texts)
    
    @staticmethod
    def extract_semantic_features(texts):
        """Extract semantic features with sentiment analysis"""
        features = []
        for text in texts:
            blob = TextBlob(str(text))
            features.append([
                blob.sentiment.polarity,
                blob.sentiment.subjectivity,
                len(text.split()),
                len([word for word in text.split() if len(word) > 6]),
            ])
        return np.array(features)
    
    @staticmethod
    def extract_syntactic_features(texts):
        """Extract syntactic features with POS analysis"""
        processed_texts = []
        for text in texts:
            doc = nlp(str(text))
            pos_tags = [f"{token.pos_}_{token.tag_}" for token in doc]
            processed_texts.append(" ".join(pos_tags))
        return CountVectorizer(max_features=800, ngram_range=(1, 3)).fit_transform(processed_texts)
    
    @staticmethod
    def extract_pragmatic_features(texts):
        """Extract pragmatic features - context and intent analysis"""
        pragmatic_features = []
        pragmatic_indicators = {
            'modality': ['must', 'should', 'could', 'would', 'might', 'may'],
            'certainty': ['certainly', 'definitely', 'obviously', 'clearly'],
            'uncertainty': ['perhaps', 'maybe', 'possibly', 'probably'],
            'question': ['what', 'why', 'how', 'when', 'where', 'which', '?'],
            'emphasis': ['very', 'extremely', 'highly', 'absolutely']
        }
        
        for text in texts:
            text_lower = str(text).lower()
            features = []
            
            for category, words in pragmatic_indicators.items():
                count = sum(text_lower.count(word) for word in words)
                features.append(count)
            
            features.extend([
                text.count('!'),
                text.count('?'),
                len([s for s in text.split('.') if s.strip()]),
                len([w for w in text.split() if w.istitle()]),
            ])
            
            pragmatic_features.append(features)
        
        return np.array(pragmatic_features)

# ============================
# Professional Model Trainer
# ============================
class ProfessionalModelTrainer:
    def __init__(self):
        self.models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced'),
            "Support Vector Machine": SVC(random_state=42, probability=True, class_weight='balanced'),
            "Naive Bayes": MultinomialNB()
        }
    
    def train_and_evaluate(self, X, y):
        """Professional model training with comprehensive evaluation"""
        results = {}
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        n_classes = len(le.classes_)
        
        test_size = max(0.15, min(0.25, 3 * n_classes / len(y_encoded)))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (name, model) in enumerate(self.models.items()):
            status_text.text(f"Training {name}...")
            
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'model': model,
                    'predictions': y_pred,
                    'true_labels': y_test,
                    'probabilities': y_proba,
                    'n_classes': n_classes,
                    'test_size': len(y_test)
                }
                
            except Exception as e:
                results[name] = {'error': str(e)}
            
            progress_bar.progress((i + 1) / len(self.models))
        
        progress_bar.empty()
        status_text.text("Training completed!")
        
        return results, le

# ============================
# Interactive Visualizations
# ============================
class InteractiveVisualizer:
    @staticmethod
    def create_performance_radar(results):
        """Create interactive radar chart"""
        models = []
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = []
        
        for model_name, result in results.items():
            if 'error' not in result:
                models.append(model_name)
                values.append([
                    result['accuracy'],
                    result['precision'],
                    result['recall'],
                    result['f1_score']
                ])
        
        fig = go.Figure()
        
        colors = ['#2563eb', '#06d6a0', '#ffd166', '#ef476f']
        
        for i, model_values in enumerate(values):
            fig.add_trace(go.Scatterpolar(
                r=model_values + [model_values[0]],  # Close the circle
                theta=metrics + [metrics[0]],
                fill='toself',
                name=models[i],
                line=dict(color=colors[i], width=2),
                opacity=0.8
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            title="Model Performance Radar",
            font=dict(size=12),
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_metrics_comparison(results):
        """Create interactive metrics comparison"""
        models = []
        metrics_data = {
            'Accuracy': [],
            'Precision': [],
            'Recall': [],
            'F1-Score': []
        }
        
        for model_name, result in results.items():
            if 'error' not in result:
                models.append(model_name)
                metrics_data['Accuracy'].append(result['accuracy'])
                metrics_data['Precision'].append(result['precision'])
                metrics_data['Recall'].append(result['recall'])
                metrics_data['F1-Score'].append(result['f1_score'])
        
        fig = go.Figure()
        
        colors = ['#2563eb', '#06d6a0', '#ffd166', '#ef476f']
        
        for i, (metric, values) in enumerate(metrics_data.items()):
            fig.add_trace(go.Bar(
                name=metric,
                x=models,
                y=values,
                marker_color=colors[i],
                opacity=0.8
            ))
        
        fig.update_layout(
            title="Performance Metrics Comparison",
            xaxis_title="Models",
            yaxis_title="Score",
            barmode='group',
            height=400,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_confusion_matrix(results, model_name):
        """Create confusion matrix for a specific model"""
        if model_name in results and 'error' not in results[model_name]:
            result = results[model_name]
            cm = confusion_matrix(result['true_labels'], result['predictions'])
            
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=[f'Predicted {i}' for i in range(cm.shape[1])],
                y=[f'Actual {i}' for i in range(cm.shape[0])],
                colorscale='Blues',
                showscale=True
            ))
            
            fig.update_layout(
                title=f"Confusion Matrix - {model_name}",
                xaxis_title="Predicted Label",
                yaxis_title="True Label",
                height=400
            )
            
            return fig
        return None

# ============================
# Sidebar Configuration
# ============================
def setup_sidebar():
    """Setup the sidebar for file upload and analysis configuration"""
    st.sidebar.markdown("<div class='sidebar-header'>📁 Data Configuration</div>", unsafe_allow_html=True)
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV File",
        type=["csv"],
        help="Upload your dataset in CSV format"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.file_uploaded = True
            
            st.sidebar.success(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Column selection
            st.sidebar.markdown("<div class='sidebar-header'>⚙️ Analysis Setup</div>", unsafe_allow_html=True)
            
            text_col = st.sidebar.selectbox(
                "Text Column",
                df.columns,
                help="Select the column containing text data"
            )
            
            target_col = st.sidebar.selectbox(
                "Target Column",
                df.columns,
                help="Select the column containing labels"
            )
            
            feature_type = st.sidebar.selectbox(
                "Feature Type",
                [
                    "Lexical Features",
                    "Semantic Features", 
                    "Syntactic Features",
                    "Pragmatic Features"
                ],
                help="Select the type of features to extract"
            )
            
            # Advanced options
            with st.sidebar.expander("Advanced Options"):
                test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
                max_features = st.slider("Max Features", 100, 2000, 1000, 100)
            
            st.session_state.config = {
                'text_col': text_col,
                'target_col': target_col,
                'feature_type': feature_type,
                'test_size': test_size,
                'max_features': max_features
            }
            
            # Analysis button
            if st.sidebar.button("🚀 Start Analysis", use_container_width=True):
                st.session_state.analyze_clicked = True
            else:
                st.session_state.analyze_clicked = False
                
        except Exception as e:
            st.sidebar.error(f"Error reading file: {str(e)}")
    else:
        st.session_state.file_uploaded = False
        st.session_state.analyze_clicked = False

# ============================
# Main Content
# ============================
def main_content():
    """Main content area displaying analysis results"""
    
    if not st.session_state.get('file_uploaded', False):
        show_welcome_screen()
        return
    
    df = st.session_state.df
    config = st.session_state.get('config', {})
    
    # Display dataset overview
    st.markdown("<div class='section-header'>📊 Dataset Overview</div>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df.shape[0]}</div>
            <div class="metric-label">Total Records</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df.shape[1]}</div>
            <div class="metric-label">Features</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df.isnull().sum().sum()}</div>
            <div class="metric-label">Missing Values</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{df[config.get('target_col', '')].nunique() if config.get('target_col') in df.columns else 0}</div>
            <div class="metric-label">Unique Classes</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Data preview
    with st.expander("🔍 Data Preview", expanded=True):
        tab1, tab2 = st.tabs(["First 10 Rows", "Data Statistics"])
        with tab1:
            st.dataframe(df.head(10), use_container_width=True)
        with tab2:
            st.write(df.describe(include='all'))
    
    # Analysis results
    if st.session_state.get('analyze_clicked', False):
        perform_analysis(df, config)

def show_welcome_screen():
    """Display welcome screen when no file is uploaded"""
    st.markdown("""
    <div style='text-align: center; padding: 4rem 2rem;'>
        <h1 style='color: #2563eb; font-size: 3rem; margin-bottom: 1rem;'>🔍 NLP Analyzer Pro</h1>
        <p style='color: #64748b; font-size: 1.2rem; margin-bottom: 2rem;'>
            Professional Text Analysis Platform
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="professional-card">
            <h3>📁 Upload Data</h3>
            <p>Use the sidebar to upload your CSV dataset containing text and labels</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="professional-card">
            <h3>⚙️ Configure Analysis</h3>
            <p>Select text columns, target variables, and analysis parameters</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="professional-card">
            <h3>📊 Get Insights</h3>
            <p>Receive comprehensive analysis with interactive visualizations</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div class='section-header'>✨ Key Features</div>", unsafe_allow_html=True)
    
    features = [
        {"icon": "🤖", "title": "4 ML Algorithms", "desc": "Logistic Regression, Random Forest, SVM, Naive Bayes"},
        {"icon": "🔧", "title": "Multiple Feature Types", "desc": "Lexical, Semantic, Syntactic, Pragmatic analysis"},
        {"icon": "📈", "title": "Interactive Visualizations", "desc": "Professional charts and performance metrics"},
        {"icon": "🎯", "title": "Pragmatic Analysis", "desc": "Advanced context and intent detection"}
    ]
    
    cols = st.columns(2)
    for i, feature in enumerate(features):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="professional-card">
                <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                    <span style="font-size: 2rem; margin-right: 1rem;">{feature['icon']}</span>
                    <h4 style="margin: 0; color: #1e293b;">{feature['title']}</h4>
                </div>
                <p style="color: #64748b; margin: 0;">{feature['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

def perform_analysis(df, config):
    """Perform the actual NLP analysis"""
    st.markdown("<div class='section-header'>📈 Analysis Results</div>", unsafe_allow_html=True)
    
    # Data validation
    if config['text_col'] not in df.columns or config['target_col'] not in df.columns:
        st.error("Selected columns not found in dataset.")
        return
    
    if df[config['text_col']].isnull().any():
        df[config['text_col']] = df[config['text_col']].fillna('')
    
    if df[config['target_col']].isnull().any():
        st.error("Target column contains missing values.")
        return
    
    if len(df[config['target_col']].unique()) < 2:
        st.error("Target column must have at least 2 unique classes.")
        return
    
    # Feature extraction
    with st.spinner("Extracting features..."):
        extractor = ProfessionalFeatureExtractor()
        X = df[config['text_col']].astype(str)
        y = df[config['target_col']]
        
        if config['feature_type'] == "Lexical Features":
            X_features = extractor.extract_lexical_features(X)
        elif config['feature_type'] == "Semantic Features":
            X_features = extractor.extract_semantic_features(X)
        elif config['feature_type'] == "Syntactic Features":
            X_features = extractor.extract_syntactic_features(X)
        else:  # Pragmatic Features
            X_features = extractor.extract_pragmatic_features(X)
    
    # Model training
    with st.spinner("Training machine learning models..."):
        trainer = ProfessionalModelTrainer()
        results, label_encoder = trainer.train_and_evaluate(X_features, y)
    
    # Display results
    successful_models = {k: v for k, v in results.items() if 'error' not in v}
    
    if successful_models:
        # Performance metrics
        st.markdown("#### 🎯 Model Performance")
        
        cols = st.columns(len(successful_models))
        for idx, (model_name, result) in enumerate(successful_models.items()):
            with cols[idx]:
                accuracy = result['accuracy']
                st.markdown(f"""
                <div class="professional-card">
                    <h4>{model_name}</h4>
                    <div style="font-size: 2rem; font-weight: bold; color: #2563eb; text-align: center; margin: 1rem 0;">
                        {accuracy:.1%}
                    </div>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
                        <div style="text-align: center;">
                            <small>Precision</small>
                            <div style="font-weight: bold;">{result['precision']:.3f}</div>
                        </div>
                        <div style="text-align: center;">
                            <small>Recall</small>
                            <div style="font-weight: bold;">{result['recall']:.3f}</div>
                        </div>
                        <div style="text-align: center;">
                            <small>F1-Score</small>
                            <div style="font-weight: bold;">{result['f1_score']:.3f}</div>
                        </div>
                        <div style="text-align: center;">
                            <small>Classes</small>
                            <div style="font-weight: bold;">{result['n_classes']}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Interactive visualizations
        st.markdown("#### 📊 Interactive Analytics")
        
        tab1, tab2, tab3 = st.tabs(["Performance Radar", "Metrics Comparison", "Confusion Matrix"])
        
        with tab1:
            viz = InteractiveVisualizer()
            radar_fig = viz.create_performance_radar(successful_models)
            st.plotly_chart(radar_fig, use_container_width=True)
        
        with tab2:
            metrics_fig = viz.create_metrics_comparison(successful_models)
            st.plotly_chart(metrics_fig, use_container_width=True)
        
        with tab3:
            model_choice = st.selectbox("Select Model", list(successful_models.keys()))
            cm_fig = viz.create_confusion_matrix(successful_models, model_choice)
            if cm_fig:
                st.plotly_chart(cm_fig, use_container_width=True)
        
        # Best model recommendation
        best_model = max(successful_models.items(), key=lambda x: x[1]['accuracy'])
        st.success(f"🎯 **Recommended Model**: {best_model[0]} with {best_model[1]['accuracy']:.1%} accuracy")
    
    else:
        st.error("No models were successfully trained. Please check your data and configuration.")

# ============================
# Main Application
# ============================
def main():
    # Initialize session state
    if 'file_uploaded' not in st.session_state:
        st.session_state.file_uploaded = False
    if 'analyze_clicked' not in st.session_state:
        st.session_state.analyze_clicked = False
    
    # Setup sidebar
    setup_sidebar()
    
    # Main content
    main_content()

if __name__ == "__main__":
    main()
