import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="HR Analytics Pro | Strategic Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f4f6f9; }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.05);
        border-left: 5px solid #4F8BF9;
    }
    div[data-testid="stExpander"] {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0px 2px 5px rgba(0,0,0,0.05);
    }
    h1, h2, h3 { color: #1E3A8A; font-family: 'Segoe UI', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA LOADING & CACHING ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("dataset.csv")
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        
        # Basic Cleaning & Mapping
        df['Attrition_Binary'] = df['left_company'].apply(lambda x: 1 if x == True else 0)
        df['Attrition_Label'] = df['left_company'].apply(lambda x: 'Left' if x == True else 'Stayed')
        
        # Fill missing values just in case
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        return df
    except FileNotFoundError:
        st.error("CRITICAL ERROR: 'dataset.csv' not found. Please upload the file.")
        return None

raw_df = load_data()

# --- 2. SIDEBAR & GLOBAL FILTERS ---
st.sidebar.header("🔍 Filter & Navigate")

if raw_df is not None:
    # Navigation
    page = st.sidebar.radio("Go to", ["📊 Executive Dashboard", "🔎 Deep Dive Analytics", "🤖 Predictive Modeling"])
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Global Filters")
    st.sidebar.info("Leave filters empty to select ALL")
    
    # Get unique values for filters
    all_depts = raw_df['department'].unique()
    all_roles = raw_df['role'].unique()
    all_levels = raw_df['job_level'].unique()

    # Multiselect Filters
    selected_depts = st.sidebar.multiselect("Select Department", options=all_depts)
    selected_roles = st.sidebar.multiselect("Select Role", options=all_roles)
    selected_levels = st.sidebar.multiselect("Select Job Level", options=all_levels)
    
    # Logic: If selection is empty, use ALL
    if not selected_depts: selected_depts = all_depts
    if not selected_roles: selected_roles = all_roles
    if not selected_levels: selected_levels = all_levels

    # Filter Data based on selection
    df_filtered = raw_df[
        (raw_df['department'].isin(selected_depts)) &
        (raw_df['role'].isin(selected_roles)) &
        (raw_df['job_level'].isin(selected_levels))
    ]
    
    # Display filtered count in sidebar
    st.sidebar.markdown(f"**Active Employees:** {len(df_filtered)}")
    
    # --- PAGE 1: EXECUTIVE DASHBOARD ---
    if page == "📊 Executive Dashboard":
        st.title("📊 Executive HR Overview")
        st.markdown(f"**Scope:** {len(df_filtered)} Employees | **Departments:** {len(selected_depts)} Selected")
        
        # --- Top Row KPIs ---
        total = len(df_filtered)
        attrition_count = df_filtered['Attrition_Binary'].sum()
        attrition_rate = (attrition_count / total * 100) if total > 0 else 0
        avg_sat = df_filtered['satisfaction_score'].mean()
        avg_sal = df_filtered['salary'].mean()
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Headcount", f"{total:,}")
        kpi2.metric("Attrition Rate", f"{attrition_rate:.1f}%", delta_color="inverse")
        kpi3.metric("Avg Satisfaction", f"{avg_sat:.2f} / 1.0")
        kpi4.metric("Avg Salary", f"${avg_sal:,.0f}")
        
        st.markdown("---")
        
        # --- Row 2: Sunburst & Attrition by Group ---
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.subheader("Hierarchy of Attrition")
            # Sunburst: Department -> Role -> Attrition Label
            # Aggregate data for sunburst
            if not df_filtered.empty:
                sunburst_df = df_filtered.groupby(['department', 'role', 'Attrition_Label']).size().reset_index(name='count')
                fig_sun = px.sunburst(sunburst_df, path=['department', 'role', 'Attrition_Label'], values='count',
                                      color='Attrition_Label', color_discrete_map={'Left': '#FF4B4B', 'Stayed': '#00CC96'},
                                      title="Headcount Distribution: Dept > Role > Status")
                st.plotly_chart(fig_sun, use_container_width=True)
            else:
                st.warning("No data to display for current selection.")
            
        with c2:
            st.subheader("Attrition by Job Level")
            # Grouped Bar Chart
            if not df_filtered.empty:
                fig_bar = px.histogram(df_filtered, x="job_level", color="Attrition_Label", 
                                       barmode="group", text_auto=True,
                                       color_discrete_map={'Left': '#FF4B4B', 'Stayed': '#1F77B4'},
                                       title="Leavers vs Stayers by Job Level")
                st.plotly_chart(fig_bar, use_container_width=True)

        # --- Row 3: 3D Analysis & Tenure ---
        c3, c4 = st.columns([1.2, 0.8])
        
        with c3:
            st.subheader("3D Talent View")
            st.markdown("Satisfaction vs. Performance vs. Salary (Color: Attrition)")
            # Sample down for performance if data is huge
            if not df_filtered.empty:
                plot_df = df_filtered.sample(min(2000, len(df_filtered)))
                fig_3d = px.scatter_3d(plot_df, x='satisfaction_score', y='performance_score', z='salary',
                                       color='Attrition_Label', size='tenure_months', opacity=0.6,
                                       color_discrete_map={'Left': '#FF4B4B', 'Stayed': '#00CC96'},
                                       title="Multi-Variate Talent Analysis")
                st.plotly_chart(fig_3d, use_container_width=True)
            
        with c4:
            st.subheader("Tenure Risk Analysis")
            # Histogram of Tenure
            if not df_filtered.empty:
                fig_hist = px.histogram(df_filtered, x="tenure_months", color="Attrition_Label",
                                        nbins=20, marginal="box",
                                        color_discrete_map={'Left': '#FF4B4B', 'Stayed': '#1F77B4'},
                                        title="When do people leave?")
                st.plotly_chart(fig_hist, use_container_width=True)

    # --- PAGE 2: DEEP DIVE ANALYTICS ---
    elif page == "🔎 Deep Dive Analytics":
        st.title("🔎 Deep Dive: Root Cause Analysis")
        st.markdown("Drill down into behavioral, financial, and well-being metrics based on your **Sidebar Filters**.")
        
        # --- Tabbed Interface for Organization ---
        tab1, tab2, tab3 = st.tabs(["💸 Compensation & Workload", "🧠 Well-being & Stress", "❌ Turnover Reasons"])
        
        with tab1:
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Salary Distribution (Violin Plot)")
                if not df_filtered.empty:
                    fig_vio = px.violin(df_filtered, y="salary", x="department", color="Attrition_Label", 
                                        box=True, points="all", hover_data=df_filtered.columns,
                                        title="Salary Spread by Department & Attrition")
                    st.plotly_chart(fig_vio, use_container_width=True)
            
            with col_b:
                st.subheader("Workload vs. Overtime")
                if not df_filtered.empty:
                    fig_scat = px.scatter(df_filtered, x="workload_score", y="overtime_hours", 
                                          color="Attrition_Label", size="burnout_risk",
                                          trendline="ols", 
                                          title="Impact of Workload & Overtime on Attrition")
                    st.plotly_chart(fig_scat, use_container_width=True)

        with tab2:
            st.subheader("Stress & Burnout Analysis (KDE Plot)")
            # Create a KDE plot using Figure Factory
            if not df_filtered.empty:
                hist_data = [df_filtered[df_filtered['Attrition_Label'] == 'Stayed']['stress_level'],
                             df_filtered[df_filtered['Attrition_Label'] == 'Left']['stress_level']]
                group_labels = ['Stayed', 'Left']
                colors = ['#00CC96', '#FF4B4B']
                
                try:
                    fig_kde = ff.create_distplot(hist_data, group_labels, show_hist=False, colors=colors)
                    fig_kde.update_layout(title_text="Stress Level Density Curve")
                    st.plotly_chart(fig_kde, use_container_width=True)
                except:
                    st.warning("Not enough variance to plot density curve for current selection.")

            col_c, col_d = st.columns(2)
            with col_c:
                st.subheader("Correlation Matrix (Top Drivers)")
                if not df_filtered.empty:
                    corr_cols = ['satisfaction_score', 'stress_level', 'burnout_risk', 
                                 'salary', 'performance_score', 'Attrition_Binary']
                    corr_matrix = df_filtered[corr_cols].corr()
                    fig_heat = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
                    st.plotly_chart(fig_heat, use_container_width=True)
            
            with col_d:
                st.subheader("Team Sentiment Impact")
                if not df_filtered.empty:
                    fig_box_sent = px.box(df_filtered, x="Attrition_Label", y="team_sentiment", 
                                          color="Attrition_Label", notched=True,
                                          title="Does Team Sentiment affect staying?")
                    st.plotly_chart(fig_box_sent, use_container_width=True)

        with tab3:
            st.subheader("Why are they leaving?")
            if 'turnover_reason' in df_filtered.columns and not df_filtered.empty:
                # Filter only leavers
                leavers_df = df_filtered[df_filtered['Attrition_Binary'] == 1]
                if not leavers_df.empty:
                    reason_counts = leavers_df['turnover_reason'].value_counts().reset_index()
                    reason_counts.columns = ['Reason', 'Count']
                    
                    fig_tree = px.treemap(reason_counts, path=['Reason'], values='Count',
                                          color='Count', color_continuous_scale='Reds',
                                          title="Top Reasons for Turnover")
                    st.plotly_chart(fig_tree, use_container_width=True)
                    
                    # Risk Factor Text Analysis (Simple)
                    st.subheader("Common Risk Factors (Text Analysis)")
                    st.info("Top frequent terms found in 'Risk Factors Summary'")
                    from collections import Counter
                    text_blob = " ".join(df_filtered['risk_factors_summary'].astype(str).tolist())
                    words = [w for w in text_blob.split() if len(w) > 4] # Simple filter
                    common_words = Counter(words).most_common(10)
                    word_df = pd.DataFrame(common_words, columns=['Word', 'Count'])
                    st.bar_chart(word_df.set_index('Word'))
                else:
                    st.success("No attrition found in this selection!")

    # --- PAGE 3: PREDICTIVE MODELING ---
    elif page == "🤖 Predictive Modeling":
        st.title("🤖 AI Attrition Predictor")
        st.markdown("Train models to predict employee flight risk.")

        # --- Layout: Settings vs Results ---
        col_sets, col_main = st.columns([1, 3])

        with col_sets:
            st.subheader("🛠 Model Config")
            model_option = st.selectbox("Choose Algorithm", 
                                        ["Random Forest", "Decision Tree", "Logistic Regression", "Naive Bayes", "SVM"])
            
            split_size = st.slider("Train/Test Split", 0.1, 0.4, 0.2)
            
            st.markdown("### Data Selection")
            data_scope = st.radio("Train on:", ["All Data (Recommended)", "Filtered View"])
            
            # Use appropriate dataset
            train_df = df_filtered if data_scope == "Filtered View" else raw_df

        with col_main:
            if not train_df.empty:
                # Data Preparation
                feature_cols = ['tenure_months', 'salary', 'performance_score', 'satisfaction_score', 
                                'workload_score', 'stress_level', 'burnout_risk', 'department', 'job_level']
                
                # Preprocessing Pipeline
                X = train_df[feature_cols].copy()
                y = train_df['Attrition_Binary']
                
                # Encode
                le_dept = LabelEncoder()
                le_job = LabelEncoder()
                X['department'] = le_dept.fit_transform(X['department'].astype(str))
                X['job_level'] = le_job.fit_transform(X['job_level'].astype(str))
                
                # Fill NA
                X = X.fillna(0)
                
                # Scale (Important for SVM/LogReg)
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Split
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=split_size, random_state=42)
                
                # --- Training Function ---
                @st.cache_resource
                def train_ml_model(algo, _X_tr, _y_tr):
                    if algo == "Naive Bayes": model = GaussianNB()
                    elif algo == "Decision Tree": model = DecisionTreeClassifier(max_depth=8, random_state=42)
                    elif algo == "Random Forest": model = RandomForestClassifier(n_estimators=100, random_state=42)
                    elif algo == "Logistic Regression": model = LogisticRegression(max_iter=1000)
                    elif algo == "SVM": model = SVC(probability=True, random_state=42)
                    model.fit(_X_tr, _y_tr)
                    return model

                with st.spinner(f"Training {model_option}..."):
                    model = train_ml_model(model_option, X_train, y_train)

                # Predictions
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
                acc = accuracy_score(y_test, y_pred)

                # --- Performance Metrics ---
                st.success(f"Model Trained Successfully! Accuracy: **{acc:.2%}**")
                
                m1, m2, m3 = st.columns(3)
                cm = confusion_matrix(y_test, y_pred)
                
                with m1:
                    st.subheader("Confusion Matrix")
                    fig_cm = px.imshow(cm, text_auto=True, x=['Stay', 'Leave'], y=['Stay', 'Leave'], 
                                       labels=dict(x="Predicted", y="Actual"), color_continuous_scale='Blues')
                    st.plotly_chart(fig_cm, use_container_width=True)
                
                with m2:
                    if model_option in ["Random Forest", "Decision Tree"]:
                        st.subheader("Feature Importance")
                        imps = model.feature_importances_
                        feat_df = pd.DataFrame({'Feature': feature_cols, 'Importance': imps}).sort_values(by='Importance', ascending=True)
                        fig_imp = px.bar(feat_df, x='Importance', y='Feature', orientation='h', title="Key Drivers")
                        st.plotly_chart(fig_imp, use_container_width=True)
                    elif model_option == "Logistic Regression":
                        st.subheader("Coefficients")
                        coefs = model.coef_[0]
                        feat_df = pd.DataFrame({'Feature': feature_cols, 'Coef': coefs}).sort_values(by='Coef')
                        fig_imp = px.bar(feat_df, x='Coef', y='Feature', orientation='h', color='Coef', color_continuous_scale='RdBu_r')
                        st.plotly_chart(fig_imp, use_container_width=True)
                    else:
                        st.info("Feature importance not available for this model.")

                with m3:
                    st.subheader("ROC Curve")
                    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                    roc_auc = auc(fpr, tpr)
                    fig_roc = px.area(x=fpr, y=tpr, title=f'ROC Area Under Curve = {roc_auc:.2f}',
                                      labels=dict(x='False Positive Rate', y='True Positive Rate'))
                    fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                    st.plotly_chart(fig_roc, use_container_width=True)

            # --- Interactive Prediction ---
            st.divider()
            st.subheader("🎲 Live Simulation: Single Employee Prediction")
            
            row1 = st.columns(4)
            p_sat = row1[0].slider("Satisfaction", 0.0, 1.0, 0.5)
            p_stress = row1[1].slider("Stress", 0.0, 1.0, 0.5)
            p_burn = row1[2].slider("Burnout", 0.0, 1.0, 0.5)
            p_tenure = row1[3].number_input("Tenure (Mo)", 0, 200, 24)
            
            row2 = st.columns(4)
            p_sal = row2[0].number_input("Salary ($)", 30000, 300000, 60000)
            p_perf = row2[1].slider("Performance", 0.0, 1.0, 0.7)
            p_dept = row2[2].selectbox("Department", le_dept.classes_)
            p_job = row2[3].selectbox("Level", le_job.classes_)
            
            p_work = 0.6 # default average
            
            if st.button("Predict Risk"):
                input_data = pd.DataFrame([[p_tenure, p_sal, p_perf, p_sat, p_work, p_stress, p_burn, 
                                            le_dept.transform([p_dept])[0], le_job.transform([p_job])[0]]], columns=feature_cols)
                input_scaled = scaler.transform(input_data)
                
                prob = model.predict_proba(input_scaled)[0][1]
                
                col_res1, col_res2 = st.columns([1, 3])
                with col_res1:
                    if prob > 0.6:
                        st.error(f"🚨 High Risk: {prob:.1%}")
                    elif prob > 0.3:
                        st.warning(f"⚠️ Medium Risk: {prob:.1%}")
                    else:
                        st.success(f"✅ Safe: {prob:.1%}")
                
                with col_res2:
                    st.progress(prob)
                    st.caption("Probability of Attrition")