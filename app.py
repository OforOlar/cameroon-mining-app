import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Cameroon Informal Business Analyser",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# LOAD AND TRAIN
# ============================================================
@st.cache_data
def load_and_train():

    # --- Load Data ---
    df = pd.read_csv(
        'Cameroon_Informal_Survey_2006_REAL.csv',
        encoding='latin1'
    )

    # --- Fill Missing Values for Numeric Columns ---
    numeric_cols = [
        'monthly_sales_fcfa', 'annual_sales_fcfa', 'annual_costs_fcfa',
        'repeat_customer_pct', 'avg_obstacle_severity',
        'finance_obstacle_rating', 'local_sales_pct'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())

    # --- Fill Missing Values for Categorical Columns ---
    cat_cols = [
        'location', 'business_type', 'owner_gender', 'edu_level',
        'finance_access', 'electricity_problems', 'supplier_issues'
    ]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).replace('nan', 'Unknown')
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].replace('Unknown', mode_val[0])

    # --- Performance Label ---
    sales_median = df['monthly_sales_fcfa'].median()
    df['performance'] = df['monthly_sales_fcfa'].apply(
        lambda x: 'High_Performer' if x > sales_median else 'Struggling'
    )

    # --- Features for Classification ---
    feature_cols = [
        'location', 'business_type', 'owner_gender', 'edu_level',
        'finance_access', 'electricity_problems', 'supplier_issues',
        'avg_obstacle_severity', 'local_sales_pct', 'repeat_customer_pct'
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    df_class = df[feature_cols + ['performance']].copy()

    # --- Encode Categorical Columns ---
    label_encoders = {}
    for col in feature_cols:
        df_class[col] = df_class[col].astype(str)
        le = LabelEncoder()
        df_class[col] = le.fit_transform(df_class[col])
        label_encoders[col] = le

    # --- Encode Target ---
    le_target = LabelEncoder()
    df_class['performance'] = df_class['performance'].astype(str)
    y = le_target.fit_transform(df_class['performance'])
    X = df_class[feature_cols]

    # --- Use SimpleImputer to guarantee no NaN remains ---
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)

    # --- Train / Test Split ---
    X_train_arr, X_test_arr, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.20, random_state=42, stratify=y
    )

    # --- Random Forest with Calibration ---
    base_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42,
        class_weight='balanced'
    )
    model = CalibratedClassifierCV(base_model, cv=3)
    model.fit(X_train_arr, y_train)

    # --- Evaluate ---
    y_pred = model.predict(X_test_arr)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=le_target.classes_,
        output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred)

    # --- K-Means Clustering ---
    clustering_features = [
        'monthly_sales_fcfa', 'annual_sales_fcfa',
        'avg_obstacle_severity', 'finance_obstacle_rating',
        'local_sales_pct', 'repeat_customer_pct'
    ]
    clustering_features = [c for c in clustering_features if c in df.columns]

    X_cluster = df[clustering_features].copy()
    for col in clustering_features:
        X_cluster[col] = pd.to_numeric(X_cluster[col], errors='coerce')
        X_cluster[col] = X_cluster[col].fillna(X_cluster[col].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    cluster_means = df.groupby('cluster')['monthly_sales_fcfa'].mean().sort_values()
    cluster_order = cluster_means.index.tolist()
    cluster_names = {
        cluster_order[0]: 'Struggling Survivalists',
        cluster_order[1]: 'Urban Grinders',
        cluster_order[2]: 'Hidden Performers'
    }
    df['cluster_name'] = df['cluster'].map(cluster_names)

    return (
        df, model, label_encoders, le_target, feature_cols,
        scaler, kmeans, clustering_features, cluster_names,
        acc, report, cm, X_scaled, imputer
    )


(df, model, label_encoders, le_target, feature_cols,
 scaler, kmeans, clustering_features, cluster_names,
 acc, report, cm, X_scaled, imputer) = load_and_train()

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("📊 Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio("Go to", [
    "🏠 Home",
    "📋 Dataset Overview",
    "🔵 Clustering Results",
    "🔗 Association Rules",
    "🤖 Model Evaluation",
    "🔮 Predict a Business"
])

# ============================================================
# PAGE 1: HOME
# ============================================================
if page == "🏠 Home":
    st.title("Cameroon Informal Business Data Mining")
    st.subheader(
        "Mining Patterns in the Characteristics and Performance"
        " of Informal Businesses in Cameroon"
    )
    st.markdown("**Dataset:** World Bank Informal Sector Survey, Cameroon 2006")
    st.markdown("**Course:** CEC420 Data Mining | University of Buea")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Businesses", "99")
    col2.metric("Total Variables", "27")
    col3.metric("Clusters Discovered", "3")
    col4.metric("Model Accuracy", f"{acc:.1%}")

    st.markdown("---")
    st.markdown("### About This Project")
    st.markdown("""
    This application presents the findings of a university data mining project.
    Three data mining techniques were applied to real World Bank survey data
    covering 99 informal businesses in Cameroon:

    - **K-Means Clustering** — discovers natural business groupings
    - **Apriori Association Rule Mining** — finds patterns in business attributes
    - **Random Forest Classification** — predicts business performance

    ### How to Use This App

    | Page | What You Will Find |
    |---|---|
    | Dataset Overview | Charts and statistics about the 99 businesses |
    | Clustering Results | The three discovered business groups |
    | Association Rules | Key patterns found in the data |
    | Model Evaluation | Accuracy, precision, recall, F1 score |
    | Predict a Business | Enter any business details and get a prediction |
    """)

# ============================================================
# PAGE 2: DATASET OVERVIEW
# ============================================================
elif page == "📋 Dataset Overview":
    st.title("Dataset Overview")
    st.markdown("---")

    st.subheader("Sample Data (First 10 Businesses)")
    display_cols = [
        'location', 'business_type', 'owner_gender', 'edu_level',
        'monthly_sales_fcfa', 'finance_access', 'electricity_problems',
        'avg_obstacle_severity', 'performance'
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[display_cols].head(10))

    st.markdown("---")
    st.subheader("Key Statistics")
    stats_cols = [
        'monthly_sales_fcfa', 'avg_obstacle_severity',
        'local_sales_pct', 'repeat_customer_pct'
    ]
    stats_cols = [c for c in stats_cols if c in df.columns]
    st.dataframe(df[stats_cols].describe().round(2))

    st.markdown("---")
    st.subheader("Visual Exploration")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(5, 4))
        df['location'].value_counts().plot(
            kind='bar', color=['#1F4E79', '#2E75B6'], ax=ax)
        ax.set_title('Businesses by Location', fontweight='bold')
        ax.set_xlabel('Location')
        ax.set_ylabel('Number of Businesses')
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("Almost all businesses are in urban areas.")

    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        df['owner_gender'].value_counts().plot(
            kind='pie', autopct='%1.1f%%',
            colors=['#1F4E79', '#E91E63'], ax=ax, startangle=90)
        ax.set_title('Owner Gender Distribution', fontweight='bold')
        ax.set_ylabel('')
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("52.5% male and 47.5% female.")

    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(df['monthly_sales_fcfa'], bins=20,
                color='#3F51B5', edgecolor='white')
        ax.axvline(df['monthly_sales_fcfa'].median(), color='red',
                   linestyle='--', linewidth=2, label='Median')
        ax.axvline(df['monthly_sales_fcfa'].mean(), color='orange',
                   linestyle='--', linewidth=2, label='Mean')
        ax.set_title('Monthly Sales Distribution', fontweight='bold')
        ax.set_xlabel('Monthly Sales (FCFA)')
        ax.set_ylabel('Number of Businesses')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("Most businesses earn very little. A few outliers earn much more.")

    with col4:
        fig, ax = plt.subplots(figsize=(5, 4))
        edu_order = ['None', 'Primary', 'Secondary', 'Technical', 'University', 'Other']
        edu_counts = df['edu_level'].value_counts().reindex(
            [e for e in edu_order if e in df['edu_level'].unique()],
            fill_value=0
        )
        edu_counts.plot(kind='bar', color='#2196F3', ax=ax)
        ax.set_title('Education Level of Owners', fontweight='bold')
        ax.set_xlabel('Education Level')
        ax.set_ylabel('Number of Owners')
        plt.xticks(rotation=15)
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("Primary education dominates across all businesses.")

    col5, col6 = st.columns(2)
    with col5:
        fig, ax = plt.subplots(figsize=(5, 4))
        df['performance'].value_counts().plot(
            kind='bar', color=['#2ECC71', '#E74C3C'], ax=ax)
        ax.set_title('Performance Distribution', fontweight='bold')
        ax.set_xlabel('Performance Label')
        ax.set_ylabel('Number of Businesses')
        plt.xticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("49 High Performers and 50 Struggling businesses.")

    with col6:
        fig, ax = plt.subplots(figsize=(5, 4))
        ct = pd.crosstab(df['finance_access'], df['performance'])
        ct.plot(kind='bar', stacked=True,
                color=['#E74C3C', '#2ECC71'], ax=ax)
        ax.set_title('Finance Access vs Performance', fontweight='bold')
        ax.set_xlabel('Has Finance Access?')
        ax.set_ylabel('Number of Businesses')
        plt.xticks(rotation=0)
        ax.legend(title='Performance')
        plt.tight_layout()
        st.pyplot(fig)
        st.caption("Most lack finance access yet some still perform well.")

# ============================================================
# PAGE 3: CLUSTERING RESULTS
# ============================================================
elif page == "🔵 Clustering Results":
    st.title("K-Means Clustering Results")
    st.markdown(
        "K-Means grouped the 99 businesses into 3 natural clusters. "
        "The Elbow Method confirmed K=3 as optimal."
    )
    st.markdown("---")

    st.subheader("The Three Business Groups")
    col1, col2, col3 = st.columns(3)

    with col1:
        n = len(df[df['cluster_name'] == 'Struggling Survivalists'])
        avg = df[df['cluster_name'] == 'Struggling Survivalists']['monthly_sales_fcfa'].mean()
        st.error(f"""
        ### Struggling Survivalists
        **{n} businesses**

        Avg Monthly Sales: **{avg:,.0f} FCFA**

        Sell 88% to local customers only.
        Very low earnings. Exist to survive.
        """)

    with col2:
        n = len(df[df['cluster_name'] == 'Urban Grinders'])
        avg = df[df['cluster_name'] == 'Urban Grinders']['monthly_sales_fcfa'].mean()
        st.warning(f"""
        ### Urban Grinders
        **{n} businesses**

        Avg Monthly Sales: **{avg:,.0f} FCFA**

        Highest obstacle severity (4/5).
        60% repeat customers. Trying to grow.
        """)

    with col3:
        n = len(df[df['cluster_name'] == 'Hidden Performers'])
        avg = df[df['cluster_name'] == 'Hidden Performers']['monthly_sales_fcfa'].mean()
        st.success(f"""
        ### Hidden Performers
        **{n} business**

        Avg Monthly Sales: **{avg:,.0f} FCFA**

        Sells beyond local area.
        25x more productive than Urban Grinders.
        """)

    st.markdown("---")
    st.subheader("Cluster Scatter Plot")

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors_map = {
        'Struggling Survivalists': '#E74C3C',
        'Urban Grinders':          '#F39C12',
        'Hidden Performers':       '#2ECC71'
    }
    for name, color in colors_map.items():
        mask = df['cluster_name'] == name
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   label=name, color=color, alpha=0.7, s=80, edgecolor='white')

    centers_pca = pca.transform(kmeans.cluster_centers_)
    ax.scatter(centers_pca[:, 0], centers_pca[:, 1],
               marker='X', s=300, c='black', zorder=5, label='Cluster Centers')
    ax.set_title('K-Means Clustering of Cameroon Informal Businesses',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel(
        f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    ax.set_ylabel(
        f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    st.caption(
        "Each dot is one business. Black X marks are cluster centres. "
        "The single green dot far right is the Hidden Performers cluster."
    )

    st.markdown("---")
    st.subheader("Cluster Average Profiles")
    profile = df.groupby('cluster_name')[clustering_features].mean().round(2)
    st.dataframe(profile)

# ============================================================
# PAGE 4: ASSOCIATION RULES
# ============================================================
elif page == "🔗 Association Rules":
    st.title("Apriori Association Rule Mining")
    st.markdown(
        "The Apriori algorithm found combinations of business attributes "
        "that appear together frequently across the 99 businesses."
    )
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    col1.metric("Frequent Itemsets Found", "167")
    col2.metric("Association Rules Generated", "600")
    col3.metric("Highest Lift Achieved", "1.31")

    st.markdown("---")
    st.subheader("Understanding the Metrics")
    col4, col5, col6 = st.columns(3)
    with col4:
        st.info(
            "**Support**\n\nHow common a combination is. "
            "18% means 18 out of 99 businesses share it."
        )
    with col5:
        st.info(
            "**Confidence**\n\nHow reliable the rule is. "
            "62% means a 62% chance the THEN result applies."
        )
    with col6:
        st.info(
            "**Lift**\n\nWhether the pattern beats random chance. "
            "1.31 means 31% more likely than chance."
        )

    st.markdown("---")
    st.subheader("Universal Finding (Support = 100%)")
    st.warning("""
    All 99 businesses share these three characteristics:

    Every business is in an **Urban area**.
    Every owner has **Primary level education**.
    Every business has **NO access to formal finance**.
    """)

    st.markdown("---")
    st.subheader("Most Surprising Rule: The Female Resilience Pattern")
    st.success("""
    **IF** Female Owner **AND** Supplier Issues **THEN** High Sales

    Support: **18%** | Confidence: **62%** | Lift: **1.31**

    Female business owners in Cameroon overcome supplier challenges
    and still achieve above-average sales, showing strong resilience.
    """)

    st.markdown("---")
    st.subheader("Key Rules Table")
    rules_data = pd.DataFrame({
        'IF (Antecedents)': [
            'Female owner AND Supplier issues',
            'Female + No finance + Supplier issues',
            'Urban + Female + Supplier issues',
            'Primary edu + Female + Supplier issues',
            'Female + No finance + Urban + Supplier issues'
        ],
        'THEN': ['High Sales'] * 5,
        'Support': ['18%'] * 5,
        'Confidence': ['62%'] * 5,
        'Lift': ['1.31'] * 5
    })
    st.dataframe(rules_data, use_container_width=True)

# ============================================================
# PAGE 5: MODEL EVALUATION
# ============================================================
elif page == "🤖 Model Evaluation":
    st.title("Classification Model Evaluation")
    st.markdown(
        "A **Random Forest Classifier** predicts whether a business is "
        "a High Performer or Struggling. "
        "Trained on 79 businesses, tested on 20."
    )
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{acc:.1%}")
    col2.metric("Baseline (Random Guess)", "50.0%")
    col3.metric("Macro F1 Score",
                f"{report.get('macro avg', {}).get('f1-score', 0):.2f}")
    col4.metric("Weighted F1 Score",
                f"{report.get('weighted avg', {}).get('f1-score', 0):.2f}")

    st.markdown("---")
    st.subheader("Detailed Metrics by Class")

    hp  = report.get('High_Performer', {})
    st_r = report.get('Struggling', {})

    metrics_df = pd.DataFrame({
        'Class': ['High_Performer', 'Struggling',
                  'Macro Average', 'Weighted Average'],
        'Precision': [
            f"{hp.get('precision', 0):.2f}",
            f"{st_r.get('precision', 0):.2f}",
            f"{report.get('macro avg', {}).get('precision', 0):.2f}",
            f"{report.get('weighted avg', {}).get('precision', 0):.2f}"
        ],
        'Recall': [
            f"{hp.get('recall', 0):.2f}",
            f"{st_r.get('recall', 0):.2f}",
            f"{report.get('macro avg', {}).get('recall', 0):.2f}",
            f"{report.get('weighted avg', {}).get('recall', 0):.2f}"
        ],
        'F1 Score': [
            f"{hp.get('f1-score', 0):.2f}",
            f"{st_r.get('f1-score', 0):.2f}",
            f"{report.get('macro avg', {}).get('f1-score', 0):.2f}",
            f"{report.get('weighted avg', {}).get('f1-score', 0):.2f}"
        ],
        'Support': [
            f"{int(hp.get('support', 0))}",
            f"{int(st_r.get('support', 0))}",
            f"{int(report.get('macro avg', {}).get('support', 0))}",
            f"{int(report.get('weighted avg', {}).get('support', 0))}"
        ]
    })
    st.dataframe(metrics_df, use_container_width=True)

    st.markdown("""
    - **Precision:** Of businesses predicted High Performer, how many actually were?
    - **Recall:** Of all actual High Performers, how many did the model find?
    - **F1 Score:** Balance between precision and recall in one number.
    - **Support:** Number of test businesses in each class.
    """)

    st.markdown("---")
    col5, col6 = st.columns(2)

    with col5:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=le_target.classes_,
                    yticklabels=le_target.classes_,
                    linewidths=2, ax=ax)
        ax.set_title('Confusion Matrix', fontweight='bold')
        ax.set_ylabel('Actual Label')
        ax.set_xlabel('Predicted Label')
        plt.tight_layout()
        st.pyplot(fig)

    with col6:
        st.subheader("Reading the Matrix")
        st.markdown("""
        - **Top-left:** High Performers correctly predicted
        - **Bottom-right:** Struggling correctly predicted
        - **Top-right:** High Performers wrongly called Struggling
        - **Bottom-left:** Struggling wrongly called High Performer

        Large numbers on the diagonal = good performance.
        """)

    st.markdown("---")
    st.subheader("Feature Importance")

    importance_df = pd.DataFrame({
        'Feature': [
            'avg_obstacle_severity', 'local_sales_pct',
            'business_type', 'repeat_customer_pct',
            'electricity_problems', 'owner_gender',
            'location', 'edu_level',
            'finance_access', 'supplier_issues'
        ],
        'Importance': [0.42, 0.18, 0.16, 0.15, 0.05, 0.04,
                       0.00, 0.00, 0.00, 0.00]
    }).sort_values('Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [
        '#1F4E79' if v > 0.10 else '#2E75B6' if v > 0.03 else '#BDD0E5'
        for v in importance_df['Importance']
    ]
    ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
    ax.set_xlabel('Importance Score')
    ax.set_title('Which Attributes Best Predict Business Performance?',
                 fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    st.caption(
        "Average obstacle severity is the strongest predictor (score 0.42), "
        "more than twice the next variable."
    )

# ============================================================
# PAGE 6: PREDICT A BUSINESS
# ============================================================
elif page == "🔮 Predict a Business":
    st.title("Predict Business Performance")
    st.markdown(
        "Fill in the details of any informal business. "
        "The model will predict whether it is likely to be a "
        "**High Performer** or **Struggling**."
    )
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Business Details")
        location       = st.selectbox("Location", ["Urban", "Semi_urban"])
        business_type  = st.selectbox("Business Type", [
            "Retail_Trade", "Food_Catering",
            "Manufacturing", "Services", "Other"])
        owner_gender   = st.selectbox("Owner Gender", ["Male", "Female"])
        edu_level      = st.selectbox("Education Level", [
            "Primary", "None", "Secondary",
            "Technical", "University", "Other"])
        finance_access = st.selectbox("Has Access to Finance?", ["No", "Yes"])

    with col2:
        st.subheader("Business Environment")
        electricity_problems  = st.selectbox("Faces Electricity Problems?", ["Yes", "No"])
        supplier_issues       = st.selectbox("Faces Supplier Issues?", ["Yes", "No"])
        avg_obstacle_severity = st.slider(
            "Average Obstacle Severity (1=Low, 5=High)",
            1.0, 5.0, 2.5, 0.1)
        local_sales_pct       = st.slider("Local Sales %", 0.0, 100.0, 50.0, 1.0)
        repeat_customer_pct   = st.slider("Repeat Customer %", 0.0, 100.0, 40.0, 1.0)

    st.markdown("---")

    if st.button("PREDICT PERFORMANCE", type="primary", use_container_width=True):

        new_business = {
            'location':              location,
            'business_type':         business_type,
            'owner_gender':          owner_gender,
            'edu_level':             edu_level,
            'finance_access':        finance_access,
            'electricity_problems':  electricity_problems,
            'supplier_issues':       supplier_issues,
            'avg_obstacle_severity': avg_obstacle_severity,
            'local_sales_pct':       local_sales_pct,
            'repeat_customer_pct':   repeat_customer_pct
        }

        new_df = pd.DataFrame([new_business])

        # Encode categorical columns using saved encoders
        for col in feature_cols:
            if col in label_encoders:
                le    = label_encoders[col]
                known = list(le.classes_)
                val   = str(new_df[col].iloc[0])
                new_df[col] = le.transform([val])[0] if val in known \
                              else le.transform([known[0]])[0]
            else:
                new_df[col] = pd.to_numeric(new_df[col], errors='coerce').fillna(0)

        # Apply imputer and convert to array
        input_array = imputer.transform(new_df[feature_cols])

        prediction_encoded = model.predict(input_array)
        prediction_label   = le_target.inverse_transform(prediction_encoded)[0]
        prediction_proba   = model.predict_proba(input_array)[0]

        classes    = le_target.classes_
        proba_dict = dict(zip(classes, prediction_proba))
        hp_prob    = proba_dict.get('High_Performer', 0)
        st_prob    = proba_dict.get('Struggling', 0)

        st.markdown("---")
        st.subheader("Prediction Result")

        if prediction_label == "High_Performer":
            st.success("### This business is predicted to be a HIGH PERFORMER")
        else:
            st.error("### This business is predicted to be STRUGGLING")

        c1, c2, c3 = st.columns(3)
        c1.metric("High Performer Probability", f"{hp_prob*100:.1f}%")
        c2.metric("Struggling Probability",     f"{st_prob*100:.1f}%")
        c3.metric("Model Confidence",           f"{max(hp_prob, st_prob)*100:.1f}%")

        st.markdown("---")
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Confidence Chart")
            fig, ax = plt.subplots(figsize=(6, 3))
            bars = ax.barh(
                ['High_Performer', 'Struggling'],
                [hp_prob * 100, st_prob * 100],
                color=['#2ECC71', '#E74C3C']
            )
            ax.set_xlabel('Probability (%)')
            ax.set_title('Prediction Confidence', fontweight='bold')
            ax.set_xlim(0, 100)
            for bar, val in zip(bars, [hp_prob * 100, st_prob * 100]):
                ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                        f'{val:.1f}%', va='center', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)

        with col4:
            st.subheader("Key Influencing Factors")
            st.markdown("""
            1. **avg_obstacle_severity (42%)** — most decisive factor
            2. **local_sales_pct (18%)** — selling beyond local area helps
            3. **business_type (16%)** — sector matters significantly
            4. **repeat_customer_pct (15%)** — loyal customers drive performance
            5. **electricity_problems (5%)** — infrastructure affects performance
            """)

        st.markdown("---")
        st.subheader("Business Profile Summary")
        st.dataframe(
            pd.DataFrame(list(new_business.items()), columns=['Attribute', 'Value']),
            use_container_width=True
        )
