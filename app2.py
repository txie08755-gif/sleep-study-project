import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ====================================================================
# 1. PAGE SETUP AND DATA LOADING 
# ====================================================================

st.set_page_config(
    page_title="🌙 Sleep Quality & Stress Predictors",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data(file_path):
    """Loads CSV dataset and performs basic data preparation and transformation."""
    try:
        # Use the latest uploaded file
        df = pd.read_csv(file_path)
        # Clean column names
        df.columns = df.columns.str.lower().str.replace('[^a-zA-Z0-9_]', '', regex=True)
        
        # --- CRITICAL Data Transformation (Best=5 Logic) ---
        if 'global_fatigue_avg' in df.columns:
            df['global_fatigue_avg_best5'] = 6 - df['global_fatigue_avg']
        
        # --- Data Preprocessing: Group Employment Status (RE-ADDED FOR ROBUSTNESS) ---
        # Explicitly group low-frequency or non-working statuses into 'Other'
        # This overrides any partial merging in the input file, ensuring consistency.
        employment_other_groups = ['On-disability', 'Unknown', 'Retired', 'Homemaker', 'On-leave', 'other'] 
        if 'employment_status_main' in df.columns:
            # First, normalize existing "other" labels to a consistent 'Other'
            df['employment_status_main'] = df['employment_status_main'].astype(str).str.replace('other', 'Other', case=False)
            
            # Then, apply the grouping for specific labels
            df['employment_status_main'] = df['employment_status_main'].apply(
                lambda x: 'Other' if pd.notna(x) and x in employment_other_groups else x
            )
        
        # --- Data Preprocessing for Tab 3: Risk Definition (Derived Variables) ---
        df['demo_risk'] = np.where(df['age_group'].isin(['18-29', '60+']), 'Demographic Risk (Yes)', 'Demographic Risk (No)')
        
        if all(col in df.columns for col in ['global_quality_of_life', 'global_fatigue_avg']):
            df['health_risk'] = np.where(
                (df['global_quality_of_life'].isin([1, 2])) | (df['global_fatigue_avg'].isin([4, 5])), 
                'Health Risk (Yes)', 
                'Health Risk (No)'
            )
        else:
             df['health_risk'] = 'N/A'
             
        df['anxiety_label'] = df['eq5d_anxiety_depression'].astype(str).replace({'1': 'Low Anxiety', '2': 'Moderate Anxiety', '3': 'Extreme Anxiety'})
        
        return df
    except FileNotFoundError:
        st.error(f"Error: File '{file_path}' not found. Ensure the data file is correct.")
        return pd.DataFrame()

# Load data - Using the last provided file path
data_file_path = "All_SleepWake_viz_occupation_merged.csv"
df = load_data(data_file_path)

if df.empty:
    st.stop()

# ====================================================================
# 2. SIDEBAR FILTERS (Global Filters)
# ====================================================================

st.sidebar.title("🔍 Data Filters")
st.sidebar.markdown("Select demographic characteristics to filter the analysis population.")

age_options = df['age_group'].sort_values().unique()
selected_ages = st.sidebar.multiselect("Age Filter:", options=age_options, default=age_options)

sex_options = df['sex_label'].unique()
selected_sex = st.sidebar.multiselect("Sex Filter:", options=sex_options, default=sex_options)

filtered_df = df[
    df['age_group'].isin(selected_ages) &
    df['sex_label'].isin(selected_sex)
].copy() 

st.sidebar.info(f"Current analysis dataset contains **{len(filtered_df)}** records.")

# Define color map and order for consistency
color_map = {'Good': '#4CAF50', 'Fair': '#FFC107', 'Poor': '#F44336', '(?)': '#C0C0C0'}
sleep_quality_order = ['Poor', 'Fair', 'Good', '(?)']


# ====================================================================
# 3. TAB STRUCTURE
# ====================================================================

tab1, tab2, tab3 = st.tabs(["Demographic Factors", "Health Behaviors and Quality of Life", "Personalized Profile Explorer"])

# ====================================================================
# TAB 1: Demographic Factors
# ====================================================================

with tab1:
    st.title("Who Is Stressed and Sleeps Poorly? Demographic Factors")
    st.markdown("This page explores the combined impact of age, sex, employment, and income on sleep quality and anxiety levels.")
    st.header("1. Baseline Analysis: Sleep Quality and Multidimensional Demographic Distribution")
    # --- 1.1 Sunburst Chart ---
    st.subheader("1.1 Hierarchical Distribution: Age, Sex, and Sleep Quality")
    if all(col in filtered_df.columns for col in ['age_group', 'sex_label', 'sleep_quality_cat']):
        path = ['age_group', 'sex_label', 'sleep_quality_cat']
        df_sunburst = filtered_df.groupby(path).size().reset_index(name='Count')
        total_count = df_sunburst['Count'].sum()
        df_sunburst['Percentage'] = (df_sunburst['Count'] / total_count) * 100
        # 构造稳定的 id，确保百分比与节点一一对应，避免中心文字乱码
        df_sunburst['id'] = (
            df_sunburst['age_group'].astype(str) + "/" +
            df_sunburst['sex_label'].astype(str) + "/" +
            df_sunburst['sleep_quality_cat'].astype(str)
        )
        pct_map = dict(zip(df_sunburst['id'], df_sunburst['Percentage']))

        fig_sunburst = px.sunburst(
            df_sunburst,
            path=path,
            values='Count',
            color='sleep_quality_cat',
            color_discrete_map=color_map,
            title='Hierarchical Distribution of Sleep Quality (Age → Sex → Quality)',
            branchvalues="total"
        )

        if fig_sunburst.data:
            ids = fig_sunburst.data[0].ids
            text_labels = []
            for _id in ids:
                if _id == "" or _id is None:
                    pct = 100.0  # root
                else:
                    pct = pct_map.get(_id, 0.0)
                text_labels.append(f"{pct:.2f}%")

            fig_sunburst.update_traces(
                text=text_labels,
                textinfo="label+text",
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Count: %{value}<br>"
                    "Share of total: %{text}<extra></extra>"
                )
            )

        fig_sunburst.update_layout(height=600)
        st.plotly_chart(fig_sunburst, use_container_width=True)
    st.markdown("---")
    # --- 1.2 Bubble Chart ---
    st.subheader("1.2 Interactive Analysis: Impact of Employment and Income on Anxiety Levels") 
    col2_filter, col2_chart = st.columns([1, 3])
    with col2_filter: 
        if 'employment_status_main' in filtered_df.columns:
            # employment_options will now only contain 'Other' instead of 'on-leave', 'on-disability', etc.
            employment_options = sorted(filtered_df['employment_status_main'].dropna().unique().tolist())
            selected_employment = st.multiselect("Select Employment Status:", options=employment_options, default=employment_options[0:3])
        else: selected_employment = []
        if 'income_group' in filtered_df.columns:
            income_options = sorted(filtered_df['income_group'].dropna().unique().tolist())
            selected_income = st.multiselect("Select Income Group:", options=income_options, default=income_options)
        else: selected_income = []
        st.markdown("""**Anxiety Level (eq5d_anxiety_depression):**\n* 1: Low Anxiety/Depression (Best)\n* 3: Extreme Anxiety/Depression (Worst)""")
    with col2_chart:
        required_cols = ['employment_status_main', 'income_group', 'eq5d_anxiety_depression']
        if all(col in filtered_df.columns for col in required_cols):
            custom_analysis_df = filtered_df[filtered_df['employment_status_main'].isin(selected_employment) & filtered_df['income_group'].isin(selected_income)].copy()
            if custom_analysis_df.empty: st.warning("No data points found."); 
            else:
                df_grouped = custom_analysis_df.groupby(['employment_status_main', 'income_group', 'eq5d_anxiety_depression']).size().reset_index(name='Count')
                df_grouped['group_key'] = df_grouped['employment_status_main'] + "_" + df_grouped['income_group']
                group_map = {key: i * 0.1 for i, key in enumerate(df_grouped['group_key'].unique())}
                df_grouped['Y_Offset'] = df_grouped['group_key'].map(group_map)
                fig_stress_scatter = px.scatter(df_grouped, x='eq5d_anxiety_depression', y='Y_Offset', size='Count', color='income_group', symbol='employment_status_main', hover_data={'Count': True, 'Y_Offset': False, 'eq5d_anxiety_depression': False, 'employment_status_main': True, 'income_group': True}, title='Distribution of Anxiety Levels by Employment and Income (Bubble Size = Count)', labels={'eq5d_anxiety_depression': 'Anxiety Level (1=Best, 3=Worst)', 'Y_Offset': ' ', 'employment_status_main': 'Employment Status', 'income_group': 'Income Group'}, category_orders={"eq5d_anxiety_depression": [1, 2, 3]})
                max_count = df_grouped['Count'].max() if not df_grouped.empty else 1
                fig_stress_scatter.update_traces(marker=dict(sizemode='area', sizeref=2*max_count/(60**2), sizemin=4, opacity=0.7), mode='markers')
                fig_stress_scatter.update_layout(yaxis={'visible': False, 'showticklabels': False, 'range': [df_grouped['Y_Offset'].min() - 0.1, df_grouped['Y_Offset'].max() + 0.1]}, xaxis={'tickvals': [1, 2, 3], 'ticktext': ['1 (Best)', '2', '3 (Worst)'], 'range': [0.5, 3.5], 'title': 'Anxiety Level'}, height=500)
                st.plotly_chart(fig_stress_scatter, use_container_width=True)
        else: st.warning("The dataset is missing columns required for the Bubble Chart.")
    st.markdown("---")
    # --- 1.3 Line Chart ---
    st.subheader("1.3 Relationship between BMI Group and Sleep Quality")
    if all(col in filtered_df.columns for col in ['bmi_group', 'sleep_quality_cat']):
        df_bmi_sleep = filtered_df.groupby(['bmi_group', 'sleep_quality_cat']).size().reset_index(name='Count')
        df_bmi_sleep['Total_by_BMI'] = df_bmi_sleep.groupby('bmi_group')['Count'].transform('sum')
        df_bmi_sleep['Percentage'] = (df_bmi_sleep['Count'] / df_bmi_sleep['Total_by_BMI']) * 100
        bmi_order = ['Underweight', 'Normal', 'Overweight', 'Obese', '(?)']
        sleep_order_1_3 = ['Poor', 'Fair', 'Good', '(?)']
        fig_bmi_sleep = px.line(df_bmi_sleep, x='bmi_group', y='Percentage', color='sleep_quality_cat', title='Percentage of Sleep Quality Categories Across BMI Groups', category_orders={'bmi_group': bmi_order, 'sleep_quality_cat': sleep_order_1_3}, labels={'bmi_group': 'BMI Group', 'sleep_quality_cat': 'Sleep Quality Category'}, markers=True)
        fig_bmi_sleep.update_layout(yaxis_title='Proportion (%)', height=500)
        st.plotly_chart(fig_bmi_sleep, use_container_width=True)
    st.markdown("---")

# ====================================================================
# TAB 2: Health Behaviors and Quality of Life
# ====================================================================

with tab2:
    st.title("Which Internal Factors Determine Sleep Quality?")
    st.markdown("### Health Profile and Lifestyle Habits")
    # --- 2.1 Scatter Plot ---
    st.subheader("2.1 Sleep Duration vs. Mental Health Rating")
    required_cols_2_1 = ['sleep_hours', 'global_mental_health', 'sleep_quality_cat']
    if all(col in filtered_df.columns for col in required_cols_2_1):
        jitter_amount = 0.2
        np.random.seed(42)
        filtered_df['mental_health_jitter'] = filtered_df['global_mental_health'] + np.random.uniform(-jitter_amount, jitter_amount, len(filtered_df))
        fig_sleep_scatter = px.scatter(filtered_df, x='sleep_hours', y='mental_health_jitter', color='sleep_quality_cat', hover_data=['age_group', 'sex_label', 'sleep_hours', 'global_mental_health', 'sleep_quality_cat'], title='Sleep Duration vs. Mental Health Rating, Colored by Sleep Quality', labels={'sleep_hours': 'Reported Sleep Duration (Hours)', 'mental_health_jitter': 'Global Mental Health Rating (1=Poor, 5=Best)', 'sleep_quality_cat': 'Sleep Quality'}, color_discrete_map=color_map, trendline="ols" )
        fig_sleep_scatter.update_layout(yaxis={'tickvals': [1, 2, 3, 4, 5], 'ticktext': ['1 (Poor)', '2', '3', '4', '5 (Best)'], 'title': 'Global Mental Health Rating', 'range': [0.5, 5.5], 'autorange': False }, height=550)
        st.plotly_chart(fig_sleep_scatter, use_container_width=True)
    st.markdown("---")
    # --- 2.2 Radar Chart ---
    st.subheader("2.2 Subjective Health Profile Comparison")
    radar_cols_raw = ['global_physical_health', 'global_mental_health', 'global_quality_of_life']
    radar_cols_inverted = ['global_fatigue_avg_best5']
    if all(col in filtered_df.columns for col in radar_cols_raw + radar_cols_inverted + ['sleep_quality_cat']):
        df_radar = filtered_df.groupby('sleep_quality_cat')[radar_cols_raw + radar_cols_inverted].mean().reset_index()
        categories = ['Physical Health (1=Worst, 5=Best)', 'Mental Health (1=Worst, 5=Best)', 'Quality of Life (1=Worst, 5=Best)', 'Fatigue (1=Worst, 5=Best)']
        fig_radar = go.Figure()
        for index, row in df_radar.iterrows():
            values = row[radar_cols_raw + radar_cols_inverted].tolist()
            fig_radar.add_trace(go.Scatterpolar(r=values, theta=categories, fill='toself', name=row['sleep_quality_cat'], line_color=color_map.get(row['sleep_quality_cat'], '#C0C0C0')))
        fig_radar.update_layout(title='Average Health Profile by Sleep Quality Category (4 Dimensions)', polar=dict(radialaxis=dict(visible=True, range=[1, 5], tickvals=[1, 2, 3, 4, 5])), showlegend=True, height=600)
        st.plotly_chart(fig_radar, use_container_width=True)
    st.markdown("---")
    # --- 2.3 Violin Plot ---
    st.subheader("2.3 Sleep Latency Distribution Across Anxiety Levels")
    required_cols_2_3 = ['sleep_latency_minutes', 'eq5d_anxiety_depression', 'sleep_quality_cat']
    if all(col in filtered_df.columns for col in required_cols_2_3):
        anxiety_order = [1, 2, 3]
        fig_violin = px.violin(filtered_df, x='eq5d_anxiety_depression', y='sleep_latency_minutes', color='sleep_quality_cat', box=True, points='outliers', category_orders={'eq5d_anxiety_depression': anxiety_order}, color_discrete_map=color_map, title='Distribution of Sleep Latency by Anxiety Level')
        fig_violin.update_layout(xaxis_title='Anxiety Level (1=Best, 3=Worst)', yaxis_title='Sleep Latency (Minutes to fall asleep)', height=550)
        fig_violin.update_xaxes(tickvals=[1, 2, 3], ticktext=['1 (Low)', '2 (Moderate)', '3 (Extreme)'])
        st.plotly_chart(fig_violin, use_container_width=True)
    st.markdown("---")


# ====================================================================
# TAB 3: Personalized Profile Explorer (FINAL DESIGN)
# ====================================================================

with tab3:
    st.title("How Do My Personal Choices Impact My Sleep?")
    st.markdown("### Personalized Sleep Profile Explorer")
    st.markdown("""
        **Instruction:** Select up to three personal factors below. The chart will dynamically update to show the **distribution of Sleep Quality (Poor/Fair/Good)** for those specific combinations. Use this to see how your profile compares to the overall population and identify areas for potential improvement (e.g., factors that minimize the 'Poor' sleep proportion).
    """)

    # --- 3.1 Interactive Selectors ---
    
    # Define available categorical predictors for user selection
    predictor_options = [
        'employment_status_main', 'education_group', 'relationship_group', 
        'income_group', 'bmi_group'
    ]

    col_1, col_2, col_3 = st.columns(3)
    
    with col_1:
        factor_1 = st.selectbox(
            "Factor 1 (Main X-axis Grouping):",
            options=predictor_options,
            index=predictor_options.index('employment_status_main')
        )
    
    with col_2:
        # Filter options so factors cannot be the same
        filtered_options_2 = [p for p in predictor_options if p != factor_1]
        factor_2 = st.selectbox(
            "Factor 2 (Sub-Grouping):",
            options=filtered_options_2,
            index=filtered_options_2.index('education_group') if 'education_group' in filtered_options_2 else 0
        )
    
    with col_3:
        # Filter options so factors cannot be the same as 1 or 2
        filtered_options_3 = [p for p in predictor_options if p not in [factor_1, factor_2]]
        factor_3 = st.selectbox(
            "Factor 3 (Optional Filter):",
            options=['-- No Filter --'] + filtered_options_3, 
            index=0
        )
        
    # --- 3.2 Dynamic Stacked Bar Chart Generation ---
    
    factors_to_group = [factor_1, factor_2]
    
    # 1. Apply Factor 3 as a filter if selected
    df_analysis = filtered_df.copy()
    f3_label = ""
    
    if factor_3 != '-- No Filter --':
        st.markdown(f"---")
        st.subheader(f"Optional Filter: Refine results by **{factor_3.replace('_', ' ').title()}**")
        
        unique_f3_values = sorted(df_analysis[factor_3].dropna().unique().tolist())
        selected_f3_value = st.multiselect(
            f"Select specific values for {factor_3.replace('_', ' ').title()}:",
            options=unique_f3_values,
            default=unique_f3_values
        )
        
        if selected_f3_value and len(selected_f3_value) < len(unique_f3_values):
            df_analysis = df_analysis[df_analysis[factor_3].isin(selected_f3_value)].copy()
            f3_label = f" (Filtered by {factor_3.replace('_', ' ').title()})"
        
    
    required_cols_3_1 = factors_to_group + ['sleep_quality_cat']
    
    if all(col in df_analysis.columns for col in required_cols_3_1):
        
        min_group_size = 5
        
        # New Column: Combine Factor 1 and Factor 2 for the single X-axis label (for clarity and less clutter)
        df_analysis['Combined_Group'] = df_analysis[factor_1].astype(str) + " | " + df_analysis[factor_2].astype(str)
        
        # 1. Group and Count
        df_grouped = df_analysis.groupby(['Combined_Group', 'sleep_quality_cat']).size().reset_index(name='Count')
        
        # 2. Calculate Total Count for Normalization
        df_grouped['Total'] = df_grouped.groupby('Combined_Group')['Count'].transform('sum')
        df_grouped['Percentage'] = (df_grouped['Count'] / df_grouped['Total']) * 100
        
        # 3. Filter small groups for stability
        df_grouped = df_grouped[df_grouped['Total'] >= min_group_size]
        
        if df_grouped.empty:
            st.warning(f"No groups found with N >= {min_group_size} for the selected combination and current filters.")
        else:
            
            # --- 4. Create the Stacked Bar Chart ---
            title_text = f"Sleep Quality Distribution (N={df_grouped['Total'].sum()}) by Combined Profile: {factor_1.replace('_', ' ').title()} | {factor_2.replace('_', ' ').title()}" + f3_label

            fig_stacked = px.bar(
                df_grouped,
                x='Combined_Group',
                y='Percentage',
                color='sleep_quality_cat', 
                category_orders={'sleep_quality_cat': sleep_quality_order},
                color_discrete_map=color_map,
                title=title_text,
                labels={'Percentage': 'Proportion of Group (%)', 'Combined_Group': f"{factor_1.replace('_', ' ').title()} | {factor_2.replace('_', ' ').title()}"},
                hover_data={'Count': True, 'Total': True, 'Percentage': ':.2f'}
            )
            
            fig_stacked.update_layout(height=650, yaxis_tickformat=".0f", legend_title="Sleep Quality")
            fig_stacked.update_xaxes(showticklabels=True, tickangle=-45)
            st.plotly_chart(fig_stacked, use_container_width=True)
            
            st.markdown(f"""
            > **Analysis Insight:** This chart clearly shows the proportion of **Poor** sleep (Red) for specific personal profiles. Look for the combinations where the Red segment is largest—these are the highest-risk groups for poor sleep. The X-axis combines your two selected factors to show every permutation.
            """)

    else:
        st.warning(f"The dataset is missing required columns for this analysis.")
    st.markdown("---")
