import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Page Configuration ---
st.set_page_config(
    page_title="Sleep Health & Social Good",
    page_icon="💤",
    layout="wide"
)

# --- 1. Data Loading & Preprocessing ---
@st.cache_data
def load_data():
    # Load the dataset
    df = pd.read_csv("All_SleepWake_viz.csv")
    
    # -- Data Cleaning & Mapping for Better Readability --
    
    # Map Anxiety levels (Assuming EQ-5D coding: 1=No problems, 2=Some, 3=Extreme)
    # We use a new column for display purposes
    anxiety_map = {1: 'No Anxiety/Depression', 2: 'Moderate', 3: 'Severe'}
    df['Anxiety_Label'] = df['eq5d_anxiety_depression'].map(anxiety_map)
    
    # Ensure categorical sorting for charts
    cat_orders = {
        'income_group': ['Low', 'Lower_middle', 'Upper_middle', 'High'],
        'education_group': ['HighSchool_or_less', 'SomeCollege_or_Associate', 'Bachelor', 'Graduate'],
        'bmi_group': ['Underweight', 'Normal', 'Overweight', 'Obese'],
        'Anxiety_Label': ['No Anxiety/Depression', 'Moderate', 'Severe']
    }
    
    return df, cat_orders

df, cat_orders = load_data()

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["1. Demographic Overview", "2. Health & Socio-Economic Factors", "3. Interactive Explorer"])

st.sidebar.markdown("---")
st.sidebar.info(
    "**Project: Social Good**\n\n"
    "Visualizing inequalities in sleep health to identify vulnerable populations."
)

# --- Page 1: Demographic Overview ---
if page == "1. Demographic Overview":
    st.title("📊 Sleep Quality by Demographics")
    st.markdown("""
    **Theme:** identifying sleep disparities across different population segments.
    Use the controls below to visualize how Sleep Quality Scores (0-100) are distributed across different demographic groups.
    """)
    
    # Control Panel
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Settings")
        group_option = st.selectbox(
            "Group By:",
            [
                ("Sex", "sex_label"),
                ("Age Group", "age_group"),
                ("Employment Status", "employment_status_main"),
                ("BMI Group", "bmi_group"),
                ("Education Level", "education_group")
            ],
            format_func=lambda x: x[0]
        )
        # Extract the column name from the tuple
        group_col = group_option[1]
        group_name = group_option[0]

        chart_type = st.radio("Chart Type:", ["Box Plot", "Violin Plot", "Histogram"])

    with col2:
        st.subheader(f"Sleep Score Distribution by {group_name}")
        
        # Plotting
        if chart_type == "Box Plot":
            fig = px.box(
                df, x=group_col, y="sleep_quality_score_0_100", color=group_col,
                title=f"Sleep Quality Score vs {group_name}",
                category_orders=cat_orders if group_col in cat_orders else {}
            )
        elif chart_type == "Violin Plot":
            fig = px.violin(
                df, x=group_col, y="sleep_quality_score_0_100", color=group_col, box=True,
                title=f"Sleep Quality Density by {group_name}",
                category_orders=cat_orders if group_col in cat_orders else {}
            )
        else: # Histogram
            fig = px.histogram(
                df, x="sleep_quality_score_0_100", color=group_col, barmode="overlay",
                title=f"Distribution of Sleep Scores by {group_name}",
                opacity=0.7,
                category_orders=cat_orders if group_col in cat_orders else {}
            )

        fig.update_layout(xaxis_title=group_name, yaxis_title="Sleep Quality Score (0-100)")
        st.plotly_chart(fig, use_container_width=True)

    # Summary Stats Table
    with st.expander("View Summary Statistics"):
        summary = df.groupby(group_col)['sleep_quality_score_0_100'].describe().reset_index()
        st.dataframe(summary)


# --- Page 2: Health & Socio-Economic Factors ---
elif page == "2. Health & Socio-Economic Factors":
    st.title("🏥 Drivers of Sleep Health")
    st.markdown("""
    **Theme:** Analyzing the impact of mental/physical health and socio-economic status on sleep quality.
    Select a factor to see its relationship with sleep quality.
    """)

    # Control Panel
    variable_options = {
        "Anxiety Level": "Anxiety_Label",
        "Global Physical Health (1-5)": "global_physical_health",
        "Global Quality of Life (1-5)": "global_quality_of_life",
        "Sleep Latency (Minutes)": "sleep_latency_minutes",
        "Income Group": "income_group",
        "Relationship Status": "relationship_group"
    }
    
    selected_label = st.selectbox("Select Factor to Analyze:", list(variable_options.keys()))
    selected_col = variable_options[selected_label]

    col1, col2 = st.columns([2, 1])

    with col1:
        # Determine plot type based on variable type
        if selected_col == "sleep_latency_minutes":
            # Scatter plot for continuous variable
            st.subheader(f"Correlation: {selected_label} vs Sleep Score")
            fig = px.scatter(
                df, x=selected_col, y="sleep_quality_score_0_100",
                trendline="ols",
                opacity=0.5,
                title=f"Sleep Score vs {selected_label}",
                labels={selected_col: selected_label, "sleep_quality_score_0_100": "Sleep Score"}
            )
        else:
            # Box plot for categorical/ordinal variables
            st.subheader(f"Distribution: {selected_label} vs Sleep Score")
            fig = px.box(
                df, x=selected_col, y="sleep_quality_score_0_100",
                color=selected_col,
                title=f"Impact of {selected_label} on Sleep Quality",
                category_orders=cat_orders,
                labels={selected_col: selected_label, "sleep_quality_score_0_100": "Sleep Score"}
            )
            # If it's a numeric ordinal (1-5), update x-axis to be categorical
            if "global" in selected_col:
                fig.update_xaxes(type='category')

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Insights")
        if selected_col == "sleep_latency_minutes":
             corr = df[[selected_col, 'sleep_quality_score_0_100']].corr().iloc[0,1]
             st.metric("Correlation Coefficient", f"{corr:.2f}")
             st.write("A negative correlation indicates that longer time to fall asleep relates to lower sleep quality.")
        else:
            # Calculate mean score per group
            avg_scores = df.groupby(selected_col)['sleep_quality_score_0_100'].mean().sort_values(ascending=False)
            st.write(f"**Average Sleep Score by {selected_label}:**")
            st.dataframe(avg_scores)

# --- Page 3: Interactive Exploration ---
elif page == "3. Interactive Explorer":
    st.title("🔍 Population Explorer")
    st.markdown("""
    **Theme:** Interactive "What-If" Analysis.
    Adjust the filters on the left to simulate different population segments (e.g., "Middle-aged females with High Income").
    Compare the selected group's sleep health against the total population.
    """)

    # --- Sidebar Filters ---
    st.sidebar.header("Filter Population")
    
    # 1. Demographics
    selected_sex = st.sidebar.multiselect("Sex", df['sex_label'].unique(), default=df['sex_label'].unique())
    
    min_age, max_age = int(df['age'].min()), int(df['age'].max())
    selected_age = st.sidebar.slider("Age Range", min_age, max_age, (min_age, max_age))
    
    # 2. Socio-Economic
    selected_income = st.sidebar.multiselect("Income Group", df['income_group'].dropna().unique(), default=df['income_group'].dropna().unique())
    selected_education = st.sidebar.multiselect("Education", df['education_group'].unique(), default=df['education_group'].unique())
    
    # 3. Health
    selected_anxiety = st.sidebar.multiselect("Anxiety Status", df['Anxiety_Label'].dropna().unique(), default=df['Anxiety_Label'].dropna().unique())
    
    # --- Filtering Data ---
    filtered_df = df[
        (df['sex_label'].isin(selected_sex)) &
        (df['age'].between(selected_age[0], selected_age[1])) &
        (df['income_group'].isin(selected_income)) &
        (df['education_group'].isin(selected_education)) &
        (df['Anxiety_Label'].isin(selected_anxiety))
    ]

    # --- Main Display ---
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    avg_total = df['sleep_quality_score_0_100'].mean()
    avg_selected = filtered_df['sleep_quality_score_0_100'].mean() if not filtered_df.empty else 0
    count_selected = len(filtered_df)
    
    col1.metric("Total Population Avg Score", f"{avg_total:.1f}")
    col2.metric("Selected Group Avg Score", f"{avg_selected:.1f}", delta=f"{avg_selected - avg_total:.1f}")
    col3.metric("Sample Size", f"{count_selected} people")

    if filtered_df.empty:
        st.warning("No data matches your filters. Please adjust your selection.")
    else:
        # Comparison Plot
        st.subheader("Distribution Comparison: Selected Group vs. All")
        
        # Create a combined dataset for plotting
        filtered_df['Group'] = 'Selected'
        df_copy = df.copy()
        df_copy['Group'] = 'All'
        
        # We only want to show the 'All' distribution as a background reference
        # An overlay histogram works best
        
        fig = go.Figure()
        
        # Add "All" trace
        fig.add_trace(go.Histogram(
            x=df['sleep_quality_score_0_100'],
            name='Total Population',
            opacity=0.5,
            marker_color='gray',
            histnorm='percent'
        ))
        
        # Add "Selected" trace
        fig.add_trace(go.Histogram(
            x=filtered_df['sleep_quality_score_0_100'],
            name='Selected Group',
            opacity=0.75,
            marker_color='blue',
            histnorm='percent'
        ))

        fig.update_layout(
            barmode='overlay',
            title="Sleep Score Distribution (Percentage)",
            xaxis_title="Sleep Quality Score",
            yaxis_title="Percentage of Group",
            legend=dict(x=0.01, y=0.99)
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Scatter of BMI vs Sleep for the selected group
        st.subheader("BMI vs Sleep Quality (Selected Group)")
        fig_scatter = px.scatter(
            filtered_df, 
            x="bmi", 
            y="sleep_quality_score_0_100", 
            color="sex_label",
            title="BMI vs Sleep Quality",
            hover_data=['age', 'employment_status_main']
        )
        st.plotly_chart(fig_scatter, use_container_width=True)