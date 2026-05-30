"""
Course: Python Programming (Summer 2026)
Project 1 - Data visualization.
Name: Thomas Netta
File: app.py
Data source: us_monthly_electricity.csv 
"""

import sys
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Configure page layout and geometry
st.set_page_config(page_title="US State Energy Profile", layout="wide")

# Apply custom CSS to ensure a uniform dark theme layout
st.markdown(
    """
    <style>
        stApp { background-color: #0E1117; color: #FAFAFA; }
        .stMetric { background-color: #161B22; border: 1px solid #30363D; padding: 15px; border-radius: 8px; }
        div[data-testid="stExpander"] { background-color: #161B22; border: 1px solid #30363D; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================
# 1. CORE DATA INGESTION PIPELINE
# ==========================================
@st.cache_data
def load_production_data():
    try:
        df_raw = pd.read_csv("us_monthly_electricity.csv")

        # Standardize dates using mixed format detection to support various schemas
        df_raw["Date"] = pd.to_datetime(df_raw["Date"], format="mixed")
        df_raw["Year"] = df_raw["Date"].dt.year

        # Drop rows with missing generation recordings
        df_raw = df_raw.dropna(subset=["Value"])

        return df_raw

    except FileNotFoundError:
        st.error(
            "CRITICAL ERROR: 'us_monthly_electricity.csv' not detected in the current directory."
        )
        st.info("Ensure the CSV file is located in the same directory as app.py")
        st.stop()
    except Exception as e:
        st.error(f"Pipeline Ingestion Error: {str(e)}")
        st.stop()


# Run core ingestion
df = load_production_data()

# EXPLICIT FILTER: Whitelist only individual, non-overlapping fuel categories.
target_variables = [
    "Coal", 
    "Gas", 
    "Hydro", 
    "Solar", 
    "Wind", 
    "Nuclear", 
    "Bioenergy", 
    "Other Fossil", 
    "Other Renewables"
]

# EXPLICIT COLOR MAP: Enforces a strict visual color standard across all tabs
color_mapping = {
    "Coal": "#807777",          # Dark Grey
    "Gas": "#E69F00",           # Orange
    "Hydro": "#0072B2",         # Blue
    "Solar": "#F0E442",         # Yellow
    "Wind": "#009E73",          # Green
    "Nuclear": "#CC79A7",       # Purple/Pink
    "Bioenergy": "#D55E00",     # Dark Red
    "Other Fossil": "#56B4E9",  # Light Blue
    "Other Renewables": "#999999" # Light Grey
}


# ==========================================
# 2. HIGH-PERFORMANCE MATHEMATICAL CACHING
# ==========================================
@st.cache_data
def get_bar_matrix(dataframe, year, states, allowed_vars):
    """Filters and calculates snapshot percentages for the bar chart using exact variables."""
    df_filtered = dataframe[
        (dataframe["Year"] == year)
        & (dataframe["State type"] == "state")
        & (dataframe["State"].isin(states))
        & (dataframe["Category"] == "Electricity generation")
        & (dataframe["Unit"] == "GWh")
        & (dataframe["Variable"].isin(allowed_vars))
    ]
    if df_filtered.empty:
        return pd.DataFrame()

    matrix = df_filtered.groupby(["State", "Variable"])["Value"].sum().reset_index()
    totals = matrix.groupby("State")["Value"].transform("sum")
    
    # Robust Zero-Division Guardrail
    matrix["Share_Pct"] = np.where(totals > 0, (matrix["Value"] / totals) * 100, 0.0)
    
    # Explicit sorting strategy for cleaner visual structure
    matrix = matrix.sort_values(by=["State", "Share_Pct"], ascending=[True, True])
    return matrix


@st.cache_data
def get_line_matrix(dataframe, focus_state, allowed_vars):
    """Filters and calculates historical trend percentages for the line chart using exact variables."""
    df_filtered = dataframe[
        (dataframe["State"] == focus_state)
        & (dataframe["State type"] == "state")
        & (dataframe["Category"] == "Electricity generation")
        & (dataframe["Unit"] == "GWh")
        & (dataframe["Variable"].isin(allowed_vars))
    ]
    if df_filtered.empty:
        return pd.DataFrame()

    matrix = df_filtered.groupby(["Year", "Variable"])["Value"].sum().reset_index()
    totals = matrix.groupby("Year")["Value"].transform("sum")
    
    # Robust Zero-Division Guardrail
    matrix["Share_Pct"] = np.where(totals > 0, (matrix["Value"] / totals) * 100, 0.0)
    return matrix


@st.cache_data
def get_national_matrix(dataframe, year, allowed_vars):
    """Generates matrix for all available states for a targeted year."""
    df_filtered = dataframe[
        (dataframe["Year"] == year)
        & (dataframe["State type"] == "state")
        & (dataframe["Category"] == "Electricity generation")
        & (dataframe["Unit"] == "GWh")
        & (dataframe["Variable"].isin(allowed_vars))
    ]
    if df_filtered.empty:
        return pd.DataFrame()

    matrix = df_filtered.groupby(["State", "Variable"])["Value"].sum().reset_index()
    totals = matrix.groupby("State")["Value"].transform("sum")
    matrix["Share_Pct"] = np.where(totals > 0, (matrix["Value"] / totals) * 100, 0.0)
    
    pivot_df = matrix.pivot(index="State", columns="Variable", values="Share_Pct").fillna(0.0)
    return pivot_df.reset_index()


# ==========================================
# 3. USER INTERFACE DISPLAY & CONTROLS
# ==========================================
st.title("Methodological Framework")
st.subheader("US State-Level Electricity Generation Mix Pipeline")
st.markdown("---")

min_year = int(df["Year"].min())
max_year = int(df["Year"].max())
all_states = sorted(df[df["State type"] == "state"]["State"].unique())

# Setup dynamic defaults
default_selection = [s for s in ["New Jersey", "New York", "California", "Texas"] if s in all_states]
if not default_selection and all_states:
    default_selection = all_states[:4]

# MINIMIZABLE CONTROL PANEL (Using st.expander)
with st.expander("🛠️ Control Panel (Click to Expand/Collapse)", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### **Graph Scope Configuration**")
        selected_year = st.slider(
            "Select Target Snapshot Year",
            min_value=min_year,
            max_value=max_year,
            value=max_year,
        )
        selected_states = st.multiselect(
            "Active Comparison States (Bar Graph)", default=default_selection, options=all_states
        )
        
    with col2:
        st.markdown("### **Trend Focus Options**")
        selected_focus_state = st.selectbox(
            "Select Focus State for Trend Tracking (Line Graph)", options=all_states, index=0
        )

st.markdown("---")

# Tab Layout Setup
tab1, tab2, tab3 = st.tabs([
    "📊 Multi-State Comparison", 
    "📈 Historical Trend Profile", 
    "📋 National Matrix Overview"
])

# Global style fonts optimized for big screenshots
font_config = dict(size=18, color="#FAFAFA")

# ------------------------------------------
# TAB 1: MULTI-STATE COMPARISON (BAR GRAPH)
# ------------------------------------------
with tab1:
    st.markdown(f"#### **Cross-Sectional Generation Mix Evaluation ({selected_year})**")

    bar_matrix = get_bar_matrix(df, selected_year, selected_states, target_variables)

    if not bar_matrix.empty:
        fig_bar = px.bar(
            bar_matrix,
            x="Share_Pct",
            y="State",
            color="Variable",
            orientation="h",
            labels={
                "Share_Pct": "Percentage of Total Generation (%)",
                "State": "",
            },
            color_discrete_map=color_mapping,
            text_auto=".1f",
            template="plotly_dark",
        )

        fig_bar.update_layout(
            height=650,  # Scaled up graph frame size for clear presentation
            xaxis_range=[0, 100],
            barmode="stack",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend_title_text="Energy Type",
            xaxis=dict(title=dict(font=font_config), tickfont=dict(size=14, color="#FAFAFA")),
            yaxis=dict(title=dict(font=font_config), tickfont=dict(size=14, color="#FAFAFA")),
            legend=dict(
                font=dict(size=13),
                title_font=dict(size=14),
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.02,
            ),
        )

        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": True})

        with st.expander("View Snapshot Aggregation Data Matrix"):
            table_bar = (
                bar_matrix.pivot(index="State", columns="Variable", values="Share_Pct")
                .fillna(0)
                .reset_index()
            )
            st.dataframe(
                table_bar.style.format(
                    subset=[c for c in table_bar.columns if c != "State"],
                    formatter="{:.1f}%",
                ),
                hide_index=True,
                use_container_width=True,
            )
    else:
        st.warning("No matching electricity generation metrics found for the current filter scope.")

# ------------------------------------------
# TAB 2: HISTORICAL TREND PROFILE (LINE GRAPH)
# ------------------------------------------
with tab2:
    st.markdown(f"#### **Historical Composition Trajectory: {selected_focus_state} ({min_year} - {max_year})**")

    line_matrix = get_line_matrix(df, selected_focus_state, target_variables)

    if not line_matrix.empty:
        fig_line = px.line(
            line_matrix,
            x="Year",
            y="Share_Pct",
            color="Variable",
            labels={
                "Share_Pct": "Percentage of Total Generation (%)",
                "Year": "Calendar Year",
            },
            color_discrete_map=color_mapping,
            markers=True,
            template="plotly_dark",
        )

        fig_line.update_layout(
            height=650,  # Scaled up graph frame size for clear presentation
            yaxis_range=[0, 100],
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend_title_text="Energy Type",
            xaxis=dict(title=dict(font=font_config), tickfont=dict(size=14, color="#FAFAFA")),
            yaxis=dict(title=dict(font=font_config), tickfont=dict(size=14, color="#FAFAFA")),
            legend=dict(
                font=dict(size=13),
                title_font=dict(size=14),
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.02,
            ),
        )

        st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar": True})

        with st.expander("View Historical Trend Data Matrix"):
            table_line = (
                line_matrix.pivot(index="Year", columns="Variable", values="Share_Pct")
                .fillna(0)
                .reset_index()
            )
            st.dataframe(
                table_line.style.format(
                    subset=[c for c in table_line.columns if c != "Year"],
                    formatter="{:.1f}%",
                ),
                hide_index=True,
                use_container_width=True,
            )
    else:
        st.warning(f"No historical tracking metrics available for {selected_focus_state}.")

# ------------------------------------------
# TAB 3: NATIONAL MATRIX OVERVIEW (50-STATE COMPREHENSIVE TABLE)
# ------------------------------------------
with tab3:
    st.markdown("#### **Comprehensive 50-State Generation Matrix Profile**")
    
    # Isolated dropdown selector specifically for evaluating the complete tabular matrix
    table_year = st.selectbox(
        "Select Target Year for National Dataset Evaluation:",
        options=sorted(df["Year"].unique(), reverse=True),
        key="national_table_year_select"
    )
    
    national_df = get_national_matrix(df, table_year, target_variables)
    
    if not national_df.empty:
        st.markdown(f"Showing performance calculations across all **{len(national_df)}** recorded states/territories for calendar year **{table_year}**:")
        
        # Stylize columns dynamically
        formatted_national = national_df.style.format(
            subset=[c for c in national_df.columns if c != "State"],
            formatter="{:.2f}%"
        )
        
        st.dataframe(
            formatted_national,
            hide_index=True,
            use_container_width=True,
            height=550 # Fixed tall frame layout to display all 50 states neatly without crowding
        )
    else:
        st.warning(f"No data recordings processed for the selected timeline segment ({table_year}).")