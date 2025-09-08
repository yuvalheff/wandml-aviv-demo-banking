#!/usr/bin/env python3
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# App color palette for consistent styling
app_color_palette = [
    'rgba(99, 110, 250, 0.8)',   # Blue
    'rgba(239, 85, 59, 0.8)',    # Red/Orange-Red
    'rgba(0, 204, 150, 0.8)',    # Green
    'rgba(171, 99, 250, 0.8)',   # Purple
    'rgba(255, 161, 90, 0.8)',   # Orange
    'rgba(25, 211, 243, 0.8)',   # Cyan
    'rgba(255, 102, 146, 0.8)',  # Pink
    'rgba(182, 232, 128, 0.8)',  # Light Green
    'rgba(255, 151, 255, 0.8)',  # Magenta
    'rgba(254, 203, 82, 0.8)'    # Yellow
]

def apply_consistent_styling(fig):
    """Apply consistent styling to plotly figures"""
    fig.update_layout(
        height=550,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8B5CF6', size=12),
        title_font=dict(color='#7C3AED', size=16),
        xaxis=dict(
            gridcolor='rgba(139,92,246,0.2)',
            zerolinecolor='rgba(139,92,246,0.3)',
            tickfont=dict(color='#8B5CF6', size=11),
            title_font=dict(color='#7C3AED', size=12)
        ),
        yaxis=dict(
            gridcolor='rgba(139,92,246,0.2)',
            zerolinecolor='rgba(139,92,246,0.3)',
            tickfont=dict(color='#8B5CF6', size=11),
            title_font=dict(color='#7C3AED', size=12)
        ),
        legend=dict(font=dict(color='#8B5CF6', size=11))
    )
    return fig

# Load the dataset
print("Loading dataset...")
data_path = '/Users/avivnahon/ds-agent-projects/session_89600b04-b810-4506-b66d-91e28f4f611b/data/train_set.csv'
df = pd.read_csv(data_path)

print(f"Dataset shape: {df.shape}")
print(f"Column names: {list(df.columns)}")

# Map target values to be more interpretable
target_col = 'target'
df[target_col] = df[target_col].map({1: 'yes', 0: 'no'})

# Identify categorical and numerical columns
categorical_cols = []
numerical_cols = []

# Check each column type and classify appropriately
for col in df.columns:
    if col != target_col:
        if df[col].dtype == 'object' or df[col].nunique() <= 10:
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)

print(f"Categorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# 1. Target Distribution Analysis
print("Creating target distribution plot...")
target_counts = df[target_col].value_counts()
fig = px.bar(x=target_counts.index, y=target_counts.values, 
             labels={'x': 'Target Variable', 'y': 'Count'},
             color=target_counts.index,
             color_discrete_sequence=app_color_palette[:2])

fig = apply_consistent_styling(fig)
fig.write_html("/Users/avivnahon/ds-agent-projects/session_89600b04-b810-4506-b66d-91e28f4f611b/research/plots/target_distribution.html", 
               include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})

# 2. Data Quality Analysis Plot
print("Creating data quality analysis...")
missing_values = df.isnull().sum()
data_types_df = pd.DataFrame({
    'Column': df.columns,
    'Data_Type': df.dtypes.astype(str),
    'Missing_Values': missing_values.values,
    'Unique_Values': [df[col].nunique() for col in df.columns]
})

fig = go.Figure(data=[
    go.Bar(name='Unique Values', x=data_types_df['Column'], y=data_types_df['Unique_Values'],
           marker_color=app_color_palette[0]),
    go.Bar(name='Missing Values', x=data_types_df['Column'], y=data_types_df['Missing_Values'],
           marker_color=app_color_palette[1])
])

fig.update_layout(
    xaxis_title="Features",
    yaxis_title="Count",
    barmode='overlay'
)
fig = apply_consistent_styling(fig)
fig.write_html("/Users/avivnahon/ds-agent-projects/session_89600b04-b810-4506-b66d-91e28f4f611b/research/plots/data_quality_overview.html", 
               include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})

# 3. Categorical Features Analysis
print("Creating categorical features analysis...")
if categorical_cols:
    # Get the first few categorical columns for visualization
    selected_cats = categorical_cols[:6] if len(categorical_cols) > 6 else categorical_cols
    
    n_features = len(selected_cats)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=selected_cats,
        specs=[[{"type": "bar"}] * n_cols for _ in range(n_rows)]
    )
    
    for i, col in enumerate(selected_cats):
        row = (i // n_cols) + 1
        col_pos = (i % n_cols) + 1
        
        value_counts = df[col].value_counts().head(10)  # Top 10 values
        
        fig.add_trace(
            go.Bar(
                x=value_counts.index,
                y=value_counts.values,
                name=col,
                marker_color=app_color_palette[i % len(app_color_palette)],
                showlegend=False
            ),
            row=row, col=col_pos
        )
    
    fig.update_layout(
        height=550,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8B5CF6', size=10),
        title_font=dict(color='#7C3AED', size=14)
    )
    
    fig.update_xaxes(
        gridcolor='rgba(139,92,246,0.2)',
        tickfont=dict(color='#8B5CF6', size=8),
        title_font=dict(color='#7C3AED', size=10)
    )
    
    fig.update_yaxes(
        gridcolor='rgba(139,92,246,0.2)',
        tickfont=dict(color='#8B5CF6', size=9),
        title_font=dict(color='#7C3AED', size=10)
    )
    
    fig.write_html("/Users/avivnahon/ds-agent-projects/session_89600b04-b810-4506-b66d-91e28f4f611b/research/plots/categorical_features_distribution.html", 
                   include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})

# 4. Numerical Features Distribution
print("Creating numerical features analysis...")
if numerical_cols:
    n_features = len(numerical_cols)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=numerical_cols,
        specs=[[{"type": "histogram"}] * n_cols for _ in range(n_rows)]
    )
    
    for i, col in enumerate(numerical_cols):
        row = (i // n_cols) + 1
        col_pos = (i % n_cols) + 1
        
        fig.add_trace(
            go.Histogram(
                x=df[col],
                name=col,
                marker_color=app_color_palette[i % len(app_color_palette)],
                showlegend=False,
                nbinsx=30
            ),
            row=row, col=col_pos
        )
    
    fig.update_layout(
        height=550,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8B5CF6', size=10),
        title_font=dict(color='#7C3AED', size=14)
    )
    
    fig.update_xaxes(
        gridcolor='rgba(139,92,246,0.2)',
        tickfont=dict(color='#8B5CF6', size=9),
        title_font=dict(color='#7C3AED', size=10)
    )
    
    fig.update_yaxes(
        gridcolor='rgba(139,92,246,0.2)',
        tickfont=dict(color='#8B5CF6', size=9),
        title_font=dict(color='#7C3AED', size=10)
    )
    
    fig.write_html("/Users/avivnahon/ds-agent-projects/session_89600b04-b810-4506-b66d-91e28f4f611b/research/plots/numerical_features_distribution.html", 
                   include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})

# 5. Correlation Analysis
print("Creating correlation analysis...")
if len(numerical_cols) > 1:
    df_numeric = df[numerical_cols + [target_col]].copy()
    df_numeric[target_col] = df_numeric[target_col].map({'yes': 1, 'no': 0})
    
    correlation_matrix = df_numeric.corr()
    
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        labels=dict(color="Correlation")
    )
    
    fig = apply_consistent_styling(fig)
    fig.write_html("/Users/avivnahon/ds-agent-projects/session_89600b04-b810-4506-b66d-91e28f4f611b/research/plots/correlation_matrix.html", 
                   include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})

# 6. Categorical vs Target Analysis
print("Creating categorical vs target analysis...")
if categorical_cols:
    selected_categorical = categorical_cols[:6]  # Select first 6 for readability
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=selected_categorical,
        specs=[[{"type": "bar"}] * 3 for _ in range(2)]
    )
    
    for i, col in enumerate(selected_categorical):
        if i >= 6:  # Limit to 6 subplots
            break
            
        row = (i // 3) + 1
        col_pos = (i % 3) + 1
        
        # Calculate percentage of positive class for each category
        category_target = df.groupby(col)[target_col].apply(lambda x: (x == 'yes').mean() * 100)
        
        fig.add_trace(
            go.Bar(
                x=category_target.index,
                y=category_target.values,
                name=col,
                marker_color=app_color_palette[i % len(app_color_palette)],
                showlegend=False
            ),
            row=row, col=col_pos
        )
    
    fig.update_layout(
        height=550,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8B5CF6', size=10),
        title_font=dict(color='#7C3AED', size=14)
    )
    
    fig.update_xaxes(
        gridcolor='rgba(139,92,246,0.2)',
        tickfont=dict(color='#8B5CF6', size=8),
        title_font=dict(color='#7C3AED', size=10)
    )
    
    fig.update_yaxes(
        gridcolor='rgba(139,92,246,0.2)',
        tickfont=dict(color='#8B5CF6', size=9),
        title_font=dict(color='#7C3AED', size=10),
        title_text="Positive Rate (%)"
    )
    
    fig.write_html("/Users/avivnahon/ds-agent-projects/session_89600b04-b810-4506-b66d-91e28f4f611b/research/plots/categorical_vs_target.html", 
                   include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})

# 7. Numerical vs Target Analysis
print("Creating numerical vs target analysis...")
if numerical_cols:
    n_features = len(numerical_cols)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=numerical_cols,
        specs=[[{"type": "box"}] * n_cols for _ in range(n_rows)]
    )
    
    for i, col in enumerate(numerical_cols):
        row = (i // n_cols) + 1
        col_pos = (i % n_cols) + 1
        
        for j, target_val in enumerate(['no', 'yes']):
            fig.add_trace(
                go.Box(
                    y=df[df[target_col] == target_val][col],
                    name=f'{target_val}',
                    marker_color=app_color_palette[j],
                    showlegend=(i == 0),
                    legendgroup=target_val,
                    boxpoints='outliers'
                ),
                row=row, col=col_pos
            )
    
    fig.update_layout(
        height=550,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#8B5CF6', size=10),
        title_font=dict(color='#7C3AED', size=14),
        legend=dict(font=dict(color='#8B5CF6', size=11))
    )
    
    fig.update_xaxes(
        gridcolor='rgba(139,92,246,0.2)',
        tickfont=dict(color='#8B5CF6', size=9),
        title_font=dict(color='#7C3AED', size=10)
    )
    
    fig.update_yaxes(
        gridcolor='rgba(139,92,246,0.2)',
        tickfont=dict(color='#8B5CF6', size=9),
        title_font=dict(color='#7C3AED', size=10)
    )
    
    fig.write_html("/Users/avivnahon/ds-agent-projects/session_89600b04-b810-4506-b66d-91e28f4f611b/research/plots/numerical_vs_target.html", 
                   include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})

print("EDA analysis completed successfully!")

# Generate summary statistics for the report
print(f"\nDataset Summary:")
print(f"- Shape: {df.shape}")
print(f"- Target distribution: {df[target_col].value_counts()}")
print(f"- Missing values: {df.isnull().sum().sum()}")
print(f"- Categorical features: {len(categorical_cols)}")
print(f"- Numerical features: {len(numerical_cols)}")

if numerical_cols:
    print(f"- Numerical features statistics:")
    print(df[numerical_cols].describe())