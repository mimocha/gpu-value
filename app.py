import altair as alt
import streamlit as st
import plotly.express as px
from utils.data_processing import load_data, calculate_value_proposition

## ========================================================================== ##
## Configurations ##
## ========================================================================== ##

# Set page configuration
st.set_page_config(
    page_title="GPU Value Proposition Chart",
    layout="wide"
)

# Load data
# @st.cache_data
def get_gpu_data():
    return load_data("data/gpu_data.csv")
gpu_data = get_gpu_data()

## ========================================================================== ##
## Title ##
## ========================================================================== ##

st.title("GPU Value Proposition Curve")
st.write("""
How much do you value a certain GPU?
This tool helps you visualize the value proposition of different GPUs based on their price and performance.""")

## ========================================================================== ##
## Data Editor ##
## ========================================================================== ##

# Create a section for GPU data editing (in a collapsible container)
with st.expander("GPU Data Editor", expanded=False):
    st.header("GPU Data Editor")
    st.write("""
You can add and edit GPU data in the table below. Changes will be reflected in the chart automatically.
This GPU dataset is transcribed from the Gamer's Nexus RX 9070 Review, FFXIV 4K Benchmark, and combined with data automatically collected from TechPowerUp.
Some data may be missing or inaccurate, so feel free to edit it as needed.
- [GamersNexus Source](https://gamersnexus.net/gpus/incredibly-efficient-amd-rx-9070-gpu-review-benchmarks-vs-9070-xt-rtx-5070#9070-benchmarks)
- [TechPowerUp Source](https://www.techpowerup.com/gpu-specs/)
""")

    # Create a dataframe for editing
    edited_gpu_data = st.data_editor(
        gpu_data,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        key="data_editor"
    )

## ========================================================================== ##
## Variable Selection ##
## ========================================================================== ##

# Create a section for variable selection (in a collapsible container)
with st.expander("Variable Selection", expanded=False):
    st.header("Variable Selection")

    # Add price column selection
    price_column = st.radio(
        "Select price metric:",
        options=['actual_price', 'msrp'],
        format_func=lambda x: x.replace("_", " ").title(),
        horizontal=True
    )

    # Add power column selection
    power_column = st.radio(
        "Select power metric:",
        options=['tdp'],
        format_func=lambda x: x.replace("_", " ").title(),
        horizontal=True
    )

    # Define the features that can be included in the value proposition
    features = {
        'performance_avg': 'Average Price Performance (Avg FPS/Dollar)',
        'performance_bottom_1': 'Bottom 1% Price Performance (Bottom 1% FPS/Dollar)',
        'performance_bottom_01': 'Bottom 0.1% Price Performance (Bottom 0.1% FPS/Dollar)',
        'efficiency_avg': 'Average Power Efficiency (Avg FPS/Watt)',
        'efficiency_bottom_1': 'Bottom 1% Power Efficiency (Bottom 1% FPS/Watt)',
        'efficiency_bottom_01': 'Bottom 0.1% Power Efficiency (Bottom 0.1% FPS/Watt)',
        'vram': 'VRAM Pricing (GB/Dollar)',
    }

    # Create a single column for feature selection
    st.subheader("Performance & Efficiency Features")
    enabled_features = {}

    # Add checkboxes for each simple feature
    for feature, label in features.items():
        enabled_features[feature] = st.checkbox(
            f"{label}",
            value=(feature == 'performance_avg')  # Default: enable performance_avg
        )

    # Handle categorical features with premium factors
    st.subheader("Price Premium Factors")
    st.write("""
Adjustable premium factors for how much more or less you value certain features.
- Default is 100% (no change)
- 0% means you do not value GPUs with this feature at all (i.e. you would not even get it for free)
- 200% means you value GPUs with this feature twice as much (i.e. you would pay double)""")

    # Memory Type premium
    memory_type_enabled = st.checkbox("VRAM Type Premium Factor", value=False)
    enabled_features['memory_type'] = memory_type_enabled

    memory_premium_factors = {}
    if memory_type_enabled:
        unique_memory_types = edited_gpu_data['memory_type'].unique()
        for memory_type in unique_memory_types:
            memory_premium_factors[memory_type] = st.slider(
                memory_type,
                min_value=0,
                max_value=200,
                value=100,  # Default 100%
                format="%d%%"
            ) / 100.0  # Convert percentage to multiplier

    # GPU Brand premium
    gpu_brand_enabled = st.checkbox("GPU Brand Premium Factor", value=False)
    enabled_features['gpu_brand'] = gpu_brand_enabled

    gpu_premium_factors = {}
    if gpu_brand_enabled:
        unique_gpu_brands = edited_gpu_data['gpu_brand'].unique()
        for brand in unique_gpu_brands:
            gpu_premium_factors[brand] = st.slider(
                brand,
                min_value=0,
                max_value=200,
                value=100,  # Default 100%
                format="%d%%"
            ) / 100.0  # Convert percentage to multiplier

    # AIB Brand premium
    aib_brand_enabled = st.checkbox("AIB Brand Premium Factor", value=False)
    enabled_features['aib_brand'] = aib_brand_enabled

    aib_premium_factors = {}
    if aib_brand_enabled:
        unique_aib_brands = edited_gpu_data['aib_brand'].unique()
        for brand in unique_aib_brands:
            aib_premium_factors[brand] = st.slider(
                brand,
                min_value=0,
                max_value=200,
                value=100,  # Default 100%
                format="%d%%"
            ) / 100.0  # Convert percentage to multiplier

## ========================================================================== ##
## Chart ##
## ========================================================================== ##

# Calculate value proposition based on selected features
result_data = calculate_value_proposition(
    edited_gpu_data,
    enabled_features,
    price_column,
    power_column,
    memory_premium_factors,
    gpu_premium_factors,
    aib_premium_factors
)

# Create the scatter/line plot
st.header("GPU Value Proposition Curve")
value_curve = (
    alt.Chart(
        result_data,
        height=800
    )
    .mark_line(interpolate='monotone')
    .encode(
        x="price:Q",
        y="value_score:Q",
        color="gpu_name:N"
    ))
st.altair_chart(value_curve, use_container_width=True)

# TODO: Add scatterplot on top of line plot for actual price points

# Show explanation of the value proposition calculation
st.header("How the Value Score is Calculated")
st.write(f"""
The value proposition score is calculated as a linear function of the selected variables:
- All enabled variables are weighted equally
- Variables are normalized before calculation
- {price_column.replace("_", " ").title()} is used for price-dependent calculations
- Final score is expressed as a percentage (0-100%)
- Higher scores represent better value propositions
""")