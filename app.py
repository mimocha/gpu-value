import streamlit as st
import plotly.express as px
import pandas as pd

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
with st.expander("Feature Selection", expanded=False):
    st.header("Feature Selection")

    # Define the base features that can be included in the value proposition
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
    st.write("""
Select the features you want to include in the value proposition calculation, then assign weights for how important the features are.
When multiple features are selected, the value proposition score is calculated using a weighted average.
""")
    enabled_features = {}
    feature_weights = {}

    # Iterate over each feature and create a checkbox for selection
    total_weight = 0
    for feature, label in features.items():
        enabled_features[feature] = st.checkbox(
            label,
            value=(feature == 'performance_avg')  # Default: enable performance_avg
        )

        # Feature weight input
        if enabled_features[feature]:
            # Default weight: divide 100% evenly among enabled features
            default_weight = 100 if sum(enabled_features.values()) == 1 else int(100 / sum(enabled_features.values()))
            # For the first enabled feature, adjust to make total 100%
            if feature == list(filter(lambda f: enabled_features[f], enabled_features.keys()))[0]:
                remainder = 100 - default_weight * (sum(enabled_features.values()) - 1)
                default_weight = remainder if remainder > 0 else default_weight

            weight = st.number_input(
                f"Weight for {feature}",
                min_value=1,
                max_value=100,
                value=default_weight,
                label_visibility="collapsed"
            )
            feature_weights[feature] = weight
            total_weight += weight

    # Validate that weights sum to 100%
    valid_weights = (total_weight == 100)
    if not valid_weights and sum(enabled_features.values()) > 0:
        st.error(f"⚠️ Feature weights must add up to 100% (current total: {total_weight}%)")

    # Handle categorical features with premium factors
    st.subheader("Price Premium Factors")
    st.write("""
Adjustable premium factors for how much more or less you value certain features.
- Default is 100% (no effects)
- 200% means you value GPUs with this feature twice as much (i.e. you would pay double for this feature)
- Conversely, 50% means you would want a discount of 50% to consider this GPU over an equivalent GPU without this feature
- Hint: Unselecting a feature and re-selecting it will reset the sliders across each category back to 100%
- Hint: Adding new categories into the data editor above will automatically add new sliders here
""")

    # Memory Type premium
    memory_type_enabled = st.checkbox("VRAM Type Premium Factor", value=False)
    enabled_features['memory_type'] = memory_type_enabled

    memory_premium_factors = {}
    if memory_type_enabled:
        unique_memory_types = edited_gpu_data['memory_type'].unique()
        for memory_type in unique_memory_types:
            memory_premium_factors[memory_type] = st.slider(
                memory_type,
                key=f"memory_{memory_type}",
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
                key=f"gpu_{brand}",
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
                key=f"aib_{brand}",
                min_value=0,
                max_value=200,
                value=100,  # Default 100%
                format="%d%%"
            ) / 100.0  # Convert percentage to multiplier

## ========================================================================== ##
## Plotting ##
## ========================================================================== ##

# Calculate value proposition based on selected features
result_data = calculate_value_proposition(
    edited_gpu_data,
    enabled_features,
    memory_premium_factors,
    gpu_premium_factors,
    aib_premium_factors,
    feature_weights
)

# Add GPU brand information to result data for coloring
gpu_to_brand = edited_gpu_data[['gpu_name', 'gpu_brand']].set_index('gpu_name').to_dict()['gpu_brand']
result_data['gpu_brand'] = result_data['gpu_name'].map(gpu_to_brand)

# Define color scheme for GPU brands
color_map = {
    'NVIDIA': 'green',
    'AMD': 'red',
    'Intel': 'blue'
}

# Create the line plot for curve data
st.header("GPU Value Proposition Curve")

# Split data for curve and price points
curve_data = result_data[result_data['price_type'] == 'curve']
msrp_data = result_data[result_data['price_type'] == 'msrp']
actual_data = result_data[result_data['price_type'] == 'actual']

# Create base value curve plot
value_curve = px.line(
    curve_data,
    x="price",
    y="value_score",
    color="gpu_brand",
    line_group="gpu_name",  # Group lines by GPU name
    hover_name="gpu_name",   # Show GPU name on hover
    height=800,
    line_shape="spline",
    color_discrete_map=color_map
)

# Add triangular scatter plot marker for MSRP prices
if not msrp_data.empty:
    msrp_scatter = px.scatter(
        msrp_data,
        x="price",
        y="value_score",
        color="gpu_brand",
        hover_name="gpu_name",
        color_discrete_map=color_map
    )

    # Update marker properties for MSRP (triangles)
    for trace in msrp_scatter.data:
        trace.marker.symbol = "triangle-up"
        trace.marker.size = 10
        trace.name = f"{trace.name} (MSRP)"
        value_curve.add_trace(trace)

# Add round scatter plot marker for actual prices
if not actual_data.empty:
    actual_scatter = px.scatter(
        actual_data,
        x="price",
        y="value_score",
        color="gpu_brand",
        hover_name="gpu_name",
        color_discrete_map=color_map
    )

    # Update marker properties for actual prices (circles)
    for trace in actual_scatter.data:
        trace.marker.size = 10
        trace.name = f"{trace.name} (Actual)"
        value_curve.add_trace(trace)

# Display the combined plot
st.plotly_chart(value_curve, use_container_width=True)

## ========================================================================== ##
## Ranking Table ##
## ========================================================================== ##

# Create a section for GPU value ranking table (in a collapsible container)
with st.expander("GPU Value Ranking", expanded=False):
    st.header("GPU Value Ranking")

    # Create a DataFrame for the table with unique GPUs and their best value scores
    if not result_data.empty:
        # Get the unique GPUs and their highest value scores
        gpu_table_data = []
        for gpu_name in result_data['gpu_name'].unique():
            gpu_data = result_data[result_data['gpu_name'] == gpu_name]

            # Get MSRP and actual prices for this GPU
            msrp_row = gpu_data[gpu_data['price_type'] == 'msrp']
            msrp_price = msrp_row['price'].values[0] if not msrp_row.empty else None
            msrp_score = msrp_row['value_score'].values[0] if not msrp_row.empty else None

            actual_row = gpu_data[gpu_data['price_type'] == 'actual']
            actual_price = actual_row['price'].values[0] if not actual_row.empty else None
            actual_score = actual_row['value_score'].values[0] if not actual_row.empty else None

            # Get the highest value score and its corresponding price
            curve_data = gpu_data[gpu_data['price_type'] == 'curve']
            if not curve_data.empty:
                best_value_row = curve_data.loc[curve_data['value_score'].idxmax()]
                best_value_score = best_value_row['value_score']
            else:
                # If no curve data, use the highest of MSRP or actual price scores
                if msrp_score is not None and actual_score is not None:
                    if msrp_score >= actual_score:
                        best_value_score = msrp_score
                    else:
                        best_value_score = actual_score
                elif msrp_score is not None:
                    best_value_score = msrp_score
                elif actual_score is not None:
                    best_value_score = actual_score
                else:
                    best_value_score = None

            # Get the original GPU data
            original_gpu = edited_gpu_data[edited_gpu_data['gpu_name'] == gpu_name].iloc[0]

            # Calculate performance per dollar metrics (use actual price if available, otherwise MSRP)
            price_for_calc = actual_price if actual_price is not None else msrp_price

            # Calculate FPS per dollar metrics
            avg_fps_per_dollar = original_gpu['avg_fps'] / price_for_calc if price_for_calc else None
            bottom_1_fps_per_dollar = original_gpu['bottom_1_fps'] / price_for_calc if price_for_calc else None
            bottom_01_fps_per_dollar = original_gpu['bottom_01_fps'] / price_for_calc if price_for_calc else None

            # Calculate FPS per watt metrics
            avg_fps_per_watt = original_gpu['avg_fps'] / original_gpu['tdp'] if original_gpu['tdp'] else None
            bottom_1_fps_per_watt = original_gpu['bottom_1_fps'] / original_gpu['tdp'] if original_gpu['tdp'] else None
            bottom_01_fps_per_watt = original_gpu['bottom_01_fps'] / original_gpu['tdp'] if original_gpu['tdp'] else None

            # Calculate VRAM per dollar
            vram_per_dollar = original_gpu['vram_amount'] / price_for_calc if price_for_calc else None

            # Add to table data
            gpu_table_data.append({
                'GPU': gpu_name,
                'Value Score': round(best_value_score, 1) if best_value_score is not None else None,
                'Actual Price ($)': actual_price,
                'MSRP ($)': msrp_price,
                'Avg FPS': original_gpu['avg_fps'],
                'Bottom 1% FPS': original_gpu['bottom_1_fps'],
                'Bottom 0.1% FPS': original_gpu['bottom_01_fps'],
                'Avg FPS/Dollar': round(avg_fps_per_dollar, 3) if avg_fps_per_dollar else None,
                'Bottom 1% FPS/Dollar': round(bottom_1_fps_per_dollar, 3) if bottom_1_fps_per_dollar else None,
                'Bottom 0.1% FPS/Dollar': round(bottom_01_fps_per_dollar, 3) if bottom_01_fps_per_dollar else None,
                'Avg FPS/Watt': round(avg_fps_per_watt, 3) if avg_fps_per_watt else None,
                'Bottom 1% FPS/Watt': round(bottom_1_fps_per_watt, 3) if bottom_1_fps_per_watt else None,
                'Bottom 0.1% FPS/Watt': round(bottom_01_fps_per_watt, 3) if bottom_01_fps_per_watt else None,
                'VRAM (GB)': original_gpu['vram_amount'],
                'VRAM (GB)/Dollar': round(vram_per_dollar, 3) if vram_per_dollar else None
            })

        # Create DataFrame and sort by value score
        gpu_table_df = pd.DataFrame(gpu_table_data)
        gpu_table_df = gpu_table_df.sort_values(by='Value Score', ascending=False)

        # Display the table
        st.dataframe(
            gpu_table_df,
            use_container_width=True,
            column_config={
                'GPU': st.column_config.TextColumn("GPU Model"),
                'Value Score': st.column_config.NumberColumn("Value Score", format="%.1f"),
                'Actual Price ($)': st.column_config.NumberColumn("Actual Price ($)", format="$%d"),
                'MSRP ($)': st.column_config.NumberColumn("MSRP ($)", format="$%d"),
                'Avg FPS': st.column_config.NumberColumn("Avg FPS", format="%.1f"),
                'Bottom 1% FPS': st.column_config.NumberColumn("Bottom 1% FPS", format="%.1f"),
                'Bottom 0.1% FPS': st.column_config.NumberColumn("Bottom 0.1% FPS", format="%.1f"),
                'Avg FPS/Dollar': st.column_config.NumberColumn("Avg FPS/Dollar", format="%.3f"),
                'Bottom 1% FPS/Dollar': st.column_config.NumberColumn("Bottom 1% FPS/Dollar", format="%.3f"),
                'Bottom 0.1% FPS/Dollar': st.column_config.NumberColumn("Bottom 0.1% FPS/Dollar", format="%.3f"),
                'Avg FPS/Watt': st.column_config.NumberColumn("Avg FPS/Watt", format="%.3f"),
                'Bottom 1% FPS/Watt': st.column_config.NumberColumn("Bottom 1% FPS/Watt", format="%.3f"),
                'Bottom 0.1% FPS/Watt': st.column_config.NumberColumn("Bottom 0.1% FPS/Watt", format="%.3f"),
                'VRAM (GB)': st.column_config.NumberColumn("VRAM (GB)", format="%d"),
                'VRAM (GB)/Dollar': st.column_config.NumberColumn("VRAM (GB)/Dollar", format="%.3f")
            },
            hide_index=True
        )
    else:
        st.write("No data available for ranking. Please select at least one feature.")

## ========================================================================== ##
## Explanation ##
## ========================================================================== ##

# Show explanation of the value proposition calculation
st.header("How the Value Score is Calculated")
st.write("The value proposition score is calculated as a linear function of the selected variables:")

# Add LaTeX formula for the value proposition calculation
st.latex(r'''
\text{Value Score} = \left( \sum_{i \in \text{PriceFeatures}} \frac{\text{Importance}_i \times \text{Feature}_i}{\text{Price}} + \sum_{j \in \text{PowerFeatures}} \frac{\text{Feature}_j}{\text{TDP}} \right) \times \prod_{k \in \text{PremiumFactors}} \text{Factor}_k
''')

st.write("""
Where:
- **Price Features**: Performance per dollar metrics (Avg FPS, Bottom 1% FPS, Bottom 0.1% FPS, and VRAM amount divided by price)
- **Importance**: User-assigned weights for each price feature
- **Power Features**: Performance per wattage metrics (same performance metrics divided by power consumption)
- **Premium Factors**: Optional multipliers for memory type, GPU brand, and AIB brand preferences

The formula calculates:
- All enabled variables are weighted equally
- Scores are expressed as percentage, normalized to 100%, with higher scores indicating better value propositions
""")