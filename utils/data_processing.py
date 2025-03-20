from typing import Dict, Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path: str) -> pd.DataFrame:
    """Load GPU data from CSV file."""
    return pd.read_csv(file_path)

def calculate_value_proposition(
        data: pd.DataFrame,
        enabled_features: dict,
        memory_premium_factors: Dict[str, float],
        gpu_premium_factors: Dict[str, float],
        aib_premium_factors: Dict[str, float],
        feature_weights: Dict[str, float] = None,
        price_step: Optional[int] = 10
    ) -> pd.DataFrame:
    """
    Calculate the value proposition score based on enabled features.

    Parameters:
    -----------
    data : pandas.DataFrame
        The GPU dataset
    enabled_features : dict
        Dictionary with feature names as keys and boolean values indicating if they're enabled
    memory_premium_factors : dict
        Dictionary with memory types as keys and premium factors as values
    gpu_premium_factors : dict
        Dictionary with GPU brands as keys and premium factors as values
    aib_premium_factors : dict
        Dictionary with AIB brands as keys and premium factors as values
    feature_weights : dict, optional
        Dictionary with feature names as keys and weight values (0-100)
    price_step : int, optional
        Price step size, by default 10

    Returns:
    --------
    pandas.DataFrame
        Processed data in long form, with columns 'price', 'value_score', and GPU names
    """
    # Select base features for calculation (i.e. non-price-premium features)
    premium_features: set = {'memory_type', 'gpu_brand', 'aib_brand'}
    enabled_base_features: list[str] = [feat for feat, enabled in enabled_features.items() if enabled and feat not in premium_features]
    enabled_premium_features: list[str] = [feat for feat, enabled in enabled_features.items() if enabled and feat in premium_features]

    # Initialize results dataframe in long-form
    results = []

    # Return empty dataframe if no base features selected
    if len(enabled_base_features) == 0:
        return pd.DataFrame(columns=['gpu_name', 'price', 'value_score', 'price_type'])

    # Normalize weights if provided
    if feature_weights is None:
        # Default: equal weights
        feature_weights = {feature: 1.0 / len(enabled_base_features) for feature in enabled_base_features}
    else:
        # Convert percentage weights to decimals (0-1)
        feature_weights = {k: v / 100.0 for k, v in feature_weights.items() if k in enabled_base_features}

    # Mapping feature names to required numerator columns
    feature_price = {
        'performance_avg': 'avg_fps',
        'performance_bottom_1': 'bottom_1_fps',
        'performance_bottom_01': 'bottom_01_fps',
        'vram': 'vram_amount',
    }
    feature_power = {
        'efficiency_avg': 'avg_fps',
        'efficiency_bottom_1': 'bottom_1_fps',
        'efficiency_bottom_01': 'bottom_01_fps',
    }

    # Process each GPU separately
    for _, gpu_row in data.iterrows():
        gpu_name = gpu_row['gpu_name']
        msrp = gpu_row['msrp']
        actual_price = gpu_row['actual_price']

        # Generate price range between MSRP and actual_price (if available)
        if (not pd.isna(actual_price)) and (not pd.isna(msrp)):
            min_price = min(msrp, actual_price)
            max_price = max(msrp, actual_price)
            price_range = np.arange(min_price, max_price + price_step, price_step)
        elif (not pd.isna(msrp)):
            # If actual_price is missing, just use MSRP as a single point
            price_range = np.array([msrp])
        elif (not pd.isna(actual_price)):
            # If MSRP is missing, just use actual_price as a single point
            price_range = np.array([actual_price])
        else:
            # Both prices are missing, skip
            continue

        # Calculate value scores for each price point
        for price in price_range:
            value_score = 0
            # Price Performance Ratios with weights
            for ui_name, column_name in feature_price.items():
                if ui_name not in enabled_base_features:
                    continue
                feature_value = gpu_row[column_name]
                weight = feature_weights.get(ui_name, 0)
                value_score += weight * (feature_value / price)

            # Power Efficiency Ratios with weights
            for ui_name, column_name in feature_power.items():
                if ui_name not in enabled_base_features:
                    continue
                feature_value = gpu_row[column_name]
                power_consumption = gpu_row['tdp']
                weight = feature_weights.get(ui_name, 0)
                value_score += weight * (feature_value / power_consumption)

            # Apply premium factors
            # Memory type premium
            if 'memory_type' in enabled_premium_features:
                memory_type = gpu_row['memory_type']
                if memory_type in memory_premium_factors:
                    value_score *= memory_premium_factors[memory_type]

            # GPU brand premium
            if 'gpu_brand' in enabled_premium_features:
                brand = gpu_row['gpu_brand']
                if brand in gpu_premium_factors:
                    value_score *= gpu_premium_factors[brand]

            # AIB brand premium
            if 'aib_brand' in enabled_premium_features:
                brand = gpu_row['aib_brand']
                if brand in aib_premium_factors:
                    value_score *= aib_premium_factors[brand]

            # Add to results
            results.append({
                'gpu_name': gpu_name,
                'price': price,
                'value_score': value_score,
                'price_type': 'curve'  # This helps identify points vs curve in plotting
            })

        # Add MSRP and actual_price as specific points
        if not pd.isna(msrp):
            results.append({
                'gpu_name': gpu_name,
                'price': msrp,
                'value_score': None,  # Will be filled later
                'price_type': 'msrp'
            })

        if not pd.isna(actual_price):
            results.append({
                'gpu_name': gpu_name,
                'price': actual_price,
                'value_score': None,  # Will be filled later
                'price_type': 'actual'
            })

    # Convert results to DataFrame
    df_results = pd.DataFrame(results)

    # If no results, return empty DataFrame
    if len(df_results) == 0:
        return df_results

    # Normalize to 0-100 range, if there are curve points to scale
    scaler = MinMaxScaler(feature_range=(0, 100))
    curve_mask = df_results['price_type'] == 'curve'
    if curve_mask.any():
        df_results.loc[curve_mask, 'value_score'] = scaler.fit_transform(
            df_results.loc[curve_mask, 'value_score'].values.reshape(-1, 1)
        ).flatten()

        # Calculate value scores for MSRP and actual price points
        for gpu_name in df_results['gpu_name'].unique():
            gpu_curve = df_results[(df_results['gpu_name'] == gpu_name) & (df_results['price_type'] == 'curve')]

            # Fill MSRP value score
            msrp_mask = (df_results['gpu_name'] == gpu_name) & (df_results['price_type'] == 'msrp')
            if msrp_mask.any():
                msrp_price = df_results.loc[msrp_mask, 'price'].values[0]
                closest_curve_point = gpu_curve.iloc[(gpu_curve['price'] - msrp_price).abs().argsort()[:1]]
                if not closest_curve_point.empty:
                    df_results.loc[msrp_mask, 'value_score'] = closest_curve_point['value_score'].values[0]

            # Fill actual price value score
            actual_mask = (df_results['gpu_name'] == gpu_name) & (df_results['price_type'] == 'actual')
            if actual_mask.any():
                actual_price = df_results.loc[actual_mask, 'price'].values[0]
                closest_curve_point = gpu_curve.iloc[(gpu_curve['price'] - actual_price).abs().argsort()[:1]]
                if not closest_curve_point.empty:
                    df_results.loc[actual_mask, 'value_score'] = closest_curve_point['value_score'].values[0]

    return df_results
