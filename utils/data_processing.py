from typing import Literal, Dict, Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path: str) -> pd.DataFrame:
    """Load GPU data from CSV file."""
    return pd.read_csv(file_path)

def calculate_value_proposition(
        data: pd.DataFrame,
        enabled_features: dict,
        price_feature: Literal['actual_price','msrp'],
        power_feature: Literal['tdp'],
        memory_premium_factors: Dict[str, float],
        gpu_premium_factors: Dict[str, float],
        aib_premium_factors: Dict[str, float],
        price_min: Optional[int] = 100,
        price_max: Optional[int] = 2001,
        price_step: Optional[int] = 10
    ) -> pd.DataFrame:
    """
    Calculate the value proposition score based on enabled features.

    Parameters:
    -----------
    data : pandas.DataFrame
        The GPU dataset
    enabled_features : dict
        Dictionary with feature names as keys and boolean values
        indicating if they're enabled
    price_feature : str
        The name of the column to use for price-based calculations ('actual_price' or 'msrp')
    power_feature : str
        The name of the column to use for power-based calculations ('tdp')
    memory_premium_factors : dict
        Dictionary with memory types as keys and premium factors as values
    gpu_premium_factors : dict
        Dictionary with GPU brands as keys and premium factors as values
    aib_premium_factors : dict
        Dictionary with AIB brands as keys and premium factors as values
    price_min : int, optional
        Minimum price to consider, by default 100
    price_max : int, optional
        Maximum price to consider, by default 2001
    price_step : int, optional
        Price step size, by default 10

    Returns:
    --------
    pandas.DataFrame
        Processed data in long form, with columns 'price', 'value_score', and GPU names
    """
    # Unique GPU names
    gpu_names = data['gpu_name'].unique()
    # Generate Price Range (shape: (num_prices, 1))
    price_range = np.arange(price_min, price_max, price_step)

    # Initialize working dataframe, using wide-form for easier calculation
    df_wide_form = pd.DataFrame({'price': price_range})
    for gpu_name in gpu_names:
        df_wide_form[gpu_name] = 0

    # Select base features for calculation (exclude premium features)
    base_features = [feature for feature in enabled_features if feature not in ['memory_type', 'gpu_brand', 'aib_brand']]
    # Return zero scores if no base features selected
    if len(base_features) == 0:
        return df_wide_form.melt(id_vars='price', var_name='gpu_name', value_name='value_score')

    # ======================================================================== #
    # Calculate value score
    # ======================================================================== #

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

    for gpu_name in gpu_names:
        # Price Performance Ratios
        # For each enabled feature, prepare the matrix of feature values (shape: (num_features, 1))
        feature_vector = []
        for ui_name,column_name in feature_price.items():
            # Skip if feature not enabled
            if ui_name not in base_features:
                continue
            mask = (data['gpu_name'] == gpu_name).values
            feature_vector.append(data.loc[mask, column_name].values)
        if len(feature_vector) > 0:
            # Convert to shape: (1, num_features)
            feature_vector = np.array(feature_vector).reshape(1, -1)
            # Dot product (division) calculation (shape: (num_prices, 1) @ (1, num_features) = (num_prices, num_features))
            value_score = (1/price_range).reshape(-1,1) @ feature_vector
            # Sum by feature (shape: (num_prices, 1))
            value_score = value_score.sum(axis=1)
            # Store results in wide-form dataframe
            df_wide_form[gpu_name] += value_score

        # Power Efficiency Ratios
        # For each enabled feature, prepare the matrix of feature values (shape: (num_features, 1))
        feature_vector = []
        for ui_name,column_name in feature_power.items():
            # Skip if feature not enabled
            if ui_name not in base_features:
                continue
            mask = (data['gpu_name'] == gpu_name).values
            feature_vector.append(data.loc[mask, column_name].values)
        if len(feature_vector) > 0:
            # Shape: (1, num_features)
            feature_vector = np.array(feature_vector)
            # Power consumption (shape: scalar)
            power_consumption = data.loc[mask, power_feature].values
            # Divide by power consumption
            efficiency_score = feature_vector / power_consumption
            # Sum all and store results in wide-form dataframe
            df_wide_form[gpu_name] += efficiency_score.sum()

    # ======================================================================== #
    # Apply premium scaling factors
    # ======================================================================== #

    # Memory type premium
    if 'memory_type' in enabled_features.keys():
        for memory_type, factor in memory_premium_factors.items():
            mask = (data['memory_type'] == memory_type)
            applicable_gpus = data[mask]['gpu_name'].values
            df_wide_form[applicable_gpus] *= factor

    # GPU brand premium
    if 'gpu_brand' in enabled_features.keys():
        for brand, factor in gpu_premium_factors.items():
            mask = data['gpu_brand'] == brand
            applicable_gpus = data[mask]['gpu_name'].values
            df_wide_form[applicable_gpus] *= factor

    # AIB brand premium
    if 'aib_brand' in enabled_features.keys():
        for brand, factor in aib_premium_factors.items():
            mask = data['aib_brand'] == brand
            applicable_gpus = data[mask]['gpu_name'].values
            df_wide_form[applicable_gpus] *= factor

    # ======================================================================== #
    # Normalize scores to 0-100 range, and return long-form dataframe
    # ======================================================================== #

    # Melt wide-form dataframe to long-form
    df_long_form = df_wide_form.melt(id_vars='price', var_name='gpu_name', value_name='value_score')

    # Normalize to 0-100 range
    df_long_form['value_score'] = MinMaxScaler(feature_range=(0, 100)).fit_transform(df_long_form['value_score'].values.reshape(-1,1))

    return df_long_form
