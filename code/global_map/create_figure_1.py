#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stack before and after retaliation maps vertically with a single legend
"""

import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import pandas as pd
import os
import sys

# Add code_python to path to import config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'code_python'))
try:
    from config import get_output_dir
except ImportError:
    # Fallback if config not available
    def get_output_dir():
        return 'output'

def create_stacked_maps(before_data_file, after_data_file, output_file):
    # Load data
    before_data = pd.read_csv(before_data_file)
    after_data = pd.read_csv(after_data_file)
    
    # Load world map
    world = gpd.read_file("https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip")
    
    # Fix country codes in the world map
    # France is 'France' in the map but 'FRA' in the data
    france_mask = world['ADMIN'] == 'France'
    if france_mask.any():
        world.loc[france_mask, 'ISO_A3'] = 'FRA'
    
    # Norway is 'Norway' in the map but 'NOR' in the data
    norway_mask = world['ADMIN'] == 'Norway'
    if norway_mask.any():
        world.loc[norway_mask, 'ISO_A3'] = 'NOR'
    
    # Create dictionaries for quick lookup
    before_value_dict = dict(zip(before_data['Country'], before_data['Value_1']))
    after_value_dict = dict(zip(after_data['Country'], after_data['Value_1']))
    
    # Create copies of the world dataframe for before and after
    world_before = world.copy()
    world_after = world.copy()
    
    # Create columns for values
    world_before['value'] = np.nan
    world_after['value'] = np.nan
    
    # Directly assign values to countries by ISO code
    for country_code, value in before_value_dict.items():
        world_before.loc[world_before['ISO_A3'] == country_code, 'value'] = value
    
    for country_code, value in after_value_dict.items():
        world_after.loc[world_after['ISO_A3'] == country_code, 'value'] = value
    
    # Winsorize values between -3% and 3%
    world_before['value_winsorized'] = world_before['value'].clip(lower=-3.0, upper=3.0)
    world_after['value_winsorized'] = world_after['value'].clip(lower=-3.0, upper=3.0)
    
    # Create figure with proper spacing for maps and legend
    fig = plt.figure(figsize=(12, 14))
    
    # Create a gridspec with tighter spacing
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.05)
    
    # Create axes for the maps
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Create a smaller subplot for the colorbar (40% smaller)
    # Center it horizontally and position it closer to the lower map
    # [left, bottom, width, height]
    cax = fig.add_axes([0.3, 0.08, 0.4, 0.02])  # Increased bottom value to move it closer to the map
    
    # Use the built-in RdBu colormap - a reliable standard
    # This ensures negative values are red and positive values are blue
    cmap = plt.cm.RdBu
    
    # Plot the maps with the colormap
    world_before.plot(
        column='value_winsorized',
        cmap=cmap,
        linewidth=0.5,
        ax=ax1,
        edgecolor='0.8',
        missing_kwds={'color': 'lightgray'},
        vmin=-3.0,
        vmax=3.0,
        legend=False  # No legend for the first map
    )
    
    # Plot the second map
    world_after.plot(
        column='value_winsorized',
        cmap=cmap,
        linewidth=0.5,
        ax=ax2,
        edgecolor='0.8',
        missing_kwds={'color': 'lightgray'},
        vmin=-3.0,
        vmax=3.0,
        legend=False  # No legend for the second map
    )
    
    # Add titles to each subplot
    ax1.set_title('Before Retaliation', fontsize=16)
    ax2.set_title('After Retaliation', fontsize=16)
    
    # Remove axes
    ax1.set_axis_off()
    ax2.set_axis_off()
    
    # Add a single colorbar at the bottom
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-3.0, vmax=3.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_label('Welfare Effect (%)', fontsize=10)
    
    # Add tick marks at specific values
    cbar.set_ticks([-3, -2, -1, 0, 1, 2, 3])
    cbar.set_ticklabels(['-3%', '-2%', '-1%', '0%', '1%', '2%', '3%'])
    cbar.ax.tick_params(labelsize=8)  # Smaller tick labels
    
    # Save the stacked figure with tighter layout
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"Stacked map saved to {output_file}")

# Run the script
# Find the replication_package directory
script_dir = os.path.dirname(os.path.abspath(__file__))
replication_dir = os.path.join(script_dir, '..', '..')
output_dir = get_output_dir()
data_dir = os.path.join(replication_dir, output_dir)

# Ensure output directory exists
os.makedirs(data_dir, exist_ok=True)

create_stacked_maps(
    os.path.join(data_dir, 'output_map.csv'),
    os.path.join(data_dir, 'output_map_retal.csv'),
    os.path.join(data_dir, 'figure_1.png')
)

print(f"Figure saved to {os.path.join(data_dir, 'figure_1.png')}")
# Note: Not removing CSV files as they may be needed for other analyses
