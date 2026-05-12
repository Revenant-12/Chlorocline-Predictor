import pandas as pd
import matplotlib.pyplot as plt
import os
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point
from adjustText import adjust_text

# File paths
cords_path = os.path.join(os.path.dirname(__file__), 'SOPS_Cords.csv')
raw_data_path = os.path.join(os.path.dirname(__file__), '..', 'Raw data chlorocline.csv')

def create_map():
    # Load coordinates
    cords_df = pd.read_csv(cords_path)
    cords_df = cords_df[['Site Name', 'Latitude', 'Longitude']].dropna(subset=['Site Name', 'Latitude', 'Longitude'])
    
    # Load raw data to count occurrences
    raw_df = pd.read_csv(raw_data_path)
    
    # Clean site names for better matching
    cords_df['Site_Clean'] = cords_df['Site Name'].str.strip().str.upper()
    raw_df['Site_Clean'] = raw_df['Site'].str.strip().str.upper()
    
    # Count occurrences
    site_counts = raw_df['Site_Clean'].value_counts().reset_index()
    site_counts.columns = ['Site_Clean', 'Count']
    
    # Merge counts into coordinates
    merged_df = pd.merge(cords_df, site_counts, on='Site_Clean', how='left')
    merged_df['Count'] = merged_df['Count'].fillna(0)
    
    # Create a GeoDataFrame
    # EPSG:4326 is standard WGS84 (Lat/Lon)
    geometry = [Point(xy) for xy in zip(merged_df['Longitude'], merged_df['Latitude'])]
    gdf = gpd.GeoDataFrame(merged_df, geometry=geometry, crs="EPSG:4326")
    
    # Convert to Web Mercator (EPSG:3857) for compatibility with most basemaps
    gdf = gdf.to_crs(epsg=3857)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # Scale factor for dot size
    scale = 30
    
    # Filter for sites with data for main plotting
    with_data = gdf[gdf['Count'] > 0]
    no_data = gdf[gdf['Count'] == 0]
    
    # Plot sites with data
    with_data.plot(
        ax=ax,
        markersize=with_data['Count'] * scale,
        alpha=0.7,
        column='Count',
        cmap='viridis',
        edgecolor='white',
        linewidth=1,
        legend=True,
        legend_kwds={'label': "Occurrence Count", 'orientation': "vertical", 'shrink': 0.5}
    )
    
    # Plot sites with no data as small points
    no_data.plot(
        ax=ax,
        markersize=20,
        color='gray',
        alpha=0.5,
        edgecolor='none'
    )
    
    # Add labels for sites with data
    texts = []
    for x, y, label, count in zip(with_data.geometry.x, with_data.geometry.y, with_data['Site Name'], with_data['Count']):
        texts.append(ax.text(
            x, y,
            f"{label} ({int(count)})", 
            fontsize=9,
            fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
        ))
    
    # Adjust text to prevent overlap
    if texts:
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='black', lw=0.5))

    # Add basemap
    # Using OpenStreetMap tiles
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
    
    ax.set_axis_off()
    plt.title('Data Collection Sites Map Overlay\n(Size represents frequency in of sampling)', fontsize=16, pad=20)
    
    # Save the map
    output_path = os.path.join(os.path.dirname(__file__), 'site_coordinate_map_overlay.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Map with overlay saved to {output_path}")

if __name__ == "__main__":
    create_map()
