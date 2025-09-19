import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import logging
import os
from contextlib import contextmanager
from urllib.parse import urlparse
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.warning("python-dotenv not installed. Please install it with: pip install python-dotenv")
    logger.warning("Or set environment variables manually.")

# Configuration
class Config:
    # Database URL from environment variable (required)
    DATABASE_URL = os.getenv('DATABASE_URL')
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is required. Please set it in your .env file.")
    
    SAMPLE_SIZE = int(os.getenv('SAMPLE_SIZE', '25000'))
    MIN_FLOW_COUNT = int(os.getenv('MIN_FLOW_COUNT', '1'))
    OUTPUT_FILE = os.getenv('OUTPUT_FILE', 'london_flowmap.html')

@contextmanager
def get_db_connection():
    """Context manager for database connections with proper error handling."""
    conn = None
    try:
        # Parse database URL
        parsed = urlparse(Config.DATABASE_URL)
        
        conn = psycopg2.connect(
            dbname=parsed.path[1:],  # Remove leading slash
            user=parsed.username,
            password=parsed.password,
            host=parsed.hostname,
            port=parsed.port
        )
        logger.info("Database connection established")
        yield conn
    except psycopg2.Error as e:
        logger.error(f"Database connection error: {e}")
        raise
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed")

def get_data_overview(conn) -> pd.DataFrame:
    """Get overview statistics of the journey data."""
    overview_query = """
    SELECT
        MIN(start_date) as earliest_date,
        MAX(start_date) as latest_date,
        COUNT(*) as total_journeys,
        COUNT(DISTINCT start_station_number) as unique_start_stations,
        COUNT(DISTINCT end_station_number) as unique_end_stations
    FROM tfl_bike_journeys
    WHERE start_station_number IS NOT NULL
      AND end_station_number IS NOT NULL;
    """
    
    logger.info("Getting data overview...")
    overview = pd.read_sql_query(overview_query, conn)
    
    logger.info(f"Data spans from {overview['earliest_date'].iloc[0]} to {overview['latest_date'].iloc[0]}")
    logger.info(f"Total journeys in database: {overview['total_journeys'].iloc[0]:,}")
    
    return overview

def load_journey_data(conn, use_date_filter: bool = True) -> pd.DataFrame:
    """Load journey data for July 3rd, 2024."""
    if use_date_filter:
        # July 3rd, 2024 only
        query = """
        SELECT
            start_date,
            end_date,
            start_station_number as origin_id,
            end_station_number as dest_id,
            bike_number,
            bike_model,
            "total_duration_(ms)" / 1000.0 as duration_seconds
        FROM tfl_bike_journeys
        WHERE start_station_number IS NOT NULL
          AND end_station_number IS NOT NULL
          AND start_station_number != end_station_number
          AND "total_duration_(ms)" > 0
          AND start_date >= '2024-07-03'
          AND start_date < '2024-07-04'
        ORDER BY start_date;
        """
        logger.info("Loading cycle hire data for July 3rd, 2024...")
        cycle_data = pd.read_sql_query(query, conn)
    else:
        # Fallback to sample if no data in July 2024
        query = """
        SELECT
            start_date,
            end_date,
            start_station_number as origin_id,
            end_station_number as dest_id,
            bike_number,
            bike_model,
            "total_duration_(ms)" / 1000.0 as duration_seconds
        FROM tfl_bike_journeys
        WHERE start_station_number IS NOT NULL
          AND end_station_number IS NOT NULL
          AND start_station_number != end_station_number
          AND "total_duration_(ms)" > 0
        ORDER BY RANDOM()
        LIMIT %s;
        """
        logger.info("Loading sample cycle hire data...")
        cycle_data = pd.read_sql_query(query, conn, params=[Config.SAMPLE_SIZE])
    
    logger.info(f"Loaded {len(cycle_data)} journey records")
    
    # Data validation
    if cycle_data.empty:
        raise ValueError("No journey data loaded")
    
    # Remove any remaining invalid data
    initial_count = len(cycle_data)
    cycle_data = cycle_data.dropna()
    cycle_data = cycle_data[cycle_data['duration_seconds'] > 0]
    
    if len(cycle_data) != initial_count:
        logger.info(f"Cleaned data: removed {initial_count - len(cycle_data)} invalid records")
    
    return cycle_data

def load_station_data(conn) -> Tuple[pd.DataFrame, Tuple[float, float]]:
    """Load station data and calculate center coordinates."""
    logger.info("Loading bike station locations...")
    
    # Load all stations with valid coordinates
    station_query = """
    SELECT station_id as id, station_name, latitude as lat, longitude as lon
    FROM bike_stations
    WHERE latitude IS NOT NULL 
      AND longitude IS NOT NULL
      AND latitude BETWEEN -90 AND 90
      AND longitude BETWEEN -180 AND 180;
    """
    
    stations = pd.read_sql_query(station_query, conn)
    logger.info(f"Loaded {len(stations)} stations with valid coordinates")
    
    # Additional data validation
    stations = stations.dropna(subset=['lat', 'lon'])
    
    if stations.empty:
        raise ValueError("No valid station data loaded")
    
    # Calculate London center from actual station data
    london_center_lat = stations['lat'].mean()
    london_center_lon = stations['lon'].mean()
    logger.info(f"London center coordinates: {london_center_lat:.4f}, {london_center_lon:.4f}")
    
    return stations, (london_center_lat, london_center_lon)

def process_flows(cycle_data: pd.DataFrame, stations: pd.DataFrame, 
                 min_count: int = Config.MIN_FLOW_COUNT) -> pd.DataFrame:
    """Aggregate trips into flows and merge with station coordinates."""
    logger.info("Aggregating flows...")
    
    # Aggregate trips between station pairs using vectorized operations
    flows = (cycle_data.groupby(['origin_id', 'dest_id'], as_index=False)
             .agg({
                 'bike_number': 'count',
                 'duration_seconds': 'mean'
             }))
    flows.columns = ['origin', 'dest', 'count', 'avg_duration']
    
    # Filter for significant flows
    flows = flows[flows['count'] > min_count]
    logger.info(f"Found {len(flows)} significant flow pairs (>{min_count} trips)")
    
    # Prepare station data for merging (avoid duplicate column renaming)
    origin_stations = stations.rename(columns={
        'id': 'origin', 'lat': 'origin_lat', 'lon': 'origin_lon', 'station_name': 'origin_name'
    })
    dest_stations = stations.rename(columns={
        'id': 'dest', 'lat': 'dest_lat', 'lon': 'dest_lon', 'station_name': 'dest_name'
    })
    
    # Merge with station coordinates using inner joins for efficiency
    flows_with_coords = (flows
                        .merge(origin_stations[['origin', 'origin_lat', 'origin_lon', 'origin_name']], 
                              on='origin', how='inner')
                        .merge(dest_stations[['dest', 'dest_lat', 'dest_lon', 'dest_name']], 
                              on='dest', how='inner'))
    
    logger.info(f"Final dataset: {len(flows_with_coords)} flows with coordinates")
    
    if flows_with_coords.empty:
        raise ValueError("No flows with valid coordinates found")
    
    return flows_with_coords

def prepare_station_stats(stations: pd.DataFrame, flows: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """Prepare station statistics for visualization using vectorized operations."""
    logger.info("Preparing station statistics...")
    
    # Calculate origin and destination trip counts using vectorized operations
    origin_counts = flows.groupby('origin')['count'].sum().reset_index()
    origin_counts.columns = ['id', 'origin_trips']
    
    dest_counts = flows.groupby('dest')['count'].sum().reset_index()
    dest_counts.columns = ['id', 'dest_trips']
    
    # Merge with station data
    station_stats = (stations
                    .merge(origin_counts, on='id', how='left')
                    .merge(dest_counts, on='id', how='left'))
    
    # Fill NaN values with 0 and calculate total trips
    station_stats['origin_trips'] = station_stats['origin_trips'].fillna(0)
    station_stats['dest_trips'] = station_stats['dest_trips'].fillna(0)
    station_stats['total_trips'] = station_stats['origin_trips'] + station_stats['dest_trips']
    
    # Filter stations with trips and get top N by total traffic
    station_stats = station_stats[station_stats['total_trips'] > 0]
    station_stats = station_stats.nlargest(top_n, 'total_trips')
    
    logger.info(f"Filtered to top {len(station_stats)} busiest stations (max {top_n})")
    logger.info(f"Top station has {station_stats['total_trips'].max()} trips")
    
    return station_stats

def filter_flows_for_top_stations(flows_with_coords: pd.DataFrame, top_station_ids: set) -> pd.DataFrame:
    """Filter flows to only show connections between top stations."""
    logger.info("Filtering flows to top stations only...")
    
    # Keep only flows where both origin and destination are in top stations
    filtered_flows = flows_with_coords[
        (flows_with_coords['origin'].isin(top_station_ids)) & 
        (flows_with_coords['dest'].isin(top_station_ids))
    ]
    
    logger.info(f"Filtered flows from {len(flows_with_coords)} to {len(filtered_flows)} (top stations only)")
    
    return filtered_flows

def create_flow_traces(flows_with_coords: pd.DataFrame) -> List[go.Scattermapbox]:
    """Create flow line traces with optimized performance."""
    logger.info("Creating flow traces...")
    traces = []
    
    if flows_with_coords.empty:
        return traces
    
    max_count = flows_with_coords['count'].max()
    
    # Define flow categories with their properties
    flow_categories = [
        {
            'name': 'High Flow (70%+)',
            'filter': flows_with_coords['count'] > max_count * 0.7,
            'color': 'red',
            'width_multiplier': 8,
            'opacity': 0.8
        },
        {
            'name': 'Medium Flow (40-70%)',
            'filter': (flows_with_coords['count'] > max_count * 0.4) & 
                     (flows_with_coords['count'] <= max_count * 0.7),
            'color': 'orange', 
            'width_multiplier': 6,
            'opacity': 0.6
        },
        {
            'name': 'Low Flow (<40%)',
            'filter': flows_with_coords['count'] <= max_count * 0.4,
            'color': 'green',
            'width_multiplier': 4,
            'opacity': 0.4
        }
    ]
    
    for category in flow_categories:
        category_flows = flows_with_coords[category['filter']]
        
        if len(category_flows) == 0:
            continue
            
        # Create vectorized arrays for all lines in this category
        lats = []
        lons = []
        hover_texts = []
        line_widths = []
        
        for _, flow in category_flows.iterrows():
            lats.extend([flow['origin_lat'], flow['dest_lat'], None])
            lons.extend([flow['origin_lon'], flow['dest_lon'], None])
            
            width = max(1, min(category['width_multiplier'], 
                             flow['count'] / max_count * category['width_multiplier']))
            line_widths.extend([width, width, width])
            
            # Create detailed tooltip with station names and numbers
            origin_name = flow.get('origin_name', 'Unknown')
            dest_name = flow.get('dest_name', 'Unknown')
            
            hover_text = (f"<b>{flow['count']} journeys</b><br>"
                         f"<b>From:</b> {origin_name} (#{flow['origin']})<br>"
                         f"<b>To:</b> {dest_name} (#{flow['dest']})<br>"
                         f"<b>Avg Duration:</b> {flow['avg_duration']/60:.1f} minutes<br>"
                         f"<b>Flow Category:</b> {category['name']}")
            hover_texts.extend([hover_text, hover_text, ''])
        
        # Create single trace for entire category
        if lats:
            traces.append(go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode='lines',
                line=dict(width=2, color=category['color']),
                opacity=category['opacity'],
                hovertext=hover_texts,
                name=category['name'],
                showlegend=True
            ))
    
    logger.info(f"Created {len(traces)} flow trace categories")
    return traces

def create_station_trace(station_stats: pd.DataFrame) -> go.Scattermapbox:
    """Create station marker trace."""
    logger.info("Creating station markers...")
    
    if station_stats.empty:
        return go.Scattermapbox()
    
    # Calculate marker sizes using vectorized operations
    marker_sizes = np.clip(station_stats['total_trips'] / 5, 8, 20)
    
    # Create more informative hover text using vectorized operations
    hover_texts = ('<b>' + station_stats['station_name'].astype(str) + '</b><br>' +
                  '<b>Station ID:</b> #' + station_stats['id'].astype(str) + '<br>' +
                  '<b>Total Journeys:</b> ' + station_stats['total_trips'].astype(str) + '<br>' +
                  '<b>Departures:</b> ' + station_stats['origin_trips'].astype(str) + '<br>' +
                  '<b>Arrivals:</b> ' + station_stats['dest_trips'].astype(str) + '<br>' +
                  '<b>Net Flow:</b> ' + (station_stats['origin_trips'] - station_stats['dest_trips']).astype(str))
    
    return go.Scattermapbox(
        lat=station_stats['lat'],
        lon=station_stats['lon'],
        mode='markers',
        marker=dict(
            size=marker_sizes,
            color=station_stats['total_trips'],
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Total Trips"),
            opacity=0.8
        ),
        text=hover_texts,
        hovertemplate='%{text}<extra></extra>',
        name='Bike Stations',
        showlegend=True
    )

def create_visualization(overview: pd.DataFrame, cycle_data: pd.DataFrame, 
                        flows_with_coords: pd.DataFrame, station_stats: pd.DataFrame,
                        center_coords: Tuple[float, float]) -> go.Figure:
    """Create the complete visualization."""
    logger.info("Creating interactive flow map...")
    
    fig = go.Figure()
    
    # Add flow traces
    flow_traces = create_flow_traces(flows_with_coords)
    for trace in flow_traces:
        fig.add_trace(trace)
    
    # Add station trace
    station_trace = create_station_trace(station_stats)
    fig.add_trace(station_trace)
    
    # Update layout
    fig.update_layout(
        title={
            'text': (f"<b>London Cycle Hire Flow Map - Top 50 Busiest Stations</b><br>"
                    f"<span style='font-size:14px'>"
                    f"July 3rd, 2024 (Wednesday) | "
                    f"{len(cycle_data):,} total journeys | "
                    f"Top {len(station_stats)} stations shown</span>"),
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        mapbox=dict(
            style='open-street-map',
            center=dict(lat=center_coords[0], lon=center_coords[1]),
            zoom=11
        ),
        height=700,
        margin=dict(l=0, r=0, t=80, b=0),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left", 
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )
    
    return fig

def print_summary(stations: pd.DataFrame, flows_with_coords: pd.DataFrame, cycle_data: pd.DataFrame):
    """Print summary statistics."""
    logger.info("Summary statistics:")
    logger.info(f"- Total stations: {len(stations)}")
    logger.info(f"- Total flows: {len(flows_with_coords)}")
    logger.info(f"- Total trips: {flows_with_coords['count'].sum()}")
    logger.info(f"- Max flow between stations: {flows_with_coords['count'].max()} trips")
    logger.info(f"- Average trip duration: {cycle_data['duration_seconds'].mean()/60:.1f} minutes")

def main():
    """Main execution function."""
    try:
        with get_db_connection() as conn:
            # Load data
            overview = get_data_overview(conn)
            
            # Try to load July 3rd, 2024 data first
            try:
                cycle_data = load_journey_data(conn, use_date_filter=True)
                if len(cycle_data) == 0:
                    logger.warning("No data found for July 3rd, 2024, using sample data instead")
                    cycle_data = load_journey_data(conn, use_date_filter=False)
            except Exception as e:
                logger.warning(f"Error loading July 3rd, 2024 data: {e}, using sample data instead")
                cycle_data = load_journey_data(conn, use_date_filter=False)
            
            stations, center_coords = load_station_data(conn)
            
            # Process data
            flows_with_coords = process_flows(cycle_data, stations)
            station_stats = prepare_station_stats(stations, flows_with_coords, top_n=50)
            
            # Filter flows to only show connections between top 50 stations
            top_station_ids = set(station_stats['id'])
            filtered_flows = filter_flows_for_top_stations(flows_with_coords, top_station_ids)
            
            # Create visualization
            fig = create_visualization(overview, cycle_data, filtered_flows, 
                                     station_stats, center_coords)
            
            # Save the map
            logger.info(f"Saving interactive map to {Config.OUTPUT_FILE}...")
            fig.write_html(Config.OUTPUT_FILE, include_plotlyjs=True)
            logger.info(f"✓ Saved as '{Config.OUTPUT_FILE}'")
            
            # Print summary
            print_summary(station_stats, filtered_flows, cycle_data)
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
