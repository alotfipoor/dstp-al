import osmnx as ox
import geopandas as gpd
import matplotlib.pyplot as plt
import shapely

study_point = shapely.Point(-1.55, 53.80)  # Latitude and Longitude for Leeds
study_geom = gpd.GeoSeries([study_point], crs=4326)
study_polygon = study_geom.to_crs(epsg=3857).buffer(6000).to_crs(epsg=4326).union_all()
study_polygon_gpd = gpd.GeoDataFrame(geometry=[study_polygon], crs="EPSG:4326")
# Read-in geojson already saved from R
study_polygon_gdf = gpd.read_file("leeds_study_area.geojson")
# Extract the actual geometry from the GeoDataFrame
study_polygon_geom = study_polygon_gdf.geometry.iloc[0]  # Get first geometry
# study_polygon_gdf.explore()
tags = {"highway": True, "maxspeed": True, "lit": True, "cycleway": True}
gdf = ox.features_from_polygon(study_polygon_geom, tags)
gdf = gdf[gdf.geom_type.isin(["LineString", "MultiLineString"])]
gdf = gdf.to_crs(epsg=3857)
gdf.plot(column="maxspeed", figsize=(10, 10), legend=True)
plt.show()