## Data Science for Transport Planning (dstp-al)

This repository contains resources for the **Data Science for Transport Planning** course project at the University of Leeds, led by Prof. Robin Lovelace.

### Project Overview

The project applies practical data science methods to real-world transport datasets, focusing on West Yorkshire, UK. Key topics covered include:

- Downloading and processing OpenStreetMap (OSM) data for cycleways, roads, and amenities
- Working with UK road traffic casualty data (STATS19)
- Analyzing origin-destination (OD) data to understand travel patterns
- Using boundary and census data for spatial analysis
- Cleaning, transforming, and handling missing data in transport datasets

### Key Files

- `s1.ipynd`: Main analysis notebook.
- `tests/`: Directory containing example scripts in R and Python for data access and validation.
- `data/`: Directory for storing downloaded and processed data files.

### Requirements

- R
- R packages:
  - `osmextract`
  - `sf`
  - `stats19`
  - `pct`
  - `dplyr`
  - `tidyr`
  - `ggplot2`
  - `tmap`
  - `zonebuilder`
  - (and other tidyverse packages as needed)
- (Optional) Python 3 with:
  - `osmnx`
  - `geopandas`
  - `matplotlib`
  - `shapely`

To install the required R packages, you can use the following code in your R console:
