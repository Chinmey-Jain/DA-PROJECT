# Tata Motors Sales Dashboard

This project is a Dash application that provides an interactive dashboard for analyzing Tata Motors sales data. The dashboard includes various visualizations and key performance indicators (KPIs) to help users understand sales trends, model performance, fuel type distribution, and geographical sales distribution.
To make the Virtual Environment open termninal with this project directory and type

## Features

- Interactive date range filter
- Model, fuel type, and customer segment filters
- KPIs for total sales, total revenue, average price, and unique customers
- Visualizations including sales trend, model performance, fuel type distribution, and geographical sales distribution
- Data table displaying detailed sales records

## Installation

1. Unzip the downloaded project folder.
2. Navigate to the project directory in your terminal.
3. Create and activate a virtual environment (optional but recommended):
   - On Windows:
     ```
     python -m venv venv
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     python -m venv venv
     source venv/bin/activate
     ```
4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
5. Ensure the sales data file (Excel or CSV) is in the project directory.
6. Run the Dash application:
   ```
   python task.py
   ```

## Usage

1. Ensure that the sales data CSV or Excel file is located in the project directory.
2. Run the Dash application:
   ```
   python task.py
   ```
3. Open your web browser and navigate to `http://127.0.0.1:8050` to view the dashboard.

## Project Structure

```
main()
  │
  ├── load_cache()                 # Warm geocode cache from disk
  ├── load_sales_data()            # Load Excel/CSV file
  ├── clean_sales_data()           # Normalize column names, types, derived fields
  ├── initialize helpers
  │       │
  │       ├── _normalize_*()       # Shared selection-cleaning helpers
  │       └── get_city_coordinates()# Used later during geo plotting
  │
  └── create_app()
          │
          ├── get_app_layout()      # Builds layout: filters, KPIs, charts, table, footer
          └── register_callbacks()
                  │
                  ├── update_kpis()   # Filters data and populates KPI cards
                  └── update_charts() # Filters data, renders charts/table, uses get_city_coordinates()
```

## Function Documentation

| Function                                                                                                       | Purpose                                                                                                                                          |
| -------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `load_cache()`                                                                                                 | Loads cached city coordinates from JSON. Initializes an empty cache when the file is missing/invalid.                                            |
| `save_cache()`                                                                                                 | Persists the in-memory coordinate cache to disk.                                                                                                 |
| `get_city_coordinates(city)`                                                                                   | Returns `(lat, lon)` for a city via cache-first lookup, falling back to Nominatim and refreshing the cache on success.                           |
| `_normalize_list_selection(selection)`                                                                         | Utility that converts multi-select inputs (e.g., models, cities) into a sanitized list, ignoring `"All"` and duplicates.                         |
| `_normalize_scalar_selection(selection)`                                                                       | Utility that returns `None` when a scalar dropdown is `"All"`/empty so downstream filters can skip it.                                           |
| `_build_empty_figure(title, subtitle)`                                                                         | Generates a placeholder Plotly figure used when no data is available for a chart.                                                                |
| `load_sales_data(file_path)`                                                                                   | Loads sales data from Excel/CSV based on file extension, raising if unsupported.                                                                 |
| `clean_sales_data(df)`                                                                                         | Cleans and preprocesses sales data: standardizes names, parses dates, fills categorical defaults, de-noises currency field, and derives `Month`. |
| `create_app(df_clean)`                                                                                         | Creates the Dash app instance, applies layout, and registers callbacks.                                                                          |
| `get_app_layout(df_clean)`                                                                                     | Defines the UI layout: filters (date/model/fuel/segment/city), KPI cards, charts, table, and footer.                                             |
| `filter_data(df_clean, start_date, end_date, selected_models, selected_fuel, selected_segment, selected_city)` | Applies all active filters (dates, models, fuel, segment, city) using the normalization helpers to drive KPIs and charts.                        |
| `register_callbacks(app, df_clean)`                                                                            | Registers the Dash callbacks (`update_kpis`, `update_charts`) that orchestrate filtering, aggregation, chart rendering, and map geocoding.       |
| `main()`                                                                                                       | Entry point: loads cache and data, cleans it, builds the Dash app, and runs the server.                                                          |

## Dash App Flow

1. main() is executed → loads city cache, loads sales data, cleans data, and creates the Dash app.

2. create_app() calls get_app_layout() to define the dashboard layout.

3. register_callbacks() hooks up all user interactions:

   - KPI updates → update_kpis()

   - Charts & table updates → update_charts()

4. User interacts with filters → callbacks are triggered → filtered data is passed to functions for charts, KPIs, and tables.

5. Geographic data for the map is fetched via get_city_coordinates(). Cache is updated if new cities are found.

## Usage & Gotchas

- Input file: By default, the dashboard reads Sales Dashboard Tata Motors.xlsx. If you want to use a different file:

  - Update the file_path variable in app.py.

  - Supported formats: .xlsx, .xls, .csv.

- City coordinates cache:

  - Stored in city_coordinates.json to reduce geocoding API calls.

  - If adding new cities in your dataset, this file will automatically update when the dashboard runs.

- Date formatting:

  - The Month column is formatted for display (Jan 2025) but for chart grouping, ensure your Date_of_Sold column is in datetime format.

- Dependencies: Ensure all required packages are installed (pandas, dash, plotly, geopy, babel, dash-bootstrap-components, openpyxl).

- Geopy API limits: Too many city geocoding requests may hit rate limits. Cached coordinates help avoid repeated API calls.
