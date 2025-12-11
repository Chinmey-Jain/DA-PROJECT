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
  ├── load_cache()           # Load city coordinates cache
  ├── load_sales_data()      # Load Excel/CSV file
  ├── clean_sales_data()     # Clean & preprocess data
  │       │
  │       └── get_city_coordinates()  # Used during map generation for lat/lon
  │
  └── create_app()
          │
          ├── get_app_layout()        # Builds layout: filters, KPIs, charts, table
          └── register_callbacks()   # Hooks callbacks for interactivity
                  │
                  ├── update_kpis()       # Updates total sales, revenue, top model, avg price
                  └── update_charts()     # Updates trend chart, model performance, fuel chart, geo map, table
```

## Function Documentation

| Function                                                                                        | Purpose                                                                                                                                           |
| ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `load_cache()`                                                                                  | Loads cached city coordinates from JSON. Initializes empty cache if file is missing or invalid.                                                   |
| `save_cache()`                                                                                  | Saves in-memory city coordinates to JSON.                                                                                                         |
| `get_city_coordinates(city)`                                                                    | Fetches latitude and longitude for a city, using cache if available.                                                                              |
| `load_sales_data(file_path)`                                                                    | Loads sales data from Excel or CSV.                                                                                                               |
| `clean_sales_data(df)`                                                                          | Cleans and preprocesses sales data: standardizes column names, converts dates, fills missing values, and creates a human-readable `Month` column. |
| `create_app(df_clean)`                                                                          | Creates the Dash app instance, sets layout, and registers callbacks.                                                                              |
| `get_app_layout(df_clean)`                                                                      | Defines the app layout: filters, KPIs, charts, table, footer.                                                                                     |
| `filter_data(df_clean, start_date, end_date, selected_models, selected_fuel, selected_segment)` | Filters the DataFrame based on user selections for KPIs, charts, and table.                                                                       |
| `register_callbacks(app, df_clean)`                                                             | Registers all Dash callbacks to update KPIs, charts, and table dynamically.                                                                       |
| `main()`                                                                                        | Entry point: loads cache, loads and cleans data, creates Dash app, and runs server.                                                               |

## Dash App Flow

1) main() is executed → loads city cache, loads sales data, cleans data, and creates the Dash app.

2) create_app() calls get_app_layout() to define the dashboard layout.

3) register_callbacks() hooks up all user interactions:

    - KPI updates → update_kpis()

    - Charts & table updates → update_charts()

4) User interacts with filters → callbacks are triggered → filtered data is passed to functions for charts, KPIs, and tables.

5) Geographic data for the map is fetched via get_city_coordinates(). Cache is updated if new cities are found.

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