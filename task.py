import os
import json
import logging
from typing import List, Optional

import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, dash_table
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from babel.numbers import format_currency

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

geolocator = Nominatim(user_agent="tata-sales-dashboard")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
coord_cache = {}

coordinate_json = "city_coordinates.json"
file_path = "Sales Dashboard Tata Motors.xlsx"
ALL_VALUE = "All"


def load_cache():
    """Load cached city coordinates from JSON file."""
    global coord_cache
    if os.path.exists(coordinate_json):
        try:
            with open(coordinate_json, "r") as f:
                coord_cache = json.load(f)
        except json.JSONDecodeError:
            coord_cache = {}
    else:
        coord_cache = {}


def save_cache():
    """Save cached city coordinates to JSON file."""
    with open(coordinate_json, "w") as f:
        json.dump(coord_cache, f, indent=4)


def get_city_coordinates(city):
    """
    Get latitude and longitude for a given city, using cache if available.
    param city: City name as string
    return: (latitude, longitude) tuple or (None, None) if not found
    """
    city_key = city.strip().lower()
    if city_key in coord_cache:
        return coord_cache[city_key]["LAT"], coord_cache[city_key]["LON"]
    try:
        loc = geocode(city + ", India")
        if loc:
            lat, lon = loc.latitude, loc.longitude
            coord_cache[city_key] = {"LAT": lat, "LON": lon}
            save_cache()
            return lat, lon
    except Exception as e:
        log.warning(f"Geocode failed for {city}: {e}")
    return None, None


def _normalize_list_selection(selection) -> List[str]:
    """Convert dropdown selections to a cleaned list, ignoring the 'All' option."""

    if selection in (None, ALL_VALUE):
        return []
    if isinstance(selection, list):
        if ALL_VALUE in selection:
            return []
        return [value for value in selection if value]
    return [selection]


def _normalize_scalar_selection(selection) -> Optional[str]:
    """Return None when 'All' (or empty) is chosen so filters can be skipped."""

    if not selection or selection == ALL_VALUE:
        return None
    return selection


def _build_empty_figure(title: str, subtitle: str) -> go.Figure:
    """Create a placeholder figure when no data matches the filters."""

    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": subtitle,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {"size": 14},
            }
        ],
        height=320,
    )
    return fig


def load_sales_data(file_path):
    """
    Load sales data from CSV or Excel file.
    param file_path: Path to the data file
    return: pandas DataFrame
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Use CSV or Excel.")
    return df


def clean_sales_data(df):
    """
    Clean and preprocess the sales data DataFrame.
    param df: Raw sales data DataFrame
    return: Cleaned sales data DataFrame
    """
    # Detect date column
    date_column = next((c for c in df.columns if "date" in c.lower()), "Date_of_Sold")
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

    # Standardize column names
    column_mapping = {
        "chassis": "Chassis_Number",
        "model": "Model",
        "variant": "Variant",
        "fuel": "Fuel_Type",
        "transmission": "Transmission",
        "price": "Ex_Showroom_Price",
        "city": "Dealer_City",
        "customer": "Customer_Segment",
    }
    for col in df.columns:
        col_lower = col.lower()
        for key, value in column_mapping.items():
            if key in col_lower and col != value:
                df.rename(columns={col: value}, inplace=True)

    # Required columns
    for col in ["Model", "Ex_Showroom_Price", "Date_of_Sold"]:
        if col not in df.columns:
            log.warning(f"Required column '{col}' not found.")

    # Clean numeric columns
    if "Ex_Showroom_Price" in df.columns and df["Ex_Showroom_Price"].dtype == "object":
        df["Ex_Showroom_Price"] = (
            df["Ex_Showroom_Price"].str.replace("[₹, ]", "", regex=True).astype(float)
        )

    # Fill missing values
    df["Fuel_Type"] = df.get("Fuel_Type", "Unknown").fillna("Unknown")
    df["Transmission"] = df.get("Transmission", "Unknown").fillna("Unknown")
    df["Customer_Segment"] = df.get("Customer_Segment", "Individual").fillna(
        "Individual"
    )

    # Create Month column
    df["Month"] = df["Date_of_Sold"].dt.to_period("M").astype(str)
    df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m").dt.strftime("%b %Y")

    return df


def create_app(df_clean):
    """
    Create and configure the Dash app.
    param df_clean: Cleaned sales data DataFrame
    return: Dash app instance
    """
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        title="Tata Motors Sales Dashboard",
    )
    # Layout
    app.layout = get_app_layout(df_clean)

    # Callbacks
    register_callbacks(app, df_clean)

    return app


def get_app_layout(df_clean):
    """
    Generate the layout for the Dash app.
    param df_clean: Cleaned sales data DataFrame
    return: Dash layout component
    """
    return dbc.Container(
        [
            html.H1("Tata Motors Sales Dashboard", className="text-center my-4"),
            # Filters
            dbc.Row(
                dbc.Col(
                    html.Div(
                        [
                            html.Label("Date Range"),
                            dcc.DatePickerRange(
                                id="date-range",
                                min_date_allowed=df_clean["Date_of_Sold"].min(),
                                max_date_allowed=df_clean["Date_of_Sold"].max(),
                                start_date=df_clean["Date_of_Sold"].min(),
                                end_date=df_clean["Date_of_Sold"].max(),
                            ),
                        ],
                        style={"display": "flex", "flexDirection": "column"},
                    ),
                    width=3,
                )
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Label("Model"),
                            dcc.Dropdown(
                                id="model-filter",
                                options=[{"label": "All Models", "value": "All"}]
                                + [
                                    {"label": m, "value": m}
                                    for m in sorted(df_clean["Model"].unique())
                                ],
                                value="All",
                                multi=True,
                            ),
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            html.Label("Fuel Type"),
                            dcc.Dropdown(
                                id="fuel-filter",
                                options=[{"label": "All", "value": "All"}]
                                + [
                                    {"label": f, "value": f}
                                    for f in sorted(df_clean["Fuel_Type"].unique())
                                ],
                                value="All",
                                multi=True,
                            ),
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            html.Label("Customer Segment"),
                            dcc.Dropdown(
                                id="segment-filter",
                                options=[{"label": "All", "value": "All"}]
                                + [
                                    {"label": s, "value": s}
                                    for s in sorted(
                                        df_clean["Customer_Segment"].unique()
                                    )
                                ],
                                value="All",
                                multi=True,
                            ),
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            html.Label("City"),
                            dcc.Dropdown(
                                id="city-filter",
                                options=[{"label": "All Cities", "value": "All"}]
                                + [
                                    {"label": c, "value": c}
                                    for c in sorted(df_clean["Dealer_City"].unique())
                                ],
                                value="All",
                                multi=True,
                            ),
                        ],
                        width=3,
                    ),
                ],
                className="mb-4",
            ),
            # KPI Cards
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [html.H4("Total Sales"), html.H2(id="total-sales-kpi")]
                            )
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4("Total Revenue"),
                                    html.H2(id="total-revenue-kpi"),
                                ]
                            )
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [html.H4("Top Model"), html.H2(id="top-model-kpi")]
                            )
                        ),
                        width=3,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [html.H4("Avg. Price"), html.H2(id="avg-price-kpi")]
                            )
                        ),
                        width=3,
                    ),
                ],
                className="mb-4",
            ),
            # Charts
            # Row 1
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(id="sales-trend-chart"), width=6, className="mb-4"
                    ),
                    dbc.Col(
                        dcc.Graph(id="avg-price-model-chart"), width=6, className="mb-4"
                    ),
                ]
            ),
            # Row 2
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(id="city-sales-volume-chart"),
                        width=6,
                        className="mb-4",
                    ),
                    dbc.Col(dcc.Graph(id="geo-map"), width=6, className="mb-4"),
                ]
            ),
            # Row 3
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Graph(id="city-avg-revenue-chart"),
                        width=4,
                        className="mb-4",
                    ),
                    dbc.Col(
                        dcc.Graph(id="model-performance"), width=4, className="mb-4"
                    ),
                    dbc.Col(dcc.Graph(id="fuel-type-chart"), width=4, className="mb-4"),
                ]
            ),
            # Data table
            dbc.Row(
                [
                    dbc.Col(
                        html.H3("Detailed Sales Data", style={"textAlign": "center"}),
                        width=12,
                    ),
                    dbc.Col(
                        dash_table.DataTable(
                            id="sales-data-table",
                            columns=[{"name": i, "id": i} for i in df_clean.columns],
                            page_size=10,
                            style_table={"overflowX": "auto"},
                            style_cell={
                                "textAlign": "left",
                                "padding": "10px",
                                "whiteSpace": "normal",
                                "height": "auto",
                            },
                            style_header={
                                "backgroundColor": "rgb(230, 230, 230)",
                                "fontWeight": "bold",
                            },
                            style_data_conditional=[
                                {
                                    "if": {"row_index": "odd"},
                                    "backgroundColor": "rgb(248, 248, 248)",
                                }
                            ],
                        ),
                        width=12,
                    ),
                ]
            ),
            # Footer
            dbc.Row(
                dbc.Col(
                    html.Div(
                        "Tata Motors Sales Dashboard | Made by Group 10",
                        style={
                            "textAlign": "center",
                            "padding": "10px 0",
                            "color": "#6c757d",
                            "borderTop": "1px solid #dee2e6",
                            "fontSize": "0.9rem",
                            "marginTop": "20px",
                        },
                    ),
                    width=12,
                )
            ),
        ],
        fluid=True,
    )


def filter_data(
    df_clean,
    start_date,
    end_date,
    selected_models,
    selected_fuel,
    selected_segment,
    selected_city,
):
    filtered_df = df_clean[
        (df_clean["Date_of_Sold"] >= pd.to_datetime(start_date))
        & (df_clean["Date_of_Sold"] <= pd.to_datetime(end_date))
    ]

    model_values = _normalize_list_selection(selected_models)
    if model_values:
        filtered_df = filtered_df[filtered_df["Model"].isin(model_values)]

    fuel_values = _normalize_list_selection(selected_fuel)
    if fuel_values:
        filtered_df = filtered_df[filtered_df["Fuel_Type"].isin(fuel_values)]

    segment_values = _normalize_list_selection(selected_segment)
    if segment_values:
        filtered_df = filtered_df[filtered_df["Customer_Segment"].isin(segment_values)]

    city_values = _normalize_list_selection(selected_city)
    if city_values:
        filtered_df = filtered_df[filtered_df["Dealer_City"].isin(city_values)]

    return filtered_df


def register_callbacks(app, df_clean):
    """
    Registers all callbacks for the Dash app.
    param app: Dash application instance
    param df_clean: Cleaned DataFrame with sales data
    return: None
    """

    # KPI Callback
    @app.callback(
        [
            Output("total-sales-kpi", "children"),
            Output("total-revenue-kpi", "children"),
            Output("top-model-kpi", "children"),
            Output("avg-price-kpi", "children"),
        ],
        [
            Input("date-range", "start_date"),
            Input("date-range", "end_date"),
            Input("model-filter", "value"),
            Input("fuel-filter", "value"),
            Input("segment-filter", "value"),
            Input("city-filter", "value"),
        ],
    )
    def update_kpis(
        start_date,
        end_date,
        selected_models,
        selected_fuel,
        selected_segment,
        selected_city,
    ):
        """
        Update the KPI values based on user selections.
        param start_date: Start date for filtering
        param end_date: End date for filtering
        param selected_models: List of selected models or "All"
        param selected_fuel: Selected fuel type or "All"
        param selected_segment: Selected customer segment or "All"
        param selected_city: List of selected cities or "All"
        return: Tuple containing updated KPI values
        """
        filtered_df = filter_data(
            df_clean,
            start_date,
            end_date,
            selected_models,
            selected_fuel,
            selected_segment,
            selected_city,
        )
        total_sales = len(filtered_df)
        total_revenue = (
            filtered_df["Ex_Showroom_Price"].sum() if not filtered_df.empty else 0
        )
        top_model = (
            filtered_df["Model"].value_counts().idxmax()
            if not filtered_df.empty
            else "N/A"
        )
        avg_price = (
            filtered_df["Ex_Showroom_Price"].mean() if not filtered_df.empty else 0
        )
        if pd.isna(avg_price):
            avg_price = 0
        return (
            f"{total_sales:,}",
            format_currency(total_revenue, "INR", locale="en_IN"),
            top_model,
            format_currency(avg_price, "INR", locale="en_IN"),
        )

    # Charts & Table Callback
    @app.callback(
        [
            Output("sales-trend-chart", "figure"),
            Output("avg-price-model-chart", "figure"),
            Output("geo-map", "figure"),
            Output("city-sales-volume-chart", "figure"),
            Output("city-avg-revenue-chart", "figure"),
            Output("model-performance", "figure"),
            Output("fuel-type-chart", "figure"),
            Output("sales-data-table", "data"),
        ],
        [
            Input("date-range", "start_date"),
            Input("date-range", "end_date"),
            Input("model-filter", "value"),
            Input("fuel-filter", "value"),
            Input("segment-filter", "value"),
            Input("city-filter", "value"),
        ],
    )
    def update_charts(
        start_date,
        end_date,
        selected_models,
        selected_fuel,
        selected_segment,
        selected_city,
    ):
        """
        Update charts and data table based on user selections.
        param start_date: Start date for filtering
        param end_date: End date for filtering
        param selected_models: List of selected models or "All"
        param selected_fuel: Selected fuel type or "All"
        param selected_segment: Selected customer segment or "All"
        param selected_city: List of selected cities or "All"
        return: Tuple containing updated chart figures and table data
        """
        filtered_df = filter_data(
            df_clean,
            start_date,
            end_date,
            selected_models,
            selected_fuel,
            selected_segment,
            selected_city,
        )

        if filtered_df.empty:
            empty_message = "No data available for the selected filters"
            placeholder = _build_empty_figure("No Data", empty_message)
            return (
                placeholder,  # sales-trend-chart
                placeholder,  # avg-price-model-chart
                _build_empty_figure("Sales by City (Revenue)", empty_message),
                _build_empty_figure("Sales by City (Volume)", empty_message),
                _build_empty_figure("City-wise Avg Revenue", empty_message),
                placeholder,  # model-performance
                placeholder,  # fuel-type-chart
                [],
            )
        # City-wise Average Revenue Chart
        city_data = (
            filtered_df.groupby("Dealer_City")
            .agg(
                Total_Sales=("Model", "count"),
                Total_Revenue=("Ex_Showroom_Price", "sum"),
            )
            .reset_index()
        )
        city_avg_rev_fig = _build_empty_figure("City-wise Avg Revenue", "No city data")
        # City-wise Average Revenue Chart
        city_data = (
            filtered_df.groupby("Dealer_City")
            .agg(
                Total_Sales=("Model", "count"),
                Total_Revenue=("Ex_Showroom_Price", "sum"),
            )
            .reset_index()
        )
        if not city_data.empty:
            city_data["Avg_Revenue"] = (
                city_data["Total_Revenue"] / city_data["Total_Sales"]
            )
            city_avg_sorted = city_data.sort_values("Avg_Revenue", ascending=False)
            city_avg_rev_fig = go.Figure(
                go.Bar(
                    x=city_avg_sorted["Dealer_City"],
                    y=city_avg_sorted["Avg_Revenue"],
                    marker_color="#43A047",
                    text=city_avg_sorted["Avg_Revenue"].apply(
                        lambda v: format_currency(v, "INR", locale="en_IN")
                    ),
                    textposition="outside",
                    name="Avg Revenue",
                )
            )
            city_avg_rev_fig.update_layout(
                title="City-wise Avg Revenue",
                title_x=0.5,
                xaxis_title="City",
                yaxis=dict(
                    title="Avg Revenue per Sale (₹ L)",
                    tickprefix="₹ ",
                    separatethousands=True,
                    tickformat="~s",
                    tickvals=[int(lakh) * 100000 for lakh in [0, 2, 4, 6, 8, 10, 12, 14, 16]],
                    ticktext=["0", "2", "4", "6", "8", "10", "12", "14", "16"],
                ),
                margin=dict(l=40, r=20, t=60, b=40),
            )

        # Sales Trend
        monthly_data = (
            filtered_df.groupby(pd.Grouper(key="Date_of_Sold", freq="M"))
            .agg(
                Total_Sales=("Model", "count"),
                Total_Revenue=("Ex_Showroom_Price", "sum"),
            )
            .reset_index()
        )
        monthly_data["Revenue_Crore"] = monthly_data["Total_Revenue"] / 10_000_000
        monthly_data["Revenue_Label"] = monthly_data["Total_Revenue"].apply(
            lambda value: format_currency(value, "INR", locale="en_IN")
        )
        trend_fig = go.Figure()
        trend_fig.add_trace(
            go.Scatter(
                x=monthly_data["Date_of_Sold"],
                y=monthly_data["Total_Sales"],
                name="Sales Volume",
                line=dict(color="#1f77b4", width=3),
            )
        )
        trend_fig.add_trace(
            go.Scatter(
                x=monthly_data["Date_of_Sold"],
                y=monthly_data["Revenue_Crore"],
                name="Revenue (₹ Cr)",
                yaxis="y2",
                line=dict(color="#ff7f0e", width=3),
                customdata=monthly_data["Revenue_Label"],
                hovertemplate="Month: %{x|%b %Y}<br>Revenue: %{customdata}<extra></extra>",
            )
        )
        trend_fig.update_layout(
            title="Monthly Sales Performance",
            title_x=0.5,
            xaxis_title="Month",
            yaxis_title="Sales Volume",
            yaxis2=dict(title="Revenue (₹ Cr)", overlaying="y", side="right"),
            hovermode="x unified",
        )

        # Avg Price by Model
        avg_price_data = (
            filtered_df.groupby("Model")["Ex_Showroom_Price"].mean().reset_index()
        )
        avg_price_data = avg_price_data.sort_values(
            "Ex_Showroom_Price", ascending=False
        )
        avg_price_text = avg_price_data["Ex_Showroom_Price"].apply(
            lambda value: format_currency(value, "INR", locale="en_IN")
        )
        avg_price_fig = go.Figure(
            go.Bar(
                x=avg_price_data["Model"],
                y=avg_price_data["Ex_Showroom_Price"],
                marker_color="#8E24AA",
                name="Avg Price",
                text=avg_price_text,
                textposition="outside",
                customdata=avg_price_text,
                hovertemplate="Model: %{x}<br>Avg Price: %{customdata}<extra></extra>",
            )
        )
        avg_price_fig.update_layout(
            title="Average Price by Model",
            title_x=0.5,
            xaxis_title="Model",
            yaxis=dict(
                title="Average Price (₹ L)",
                tickprefix="₹ ",
                separatethousands=True,
                tickformat="~s",
                tickvals=[100000, 500000, 1000000, 1500000, 2000000, 2500000],
                ticktext=["1", "5", "10", "15", "20", "25"],
            ),
        )

        # Model Performance
        model_perf = (
            filtered_df.groupby("Model")
            .agg(
                Total_Sales=("Model", "count"),
                Total_Revenue=("Ex_Showroom_Price", "sum"),
            )
            .reset_index()
            .sort_values("Total_Sales", ascending=False)
        )
        model_fig = go.Figure(
            go.Bar(x=model_perf["Model"], y=model_perf["Total_Sales"])
        )
        model_fig.update_layout(
            title="Sales by Model",
            title_x=0.5,
            xaxis_title="Model",
            yaxis_title="Number of Sales",
        )

        # Fuel Type
        fuel_data = filtered_df["Fuel_Type"].value_counts().reset_index()
        fuel_data.columns = ["Fuel_Type", "Count"]
        fuel_fig = go.Figure(
            go.Pie(labels=fuel_data["Fuel_Type"], values=fuel_data["Count"], hole=0.5)
        )
        fuel_fig.update_layout(title="Sales by Fuel Type", title_x=0.5)

        # Geo Map & City Sales Volume
        city_data = (
            filtered_df.groupby("Dealer_City")
            .agg(
                Total_Sales=("Model", "count"),
                Total_Revenue=("Ex_Showroom_Price", "sum"),
            )
            .reset_index()
        )
        # Geo Map
        if city_data.empty:
            geo_fig = _build_empty_figure(
                "Sales by City (Revenue)", "No city level data for this selection"
            )
            city_sales_volume_fig = _build_empty_figure(
                "Sales by City (Volume)", "No city level data for this selection"
            )
        else:
            city_data["Revenue_Label"] = city_data["Total_Revenue"].apply(
                lambda value: format_currency(value, "INR", locale="en_IN")
            )
            city_data["Revenue_Crore"] = city_data["Total_Revenue"] / 10_000_000
            coordinate_pairs = [
                coords if coords and len(coords) == 2 else (None, None)
                for coords in city_data["Dealer_City"].apply(get_city_coordinates)
            ]
            city_data["lat"] = [pair[0] for pair in coordinate_pairs]
            city_data["lon"] = [pair[1] for pair in coordinate_pairs]
            city_data = city_data.dropna(subset=["lat", "lon"])
            if city_data.empty:
                geo_fig = _build_empty_figure(
                    "Sales by City (Revenue)", "Unable to resolve city coordinates"
                )
                city_sales_volume_fig = _build_empty_figure(
                    "Sales by City (Volume)", "Unable to resolve city coordinates"
                )
            else:
                sales_counts = city_data["Total_Sales"]
                min_sales = sales_counts.min()
                max_sales = sales_counts.max()
                if max_sales == min_sales:
                    marker_sizes = [18] * len(sales_counts)
                else:
                    marker_sizes = (
                        10 + ((sales_counts - min_sales) / (max_sales - min_sales)) * 20
                    )
                geo_fig = go.Figure(
                    go.Scattergeo(
                        lat=city_data["lat"],
                        lon=city_data["lon"],
                        text=(
                            city_data["Dealer_City"]
                            + "<br>Sales: "
                            + city_data["Total_Sales"].astype(str)
                            + "<br>Revenue: "
                            + city_data["Revenue_Label"]
                        ),
                        mode="markers",
                        marker=dict(
                            size=marker_sizes,
                            color=city_data["Revenue_Crore"],
                            colorscale="Viridis",
                            showscale=True,
                            colorbar_title="Revenue (₹ Cr)",
                        ),
                    )
                )
                geo_fig.update_layout(
                    title="Sales by City (Revenue)",
                    title_x=0.5,
                    geo=dict(
                        scope="asia",
                        projection_type="mercator",
                        center=dict(lat=22.0, lon=80.0),
                        lataxis=dict(range=[6, 38]),
                        lonaxis=dict(range=[68, 98]),
                        showland=True,
                        landcolor="rgb(240,240,240)",
                        showcountries=True,
                        countrycolor="rgb(200,200,200)",
                        showcoastlines=True,
                        coastlinecolor="rgb(180,180,180)",
                    ),
                    margin=dict(l=0, r=0, t=60, b=0),
                )
                # City Sales Volume Bar Chart
                city_sales_sorted = city_data.sort_values(
                    "Total_Sales", ascending=False
                )
                city_sales_volume_fig = go.Figure(
                    go.Bar(
                        x=city_sales_sorted["Dealer_City"],
                        y=city_sales_sorted["Total_Sales"],
                        marker_color="#1976D2",
                        text=city_sales_sorted["Total_Sales"],
                        textposition="outside",
                        name="Sales Volume",
                    )
                )
                city_sales_volume_fig.update_layout(
                    title="Sales by City (Volume)",
                    title_x=0.5,
                    xaxis_title="City",
                    yaxis_title="Number of Sales",
                    margin=dict(l=40, r=20, t=60, b=40),
                )

        # Table Data
        table_data = filtered_df.copy()
        table_data["Date_of_Sold"] = table_data["Date_of_Sold"].dt.strftime("%d %b %Y")
        table_data = table_data.to_dict("records")

        return (
            trend_fig,  # sales-trend-chart
            avg_price_fig,  # avg-price-model-chart
            geo_fig,  # geo-map (Sales by City Revenue)
            city_sales_volume_fig,  # city-sales-volume-chart
            city_avg_rev_fig,  # city-avg-revenue-chart
            model_fig,  # model-performance
            fuel_fig,  # fuel-type-chart
            table_data,
        )


def initialize_dashboard() -> Dash:
    """Build and return the Dash application ready for WSGI/CLI hosts."""

    load_cache()
    df = load_sales_data(file_path)
    df_clean = clean_sales_data(df)
    return create_app(df_clean)


app = initialize_dashboard()
server = app.server


def main():
    """Main function to run the Dash app locally."""

    app.run(debug=True)


if __name__ == "__main__":
    main()
