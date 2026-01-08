import streamlit as st
import yfinance as yf
import datetime
import pandas as pd
import plotly.graph_objects as go
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

def yt_finance_historical_data(symbols_list, momentum_back_date=4):
    """
    Download historical data and calculate momentum metrics
    
    Args:
        symbols_list: List of stock symbols
        momentum_back_date: Number of trading days back to calculate momentum from (default: 4)
    """
    import datetime
    import pandas as pd
    
    end = datetime.date.today()
    start = end - datetime.timedelta(days=40)  # enough buffer for 30 trading days
    
    with st.spinner(f'Downloading data for {len(symbols_list)} symbols...'):
        df = yf.download(symbols_list, start=start, end=end, interval="1d")
    
    # Check if today's date is in the dataframe
    today = datetime.date.today()
    df_dates = [d.date() if hasattr(d, 'date') else d for d in df.index]
    has_today_data = today in df_dates
    
    # If today's data is not present, fetch intraday data
    if not has_today_data:
        with st.spinner('Fetching today\'s intraday data...'):
            df_today = yf.download(symbols_list, period="1d", interval="15m")
            
            if not df_today.empty:
                # Get the most recent close price (last row)
                latest_close = df_today["Close"].iloc[-1]
                
                # Add today's data to the main dataframe
                today_timestamp = pd.Timestamp(today)
                
                # Create new row with today's close prices
                if len(symbols_list) == 1:
                    # Single symbol - latest_close is a scalar
                    df.loc[today_timestamp, 'Close'] = latest_close
                else:
                    # Multiple symbols - latest_close is a Series
                    for symbol in symbols_list:
                        df.loc[today_timestamp, ('Close', symbol)] = latest_close[symbol]
    
    # Get last 30 trading days
    last_30 = df.tail(40)["Close"]
    last_30 = last_30.T
    last_30 = last_30.reset_index()
    last_30['SYMBOL'] = last_30['Ticker'].str.replace('.NS', '')

    # Fill NaN values in date columns starting from the second date column
    date_cols_30 = last_30.columns[1:-1]  # exclude 'Ticker' and 'SYMBOL'
    last_30[date_cols_30[1:]] = last_30[date_cols_30[1:]].ffill(axis=1)

    # Calculate momentum for 30-day period
    last_date_30 = date_cols_30[-1]
    fourth_from_last_30 = date_cols_30[-momentum_back_date]
    last_30['DIFF'] = last_30[last_date_30] - last_30[fourth_from_last_30]
    last_30['Diff_percent'] = last_30['DIFF'] * 100 / last_30[fourth_from_last_30]
    last_30 = last_30.sort_values(by='Diff_percent', ascending=False)

    # Derive 7-day view from last_30 (last 7 trading days)
    last_7 = df.tail(10)["Close"]  # buffer for 7 trading days
    last_7 = last_7.T
    last_7 = last_7.reset_index()
    last_7['SYMBOL'] = last_7['Ticker'].str.replace('.NS', '')

    # Fill NaN values in date columns
    date_cols_7 = last_7.columns[1:-1]  # exclude 'Ticker' and 'SYMBOL'
    last_7[date_cols_7[1:]] = last_7[date_cols_7[1:]].ffill(axis=1)

    # Calculate momentum for 7-day period
    last_date_7 = date_cols_7[-1]
    fourth_from_last_7 = date_cols_7[-momentum_back_date]
    last_7['DIFF'] = last_7[last_date_7] - last_7[fourth_from_last_7]
    last_7['Diff_percent'] = last_7['DIFF'] * 100 / last_7[fourth_from_last_7]
    last_7 = last_7.sort_values(by='Diff_percent', ascending=False)

    return last_7, last_30

@st.cache_data
def load_metadata():
    """Load instrument metadata"""
    try:
        df_metadata = pd.read_csv("METADATA.csv")
    except FileNotFoundError:
        try:
            df_metadata = pd.read_csv("instruments_data.csv")
        except FileNotFoundError:
            st.error("Could not find METADATA.csv or instruments_data.csv")
            return None
    return df_metadata

def apply_custom_css():
    """Apply CSS with proper light/dark mode support"""
    st.markdown("""
        <style>
        /* Base styling */
        .stDataFrame {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }
        
        /* Light mode colors (default) */
        :root {
            --color-green-bg: #C6EFCE;
            --color-green-text: #006100;
            --color-red-bg: #FFC7CE;
            --color-red-text: #9C0006;
            --color-yellow-bg: #FFEB9C;
            --color-yellow-text: #9C6500;
        }
        
        /* Dark mode colors */
        @media (prefers-color-scheme: dark) {
            :root {
                --color-green-bg: #1B5E20;
                --color-green-text: #A5D6A7;
                --color-red-bg: #B71C1C;
                --color-red-text: #EF9A9A;
                --color-yellow-bg: #F57F17;
                --color-yellow-text: #FFF59D;
            }
        }
        
        /* For Streamlit's dark theme specifically */
        [data-testid="stAppViewContainer"][class*="dark"] {
            --color-green-bg: #1B5E20;
            --color-green-text: #A5D6A7;
            --color-red-bg: #B71C1C;
            --color-red-text: #EF9A9A;
            --color-yellow-bg: #F57F17;
            --color-yellow-text: #FFF59D;
        }
        </style>
    """, unsafe_allow_html=True)

def create_aggrid_table(display_df, price_cols):
    """Create AgGrid table with conditional formatting and filtering"""
    
    # Get date columns first (needed for price comparison logic)
    date_columns = [col for col in display_df.columns if str(col).strip().startswith('20') and '-' in str(col)]
    
    # JavaScript code for conditional formatting
    cellStyle_momentum = JsCode("""
    function(params) {
        if (params.value == null || isNaN(params.value)) {
            return {};
        }
        if (params.value > 0) {
            return {
                'backgroundColor': '#2E7D32',
                'color': '#E8F5E9',
                'fontWeight': 'bold'
            };
        } else {
            return {
                'backgroundColor': '#C62828',
                'color': '#FFEBEE',
                'fontWeight': 'bold'
            };
        }
    }
    """)
    
    # Prepare a mapping of date columns to their previous column for price comparison
    date_col_pairs = {}
    for i in range(1, len(date_columns)):
        date_col_pairs[date_columns[i]] = date_columns[i-1]
    
    # Create individual cellStyle functions for each date column
    def create_price_cellstyle(current_col, prev_col):
        return JsCode(f"""
        function(params) {{
            if (params.value == null || isNaN(params.value)) {{
                return {{}};
            }}
            
            var prevValue = params.data['{prev_col}'];
            
            if (prevValue != null && !isNaN(prevValue)) {{
                if (params.value > prevValue) {{
                    return {{
                        'backgroundColor': '#2E7D32',
                        'color': '#E8F5E9',
                        'fontWeight': 'bold'
                    }};
                }} else if (params.value < prevValue) {{
                    return {{
                        'backgroundColor': '#C62828',
                        'color': '#FFEBEE',
                        'fontWeight': 'bold'
                    }};
                }} else {{
                    return {{
                        'backgroundColor': '#F9A825',
                        'color': '#FFFDE7',
                        'fontWeight': 'bold'
                    }};
                }}
            }}
            return {{}};
        }}
        """)
    
    # Build grid options
    gb = GridOptionsBuilder.from_dataframe(display_df)
    
    # Configure default column properties
    gb.configure_default_column(
        filterable=True,
        sortable=True,
        resizable=True,
        editable=False
    )
    
    # Configure specific columns
    gb.configure_column("Rank", pinned='left', width=80)
    gb.configure_column("Stock", pinned='left', width=150)
    gb.configure_column("Momentum %", cellStyle=cellStyle_momentum, width=130)
    gb.configure_column("Price Change", width=130)
    
    # Configure metadata columns if they exist
    if "Cap Category" in display_df.columns:
        gb.configure_column("Cap Category", width=120)
    if "Rating" in display_df.columns:
        gb.configure_column("Rating", width=100)
    if "Category" in display_df.columns:
        gb.configure_column("Category", width=120)
    
    # Configure price columns with conditional formatting
    # First date column (no previous column to compare)
    if len(date_columns) > 0:
        gb.configure_column(date_columns[0], width=110)
    
    # Subsequent date columns with comparison to previous
    for current_col, prev_col in date_col_pairs.items():
        cellStyle = create_price_cellstyle(current_col, prev_col)
        gb.configure_column(current_col, cellStyle=cellStyle, width=110)
    
    # Configure grid options
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=50)
    gb.configure_side_bar()
    gb.configure_selection(selection_mode="single", use_checkbox=False)
    
    gridOptions = gb.build()
    
    # Display AgGrid
    grid_response = AgGrid(
        display_df,
        gridOptions=gridOptions,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        fit_columns_on_grid_load=False,
        theme='streamlit',  # Use streamlit theme for better integration
        height=600,
        allow_unsafe_jscode=True,
        enable_enterprise_modules=False
    )
    
    return grid_response

def main():
    st.set_page_config(
        page_title="Momentum Trading Dashboard",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    apply_custom_css()
    
    st.title("📈 Momentum Trading Dashboard")
    st.markdown("Track stock momentum performance over 7-day and 30-day periods")
    
    # Load metadata
    df_metadata = load_metadata()
    if df_metadata is None:
        return
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    # Momentum lookback parameter
    momentum_back_date = st.sidebar.slider(
        "Momentum Lookback Days",
        min_value=2,
        max_value=10,
        value=4,
        help="Number of trading days back to calculate momentum from"
    )
    
    # Display options
    period_option = st.sidebar.selectbox(
        "Select Period",
        ["7-day momentum", "30-day momentum"],
        index=0  # Default to 7-day momentum
    )
    
    # Initialize session state
    if 'analysis_data' not in st.session_state:
        st.session_state.analysis_data = None
    if 'last_momentum_back_date' not in st.session_state:
        st.session_state.last_momentum_back_date = momentum_back_date
    
    # Check if parameters changed
    params_changed = (st.session_state.last_momentum_back_date != momentum_back_date)
    
    # Process button
    run_analysis = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    # Clear cache button
    if st.sidebar.button("🔄 Clear Cache"):
        st.cache_data.clear()
        st.session_state.analysis_data = None
        st.success("Cache cleared!")
        st.rerun()
    
    # Run analysis if button clicked or if parameters changed and we have data
    if run_analysis or params_changed:
        if params_changed:
            st.session_state.analysis_data = None
        
        # Process data
        sample_list = df_metadata["Instrument"].to_list()
        sample_list = [sample + ".NS" for sample in sample_list]
        
        # Download and process data
        last_7, last_30 = yt_finance_historical_data(sample_list, momentum_back_date)
        
        # Merge with metadata
        merged_df_7 = pd.merge(df_metadata, last_7, left_on='Instrument', right_on='SYMBOL', how="left")
        merged_df_7 = merged_df_7.sort_values(by='Diff_percent', ascending=False)
        
        merged_df_30 = pd.merge(df_metadata, last_30, left_on='Instrument', right_on='SYMBOL', how="left")
        merged_df_30 = merged_df_30.sort_values(by='Diff_percent', ascending=False)
        
        # Store in session state
        st.session_state.analysis_data = {
            'merged_df_7': merged_df_7,
            'merged_df_30': merged_df_30,
            'momentum_back_date': momentum_back_date
        }
        st.session_state.last_momentum_back_date = momentum_back_date
    
    # Display results if data exists in session state
    if st.session_state.analysis_data is not None:
        # Get data from session state
        merged_df_7 = st.session_state.analysis_data['merged_df_7']
        merged_df_30 = st.session_state.analysis_data['merged_df_30']
        stored_momentum_back_date = st.session_state.analysis_data['momentum_back_date']
        
        # Display results based on selection
        if period_option == "7-day momentum":
            st.header("📊 7-Day Momentum Leaders")
            merged_df = merged_df_7
            valid_data = merged_df_7.dropna(subset=['Diff_percent'])
        else:  # 30-day momentum
            st.header("📈 30-Day Momentum Leaders")
            merged_df = merged_df_30
            valid_data = merged_df_30.dropna(subset=['Diff_percent'])
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Stocks", len(merged_df))
        with col2:
            st.metric("Valid Data", len(valid_data))
        with col3:
            avg_momentum = valid_data['Diff_percent'].mean()
            st.metric("Avg Momentum", f"{avg_momentum:.2f}%")
        with col4:
            top_performer = valid_data.iloc[0]['Diff_percent'] if len(valid_data) > 0 else 0
            st.metric("Top Performer", f"{top_performer:.2f}%")
        
        # Display all performers with price history
        price_cols = [col for col in valid_data.columns if str(col).strip().startswith('20') and '-' in str(col)]
        
        # Include additional metadata columns
        display_columns = ['Instrument', 'Diff_percent', 'DIFF', 'cap_category', 'personal_rating', 'symbol_category'] + price_cols
        
        # Filter columns that actually exist in the dataframe
        display_columns = [col for col in display_columns if col in valid_data.columns]
        
        display_df = valid_data[display_columns].copy()
        display_df['Diff_percent'] = display_df['Diff_percent'].round(2)
        display_df['DIFF'] = display_df['DIFF'].round(2)
        
        # Add rank column based on Diff_percent (already sorted in descending order)
        display_df.insert(0, 'Rank', range(1, len(display_df) + 1))
        
        # Round price columns
        for col in price_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(2)
        
        # Rename columns for better display
        column_names = ['Rank', 'Stock', 'Momentum %', 'Price Change']
        
        # Add metadata column names if they exist
        if 'cap_category' in display_df.columns:
            column_names.append('Cap Category')
        if 'personal_rating' in display_df.columns:
            column_names.append('Rating')
        if 'symbol_category' in display_df.columns:
            column_names.append('Category')
        
        # Add date columns
        for col in price_cols:
            if col in display_df.columns:
                # Extract date from column name for cleaner display
                date_str = str(col).split()[0]  # Get just the date part
                column_names.append(date_str)
        
        display_df.columns = column_names[:len(display_df.columns)]
        
        # Create tabs for table and chart
        tab_table, tab_chart = st.tabs(["📊 Data Table", "📈 Price Charts"])
        
        with tab_table:
            st.info("💡 **Pro Tip:** Use the filter icon in column headers to filter data, click columns to sort, and use the sidebar menu (☰) for advanced options!")
            
            # Display AgGrid table with filtering
            grid_response = create_aggrid_table(display_df, price_cols)
            
            # Show filtered results count
            if grid_response is not None and 'data' in grid_response:
                filtered_df = pd.DataFrame(grid_response['data'])
                if len(filtered_df) < len(display_df):
                    st.success(f"🔍 Showing {len(filtered_df)} of {len(display_df)} stocks (filtered)")
        
        with tab_chart:
            # Stock selection
            stock_options = display_df['Stock'].tolist()
            selected_stock = st.selectbox(
                "Select Stock",
                stock_options,
                key="stock_selector"
            )
            
            if selected_stock:
                # Get price data for selected stock
                stock_row = display_df[display_df['Stock'] == selected_stock].iloc[0]
                
                # Extract all available date columns and prices
                all_date_columns = [col for col in display_df.columns if str(col).strip().startswith('20') and '-' in str(col)]
                all_prices = [stock_row[col] for col in all_date_columns]
                
                # Convert date strings to datetime objects
                date_objects = [pd.to_datetime(date) for date in all_date_columns]
                
                # Get available date range
                min_date = min(date_objects).date()
                max_date = max(date_objects).date()
                
                # Calculate default date range (last 7 days)
                default_end = max_date
                default_start = max_date - datetime.timedelta(days=6)
                if default_start < min_date:
                    default_start = min_date
                
                # Date range selector
                st.write("### Select Date Range")
                col_date1, col_date2 = st.columns(2)
                
                with col_date1:
                    start_date = st.date_input(
                        "Start Date",
                        value=default_start,
                        min_value=min_date,
                        max_value=max_date,
                        key="start_date"
                    )
                
                with col_date2:
                    end_date = st.date_input(
                        "End Date",
                        value=default_end,
                        min_value=min_date,
                        max_value=max_date,
                        key="end_date"
                    )
                
                # Validate date range
                if start_date > end_date:
                    st.error("❌ Start date must be before or equal to end date!")
                    return
                
                # Filter data based on selected date range
                filtered_dates = []
                filtered_prices = []
                
                for date_str, price, date_obj in zip(all_date_columns, all_prices, date_objects):
                    if start_date <= date_obj.date() <= end_date:
                        filtered_dates.append(date_str)
                        filtered_prices.append(price)
                
                if len(filtered_dates) == 0:
                    st.warning("⚠️ No data available for the selected date range.")
                    return
                
                # Create dataframe for chart
                chart_data = pd.DataFrame({
                    'Date': filtered_dates,
                    'Price': filtered_prices
                })
                
                # Display metrics
                current_price = filtered_prices[-1] if filtered_prices else 0
                first_price = filtered_prices[0] if filtered_prices else 0
                price_change = current_price - first_price
                price_change_pct = (price_change / first_price * 100) if first_price > 0 else 0
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("Current Price", f"₹{current_price:.2f}")
                with metric_col2:
                    st.metric("Change", f"₹{price_change:.2f}", f"{price_change_pct:.2f}%")
                with metric_col3:
                    momentum_pct = stock_row['Momentum %']
                    st.metric("Momentum", f"{momentum_pct:.2f}%")
                
                # Create combined bar and line chart
                fig = go.Figure()
                
                # Add bar chart
                fig.add_trace(go.Bar(
                    x=chart_data['Date'],
                    y=chart_data['Price'],
                    name='Price',
                    marker_color='lightblue',
                    opacity=0.6
                ))
                
                # Add line chart overlay with bright red
                fig.add_trace(go.Scatter(
                    x=chart_data['Date'],
                    y=chart_data['Price'],
                    name='Trend',
                    mode='lines+markers',
                    line=dict(color='#FF0000', width=3),
                    marker=dict(size=8, color='#FF0000')
                ))
                
                # Calculate dynamic y-axis range to show fluctuations
                min_price = min(filtered_prices)
                max_price = max(filtered_prices)
                price_range = max_price - min_price
                
                # Add 5% padding to top and bottom for better visualization
                padding = price_range * 0.05 if price_range > 0 else max_price * 0.05
                y_min = min_price - padding
                y_max = max_price + padding
                
                fig.update_layout(
                    title=f"{selected_stock} Price Trend ({start_date} to {end_date})",
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    hovermode='x unified',
                    height=500,
                    showlegend=True,
                    xaxis=dict(tickangle=-45),
                    yaxis=dict(range=[y_min, y_max])
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show date range info
                num_days = len(filtered_dates)
                st.info(f"📅 Showing **{num_days} trading days** from {start_date} to {end_date}")
    else:
        st.info("👆 Select your preferred period and click 'Run Analysis' to start processing the momentum data.")

if __name__ == "__main__":
    main()