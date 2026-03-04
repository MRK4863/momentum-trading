# Momentum Trading Dashboard

A Streamlit-based dashboard for analyzing stock momentum in the Indian stock market (NSE). The application tracks and visualizes momentum patterns over 6-day and 30-day periods using Yahoo Finance data.

## Features

- 📊 Real-time momentum analysis for NSE stocks
- 📈 Interactive price charts with Plotly
- 🔍 Advanced filtering and sorting with AgGrid
- 📥 Export data to CSV and Excel formats
- ☁️ Google Sheets integration for metadata management
- 🎨 Modern UI with customizable time periods

## Prerequisites

### Local Development
- Python 3.11+
- pip or conda

### Docker Deployment
- Docker 20.10+
- Docker Compose 2.0+ (optional, for easier management)

## Installation & Usage

### Option 1: Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Momentum_trading
   ```

2. **Create and activate virtual environment**
   ```bash
   # Using conda
   conda create -n finance python=3.11
   conda activate finance

   # Or using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run momentum_dashboard.py
   ```

5. **Access the dashboard**
   - Local: http://localhost:8501
   - Network: http://192.168.0.115:8501 (or your network IP)

### Option 2: Docker Deployment

#### Using Docker directly

1. **Build the Docker image**
   ```bash
   docker build -t momentum-trading-dashboard .
   ```

2. **Run the container**
   ```bash
   docker run -d \
     --name momentum-dashboard \
     -p 8501:8501 \
     momentum-trading-dashboard
   ```

3. **Access the dashboard**
   - Open browser: http://localhost:8501

4. **View logs**
   ```bash
   docker logs -f momentum-dashboard
   ```

5. **Stop the container**
   ```bash
   docker stop momentum-dashboard
   docker rm momentum-dashboard
   ```

#### Using Docker Compose (Recommended)

1. **Start the application**
   ```bash
   docker-compose up -d
   ```

2. **View logs**
   ```bash
   docker-compose logs -f
   ```

3. **Stop the application**
   ```bash
   docker-compose down
   ```

4. **Rebuild after changes**
   ```bash
   docker-compose up -d --build
   ```

## Configuration

### Google Sheets Integration

The application uses Google Sheets for metadata management. The service account credentials are hardcoded in the application for simplicity. To use your own Google Sheet:

1. Update `SHEET_ID` in `momentum_dashboard.py`
2. Update `GCP_SERVICE_ACCOUNT_INFO` with your service account credentials
3. Ensure the sheet has a worksheet named "METADATA" with the required columns

### Metadata Structure

The METADATA sheet should contain:
- `Instrument`: Stock symbol (without .NS suffix)
- `cap_category`: Market cap category
- `personal_rating`: Your personal rating
- `symbol_category`: Stock category classification

## Data Sources

- **Yahoo Finance (yfinance)**: Historical and real-time stock price data
- **Google Sheets**: Stock metadata and categorization
- **NSE India**: Stock symbols use `.NS` suffix for NSE market

## Architecture

```
momentum_dashboard.py       # Main Streamlit application
├── Data Collection
│   ├── Yahoo Finance API (yfinance)
│   └── Google Sheets (gspread)
├── Data Processing
│   ├── Momentum calculations (6-day & 30-day)
│   └── Price change analysis
└── Visualization
    ├── Interactive tables (AgGrid)
    ├── Price charts (Plotly)
    └── Export functionality (CSV/Excel)
```

## Key Metrics

- **Momentum %**: Percentage change over selected period
- **Price Change**: Absolute price difference
- **Trading Days**: Configurable lookback period (default: 4 days)

## Troubleshooting

### Failed Downloads
The application handles delisted/unavailable stocks gracefully. Failed symbols are displayed at the bottom of the dashboard.

### Port Already in Use
If port 8501 is already in use:
```bash
# Docker: Change port mapping
docker run -p 8502:8501 momentum-trading-dashboard

# Docker Compose: Edit docker-compose.yml
ports:
  - "8502:8501"

# Local: Streamlit will auto-select next available port
```

### Performance Issues
- Reduce the number of stocks in METADATA
- Use shorter analysis periods
- Enable caching in Streamlit

## Development

### File Structure
```
Momentum_trading/
├── momentum_dashboard.py    # Main application
├── METADATA.csv            # Fallback metadata (optional)
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker build instructions
├── docker-compose.yml     # Docker Compose configuration
├── .dockerignore         # Docker build exclusions
└── README.md             # This file
```

### Adding New Features

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally and with Docker
5. Submit a pull request

## License

MIT License - feel free to use and modify

## Support

For issues, questions, or contributions, please open an issue on GitHub.