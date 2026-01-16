import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

# -----------------------------
# Setup API client with caching and retry
# -----------------------------
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# -----------------------------
# Latitude/Longitude for location
# -----------------------------
latitude = 52.52   # Berlin
longitude = 13.41

# -----------------------------
# List of all desired hourly variables
# -----------------------------
HOURLY_VARIABLES = [
    "temperature_2m",
    "relativehumidity_2m",
    "dewpoint_2m",
    "apparent_temperature",
    "precipitation_probability",
    "precipitation",
    "rain",
    "showers",
    "snowfall",
    "snow_depth",
    "weathercode",
    "pressure_msl",
    "surface_pressure",
    "cloudcover",
    "cloudcover_low",
    "cloudcover_mid",
    "cloudcover_high",
    "visibility",
    "windspeed_10m",
    "windspeed_80m",
    "windspeed_120m",
    "windspeed_180m",
    "winddirection_10m",
    "winddirection_80m",
    "winddirection_120m",
    "winddirection_180m",
    "windgusts_10m",
    "temperature_80m",
    "temperature_120m",
    "temperature_180m"
]


# -----------------------------
# Fetch hourly weather data
# -----------------------------
url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": 52.52,
    "longitude": 13.41,
    "hourly": HOURLY_VARIABLES,
    "start_date": "2025-10-22",
    "end_date": "2026-01-20"  # one month
}


responses = openmeteo.weather_api(url, params=params)
response = responses[0]

# -----------------------------
# Process hourly data
# -----------------------------
hourly = response.Hourly()

# Create timestamp index
hourly_times = pd.date_range(
    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
    freq=pd.Timedelta(seconds=hourly.Interval()),
    inclusive="left"
)

# Build DataFrame dynamically
hourly_data = {"date": hourly_times}
for i, var in enumerate(HOURLY_VARIABLES):
    hourly_data[var] = hourly.Variables(i).ValuesAsNumpy()

hourly_df = pd.DataFrame(hourly_data)

# -----------------------------
# Print summary
# -----------------------------
print(f"Coordinates: {response.Latitude()}°N, {response.Longitude()}°E")
print(f"Elevation: {response.Elevation()} m")
print(f"Timezone offset to GMT: {response.UtcOffsetSeconds()}s")
print(f"\nHourly data shape: {hourly_df.shape}")
print(hourly_df.head())
hourly_df.to_csv("Data/Training set big.csv")

