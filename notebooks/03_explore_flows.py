# %% [markdown]
# # Network Flow Data Exploration
#
# This notebook explores the network flow data extracted from Zeek connection logs.
#
# ## Objectives
# 1. Load and examine the flow data structure
# 2. Analyze basic traffic statistics
# 3. Explore temporal patterns
# 4. Visualize flow characteristics
# 5. Identify potential anomalies

# %%
# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

# Configure plotting
plt.style.use('default')
sns.set_palette('husl')
warnings.filterwarnings('ignore')

# Set up paths
DATA_DIR = Path('../data')
PROCESSED_DIR = DATA_DIR / 'processed'
INTERIM_DIR = DATA_DIR / 'interim'

print("✓ Libraries imported successfully")

# %% [markdown]
# ## 1. Data Loading and Basic Structure

# %%
# Load the processed flow data
flows_file = PROCESSED_DIR / 'flows_v0.parquet'

if flows_file.exists():
    df = pd.read_parquet(flows_file)
    print(f"✓ Loaded {len(df):,} flow records")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
else:
    print(f"❌ Flow data not found at {flows_file}")
    print("Run 'python scripts/zeek2parquet.py' first to generate flow data")

# %%
# Examine data structure
print("Dataset Shape:", df.shape)
print("\nColumn Types:")
print(df.dtypes.value_counts())

print("\nFirst few rows:")
df.head()

# %%
# Check for missing values
missing_info = df.isnull().sum()
missing_info = missing_info[missing_info > 0].sort_values(ascending=False)

if len(missing_info) > 0:
    print("Missing values:")
    for col, count in missing_info.items():
        pct = count / len(df) * 100
        print(f"  {col}: {count:,} ({pct:.1f}%)")
else:
    print("✓ No missing values found")

# %% [markdown]
# ## 2. Basic Traffic Statistics

# %%
# Protocol distribution
print("Protocol Distribution:")
proto_counts = df['proto'].value_counts()
for proto, count in proto_counts.items():
    pct = count / len(df) * 100
    print(f"  {proto}: {count:,} ({pct:.1f}%)")

# Visualize protocol distribution
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
proto_counts.plot(kind='bar')
plt.title('Protocol Distribution')
plt.ylabel('Number of Flows')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
proto_counts.plot(kind='pie', autopct='%1.1f%%')
plt.title('Protocol Distribution (%)')
plt.ylabel('')

plt.tight_layout()
plt.show()

# %%
# Service analysis (if available)
if 'service' in df.columns:
    print("Top 10 Services:")
    service_counts = df['service'].value_counts().head(10)
    for service, count in service_counts.items():
        pct = count / len(df) * 100
        print(f"  {service}: {count:,} ({pct:.1f}%)")
    
    # Visualize top services
    plt.figure(figsize=(12, 6))
    service_counts.plot(kind='bar')
    plt.title('Top 10 Services')
    plt.ylabel('Number of Flows')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# %%
# Traffic volume statistics
numeric_cols = ['orig_bytes', 'resp_bytes', 'duration']
available_cols = [col for col in numeric_cols if col in df.columns]

if available_cols:
    print("Traffic Volume Statistics:")
    stats = df[available_cols].describe()
    print(stats)
    
    # Create box plots for traffic metrics
    fig, axes = plt.subplots(1, len(available_cols), figsize=(15, 5))
    if len(available_cols) == 1:
        axes = [axes]
    
    for i, col in enumerate(available_cols):
        # Use log scale for better visualization
        data = df[col][df[col] > 0]  # Remove zeros for log scale
        axes[i].boxplot(np.log10(data))
        axes[i].set_title(f'{col} (log10)')
        axes[i].set_ylabel('Log10 Value')
    
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 3. Temporal Analysis

# %%
# Convert timestamp if needed
if 'ts' in df.columns:
    if not pd.api.types.is_datetime64_any_dtype(df['ts']):
        df['ts'] = pd.to_datetime(df['ts'], unit='s')
    
    # Time range analysis
    time_range = df['ts'].max() - df['ts'].min()
    print(f"Time Range: {time_range}")
    print(f"Start: {df['ts'].min()}")
    print(f"End: {df['ts'].max()}")
    print(f"Total flows: {len(df):,}")
    
    # Flows per hour
    df['hour'] = df['ts'].dt.hour
    hourly_flows = df['hour'].value_counts().sort_index()
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    hourly_flows.plot(kind='bar')
    plt.title('Flows by Hour of Day')
    plt.xlabel('Hour')
    plt.ylabel('Number of Flows')
    
    # Traffic over time (binned)
    plt.subplot(1, 2, 2)
    df.set_index('ts').resample('1H').size().plot()
    plt.title('Traffic Volume Over Time')
    plt.xlabel('Time')
    plt.ylabel('Flows per Hour')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 4. Flow Characteristics Analysis

# %%
# Create derived features for analysis
if 'orig_bytes' in df.columns and 'resp_bytes' in df.columns:
    df['total_bytes'] = df['orig_bytes'] + df['resp_bytes']
    df['flow_ratio'] = df['orig_bytes'] / (df['resp_bytes'] + 1)  # +1 to avoid div by zero
    
    # Distribution of flow sizes
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    # Log scale histogram of total bytes
    total_bytes_nonzero = df['total_bytes'][df['total_bytes'] > 0]
    plt.hist(np.log10(total_bytes_nonzero), bins=50, alpha=0.7)
    plt.xlabel('Log10(Total Bytes)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Flow Sizes')
    
    plt.subplot(1, 3, 2)
    # Flow ratio distribution
    plt.hist(np.log10(df['flow_ratio'] + 0.001), bins=50, alpha=0.7)
    plt.xlabel('Log10(Orig/Resp Ratio)')
    plt.ylabel('Frequency')
    plt.title('Flow Direction Bias')
    
    plt.subplot(1, 3, 3)
    # Duration distribution
    if 'duration' in df.columns:
        duration_nonzero = df['duration'][df['duration'] > 0]
        plt.hist(np.log10(duration_nonzero), bins=50, alpha=0.7)
        plt.xlabel('Log10(Duration in seconds)')
        plt.ylabel('Frequency')
        plt.title('Flow Duration Distribution')
    
    plt.tight_layout()
    plt.show()

# %%
# Top talkers analysis
if 'id.orig_h' in df.columns and 'id.resp_h' in df.columns:
    print("Top Source IPs (by flow count):")
    top_sources = df['id.orig_h'].value_counts().head(10)
    for ip, count in top_sources.items():
        pct = count / len(df) * 100
        print(f"  {ip}: {count:,} flows ({pct:.1f}%)")
    
    print("\nTop Destination IPs (by flow count):")
    top_dests = df['id.resp_h'].value_counts().head(10)
    for ip, count in top_dests.items():
        pct = count / len(df) * 100
        print(f"  {ip}: {count:,} flows ({pct:.1f}%)")
    
    # Visualize top talkers
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    top_sources.plot(kind='barh')
    plt.title('Top Source IPs')
    plt.xlabel('Number of Flows')
    
    plt.subplot(1, 2, 2)
    top_dests.plot(kind='barh')
    plt.title('Top Destination IPs')
    plt.xlabel('Number of Flows')
    
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 5. Anomaly Detection Insights

# %%
# Look for potential anomalies
print("Potential Anomaly Indicators:")

# Large flows
if 'total_bytes' in df.columns:
    large_flow_threshold = df['total_bytes'].quantile(0.99)
    large_flows = df[df['total_bytes'] > large_flow_threshold]
    print(f"\n1. Large flows (>99th percentile: {large_flow_threshold:,.0f} bytes):")
    print(f"   Count: {len(large_flows)} ({len(large_flows)/len(df)*100:.2f}%)")
    if len(large_flows) > 0:
        print(f"   Largest: {large_flows['total_bytes'].max():,.0f} bytes")

# Long duration flows
if 'duration' in df.columns:
    long_duration_threshold = df['duration'].quantile(0.99)
    long_flows = df[df['duration'] > long_duration_threshold]
    print(f"\n2. Long duration flows (>99th percentile: {long_duration_threshold:.2f} seconds):")
    print(f"   Count: {len(long_flows)} ({len(long_flows)/len(df)*100:.2f}%)")
    if len(long_flows) > 0:
        print(f"   Longest: {long_flows['duration'].max():.2f} seconds")

# Unusual protocols or services
if 'proto' in df.columns:
    rare_protocols = df['proto'].value_counts()
    rare_protocols = rare_protocols[rare_protocols < len(df) * 0.01]  # <1% of traffic
    if len(rare_protocols) > 0:
        print(f"\n3. Rare protocols (<1% of traffic):")
        for proto, count in rare_protocols.items():
            print(f"   {proto}: {count} flows")

# %%
# Connection state analysis (if available)
if 'conn_state' in df.columns:
    print("Connection State Distribution:")
    conn_states = df['conn_state'].value_counts()
    for state, count in conn_states.items():
        pct = count / len(df) * 100
        print(f"  {state}: {count:,} ({pct:.1f}%)")
    
    # Look for unusual connection states
    unusual_states = ['REJ', 'RSTO', 'RSTOS0', 'SH', 'SHR']
    unusual_count = df[df['conn_state'].isin(unusual_states)].shape[0]
    if unusual_count > 0:
        print(f"\nUnusual connection states: {unusual_count} ({unusual_count/len(df)*100:.2f}%)")
    
    # Visualize connection states
    plt.figure(figsize=(12, 6))
    conn_states.plot(kind='bar')
    plt.title('Connection State Distribution')
    plt.ylabel('Number of Flows')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ## 6. Summary and Next Steps

# %%
# Summary statistics
print("=== DATASET SUMMARY ===")
print(f"Total flows: {len(df):,}")
print(f"Time span: {df['ts'].max() - df['ts'].min() if 'ts' in df.columns else 'Unknown'}")
print(f"Unique source IPs: {df['id.orig_h'].nunique() if 'id.orig_h' in df.columns else 'Unknown'}")
print(f"Unique destination IPs: {df['id.resp_h'].nunique() if 'id.resp_h' in df.columns else 'Unknown'}")
print(f"Protocols: {list(df['proto'].unique()) if 'proto' in df.columns else 'Unknown'}")

if 'total_bytes' in df.columns:
    total_traffic = df['total_bytes'].sum()
    print(f"Total traffic: {total_traffic / (1024**3):.2f} GB")

print("\n=== NEXT STEPS ===")
print("1. Generate rolling statistics features")
print("2. Create time-series tensors for deep learning")
print("3. Train baseline anomaly detection models")
print("4. Evaluate model performance on labeled data")
print("5. Deploy for real-time inference")
