import os
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec, exceptions as pinecone_exceptions
from sentence_transformers import SentenceTransformer

# 1. Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_REGION = os.getenv("PINECONE_REGION")  # e.g., "us-west1"
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD")    # e.g., "gcp"

# 2. Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# 3. Initialize local embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dim = model.get_sentence_embedding_dimension()

# 4. Create or reuse a Pinecone index with the correct dimension
index_name = "sales-index-st"
try:
    existing = pc.list_indexes()
    if index_name not in existing:
        spec = ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)
        pc.create_index(
            name=index_name,
            dimension=embedding_dim,
            metric="cosine",
            spec=spec
        )
except pinecone_exceptions.PineconeApiException as e:
    if "ALREADY_EXISTS" not in str(e):
        raise

# 5. Get a handle to the index
index = pc.Index(index_name)

# 6. Read your CSV from the data folder
csv_path = os.path.join("data", "sales.csv")
if not os.path.isfile(csv_path):
    raise FileNotFoundError(f"CSV not found at {csv_path}")
df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} rows from {csv_path}")

# 7. Ensure 'Sales' column exists or compute it using 'Quantity' and 'Price'
if "Sales" not in df.columns:
    if "Quantity" in df.columns and "Price" in df.columns:
        df["Sales"] = df["Quantity"] * df["Price"]
        print("Computed 'Sales' as Quantity * Price")
    else:
        raise ValueError(
            "Missing columns: cannot compute 'Sales'. Ensure 'Quantity' and 'Price' are present."
        )

# 8. Sample to limit to manageable size
sample_size = 10000
if len(df) > sample_size:
    df = df.sample(n=sample_size, random_state=42)
    print(f"Sampling {len(df)} rows for indexing")

# 9. Upsert vectors in batches using local embeddings
batch_size = 100
for start in range(0, len(df), batch_size):
    batch = df.iloc[start : start + batch_size]
    to_upsert = []
    for idx, row in batch.iterrows():
        # Safely extract metadata, handling NaNs
        region = row["Country"] if "Country" in row and pd.notna(row["Country"]) else ""
        product = row["Description"] if "Description" in row and pd.notna(row["Description"]) else ""
        sales_val = row["Sales"] if pd.notna(row["Sales"]) else 0.0
        date_val = row["InvoiceDate"] if "InvoiceDate" in row and pd.notna(row["InvoiceDate"]) else ""

        # Prepare text and vector
        text = f"{region} | {product} | {sales_val}"
        vector = model.encode(text).tolist()

        # Build JSON-safe metadata
        metadata = {
            "Region": region,
            "Product": product,
            "Sales": sales_val,
            "Date": str(date_val)
        }
        to_upsert.append((str(idx), vector, metadata))

    index.upsert(vectors=to_upsert)
    print(f"Upserted rows {start} to {start + len(batch)}")

print("✅ Finished indexing all data into Pinecone with local embeddings!")
