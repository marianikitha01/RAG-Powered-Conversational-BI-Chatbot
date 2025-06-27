import os
import pandas as pd
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load and prepare sales data
df = pd.read_csv("data/sales.csv")
# Ensure Sales column
if 'Sales' not in df.columns:
    if 'Quantity' in df.columns and 'Price' in df.columns:
        df['Sales'] = df['Quantity'] * df['Price']
    else:
        raise ValueError("Missing Quantity/Price to compute Sales")

# Build texts and metadata for FAISS
texts, metadatas = [], []
for _, row in df.iterrows():
    region = row.get("Country", "")
    product = row.get("Description", "")
    sales = row.get("Sales", 0)
    date = row.get("InvoiceDate", "")
    texts.append(f"{region} | {product} | {sales}")
    metadatas.append({"Region": region, "Product": product, "Sales": sales, "Date": date})

# Build FAISS index with MiniLM embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

# Cloud-based RetrievalQA chain
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
cloud_qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# pandas fallback for biggest sales increase

def fallback_biggest_increase():
    temp = df.copy()
    temp['InvoiceDate'] = pd.to_datetime(temp['InvoiceDate'], dayfirst=False, errors='coerce')
    temp['YM'] = temp['InvoiceDate'].dt.to_period('M')
    nov = temp[temp['YM']=='2010-11'].groupby('Description')['Sales'].sum()
    dec = temp[temp['YM']=='2010-12'].groupby('Description')['Sales'].sum()
    diff = (dec - nov).dropna()
    if diff.empty:
        return "No data for Nov/Dec 2010"
    prod = diff.idxmax()
    val = diff.max()
    return f"{prod} had the biggest sales increase: {val:.2f}"

# Hybrid answer: pandas BI or retrieval fallback

def hybrid_answer(query: str) -> str:
    q_lower = query.lower()
    # BI-specific fallback
    if "biggest sales increase" in q_lower:
        return fallback_biggest_increase()

    # Try cloud LLM
    try:
        return cloud_qa.run(query)
    except Exception as e:
        print(f"⚠️ Cloud error ({e}) – showing top relevant records instead.")
        # Retrieval-only fallback
        docs = vectorstore.similarity_search(query, k=5)
        lines = [f"{d.page_content}" for d in docs]
        return "Here are some relevant records:\n" + "\n".join(lines)

# Interactive chat loop
print("\n🔍 BI Chatbot Ready! (cloud first, pandas/fallback). Type 'exit' to quit.\n")
while True:
    q = input("Your question: ")
    if q.lower() in ("exit","quit"):
        print("Goodbye!")
        break
    ans = hybrid_answer(q)
    print(f"\nAnswer:\n{ans}\n")
