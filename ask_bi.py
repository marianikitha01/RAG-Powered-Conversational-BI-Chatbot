import os
import pandas as pd
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA


# Load keys (for LLM only)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Read your CSV
df = pd.read_csv("data/sales.csv")
# Prepare texts and metadata
texts = []
metadatas = []
for _, row in df.iterrows():
    region = row.get("Country", "")
    product = row.get("Description", "")
    sales = row.get("Sales", row.get("Quantity", 0) * row.get("Price", 0))
    date = row.get("InvoiceDate", "")
    text = f"{region} | {product} | {sales}"
    texts.append(text)
    metadatas.append({"Region": region, "Product": product, "Sales": sales, "Date": date})

# Build embeddings + FAISS index
embed_model_name = "all-MiniLM-L6-v2"
# Local SentenceTransformer for LangChain
hf = HuggingFaceEmbeddings(model_name=embed_model_name)
# FAISS vectorstore
vectorstore = FAISS.from_texts(texts, hf, metadatas=metadatas)

# Set up the LLM + RAG chain
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
)

# Chat loop
print("\n🔍 BI Chatbot Ready! Ask a question (type 'exit' to quit)\n")
while True:
    q = input("Your question: ")
    if q.lower() in ("exit", "quit"):
        print("Goodbye!")
        break
    ans = qa.run(q)
    print(f"\nAnswer:\n{ans}\n")
