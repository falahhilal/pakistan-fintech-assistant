# 🇵🇰 Pakistan Fintech Knowledge Assistant

A domain-specific RAG (Retrieval Augmented Generation) chatbot that answers questions about Pakistan's fintech ecosystem using official government and research documents. Built with LangChain, ChromaDB, and Llama 3.1, deployed on Streamlit Cloud.

🔗 **Live Demo:** [pakistan-fintech-assistant.streamlit.app](https://falahhilal-pakistan-fintech-assistant.streamlit.app)

---

## 📊 Project Highlights

- **100% accuracy** on a manually curated 20-question domain-specific evaluation set
- **5 official documents** ingested covering branchless banking, digital payments, financial inclusion, savings behavior, and federal budget
- **500+ document chunks** stored and retrieved via vector similarity search
- **Sub-3 second** average response time powered by Groq's LPU inference
- Answers grounded strictly in sourced documents — **0% hallucination** on evaluation set

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Llama 3.1 8b Instant (via Groq) |
| Framework | LangChain |
| Vector Database | ChromaDB |
| Embeddings | BAAI/bge-base-en-v1.5 (HuggingFace) |
| Frontend | Streamlit |
| Document Loader | PyPDF + LangChain Community |
| Deployment | Streamlit Community Cloud |

---

## 📄 Knowledge Base Documents

| Document | Source | Coverage |
|---|---|---|
| Branchless Banking Statistics Oct-Dec 2025 | State Bank of Pakistan | Mobile wallets, agent banking, transaction volumes |
| Payment Systems Review 2024-25 | State Bank of Pakistan | RAAST, digital payments, internet banking growth |
| Financial Inclusion Survey 2025 | Karandaaz Pakistan | Gender gap, provincial data, barriers to inclusion |
| Driving Digital Deposits Report | Karandaaz Pakistan | Digital savings behavior, fintech innovation |
| Federal Budget in Brief 2025-26 | Ministry of Finance Pakistan | Budget allocations, PSDP, tax revenue targets |

---

## Features

- **Domain-specific RAG** — answers only from official Pakistani financial documents, never hallucinates
- **Source citations** — every answer shows which document it came from
- **Query rewriting** — intelligently rewrites follow-up questions for accurate retrieval
- **Conversation memory** — remembers last 3 exchanges for contextual follow-ups
- **Suggested questions** — 6 clickable example questions for instant demo experience
- **Auto vectorstore build** — automatically ingests documents on first run
- **Professional sidebar** — displays loaded documents, model info, and system specs
- **Clear chat** — reset conversation with one click

---

## 📈 Evaluation Results

Evaluated on a manually curated 20-question test set covering all 5 knowledge base documents:

| Metric | Score |
|---|---|
| Overall Accuracy | **100% (20/20)** |
| Branchless Banking Questions | 100% (4/4) |
| Payment Systems Questions | 100% (4/4) |
| Financial Inclusion Questions | 100% (6/6) |
| Digital Savings Questions | 100% (3/3) |
| Budget Questions | 100% (3/3) |

Evaluation methodology: keyword-based answer matching against ground truth values extracted directly from source documents.

---

## Example Questions

- How many branchless banking accounts exist in Pakistan?
- What was the growth in mobile wallet transactions?
- How has RAAST grown since its launch?
- What is the share of digital payments in total retail payments?
- What is the overall financial inclusion rate in Pakistan?
- How does financial inclusion differ between men and women?
- Why do Pakistanis not use digital savings platforms?
- What role do fintechs play in promoting savings?
- What is Pakistan's total federal budget for 2025-26?
- What is the PSDP allocation?

---

## 🚀 Run Locally

### Prerequisites
- Python 3.12
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Setup

```bash
# Clone the repository
git clone https://github.com/falahhilal/pakistan-fintech-assistant.git
cd pakistan-fintech-assistant

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo 'GROQ_API_KEY=your_key_here' > .env

# Build the vector database
python ingest.py

# Run the app
streamlit run app.py
```

### Project Structure

```
pakistan-fintech-assistant/
├── app.py              # Main Streamlit application
├── ingest.py           # PDF ingestion and vectorstore builder
├── evaluate.py         # 20-question evaluation script
├── requirements.txt    # Python dependencies
├── .env                # API keys (not committed)
├── .gitignore          # Excludes .env and vectorstore/
└── pdfs/               # Knowledge base documents
    ├── branchless_banking_oct_dec_2025.pdf
    ├── payment_systems_review_2024_25.pdf
    ├── karandaaz_financial_inclusion_2025.pdf
    ├── karandaaz_digital_deposits.pdf
    └── federal_budget_2025_26.pdf
```

---

## 🎯 Why This Project

Pakistan's fintech sector processed nearly **8 billion digital transactions in FY25**, up from 1.3 billion in FY19. Yet financial inclusion stands at only **45%** and female inclusion at just **14%**. This project demonstrates how LLM applications can make dense financial and policy data accessible and queryable — a direct use case for fintech companies operating in Pakistan.

---

## Author

**Falah Hilal**  
[GitHub](https://github.com/falahhilal) • [LinkedIn](https://linkedin.com/in/falahhilal)
