from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# Load vectorstore and LLM
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
vectorstore = Chroma(persist_directory="vectorstore", embedding_function=embeddings)
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# 20 question evaluation set — calibrated to document content
test_set = [
    {"question": "What is the total number of branchless banking accounts in Pakistan?", "keywords": ["150,752,412", "150 million", "150.7 million"]},
    {"question": "How many active branchless banking accounts are there in Pakistan?", "keywords": ["66,593,997", "66 million", "66.5 million"]},
    {"question": "What is the share of digital payments in total retail payments in Pakistan?", "keywords": ["90", "88", "88%", "90%"]},
    {"question": "How many internet banking users does Pakistan have?", "keywords": ["14.9 million", "14.9", "12.0 million"]},
    {"question": "How many transactions has RAAST processed since its launch?", "keywords": ["1.9 billion", "1.9"]},
    {"question": "What is the overall financial inclusion rate in Pakistan?", "keywords": ["45", "financial inclusion", "inferred", "not explicitly"]},
    {"question": "What is the difference between male and female financial inclusion in Pakistan?", "keywords": ["56", "14", "male", "female"]},
    {"question": "What is female financial inclusion rate in Pakistan?", "keywords": ["14", "female", "women"]},
    {"question": "Which province has the highest financial inclusion rate in Pakistan?", "keywords": ["Punjab", "punjab"]},
    {"question": "What is the top reason people do not borrow from banks in Pakistan?", "keywords": ["other means", "borrow through other", "21%", "21 percent"]},
    {"question": "What is the PSDP allocation in the federal budget 2025-26?", "keywords": ["4,223,817", "1,530", "2,569", "682"]},
    {"question": "How did digital payments grow from FY19 to FY25?", "keywords": ["1.3 billion", "8.0 billion", "six times", "sixfold"]},
    {"question": "What is the mobile money access rate in Pakistan in 2024?", "keywords": ["37%", "37 percent", "37"]},
    {"question": "Why do Pakistanis not use digital savings platforms?", "keywords": ["trust", "awareness", "traditional", "informal", "preference"]},
    {"question": "What percentage of Pakistanis use digital means to access their bank accounts?", "keywords": ["57%", "57 percent", "57"]},
    {"question": "How many ATMs are available in Pakistan as of FY25?", "keywords": ["20,341", "20341", "20,000"]},
    {"question": "What is the financial inclusion rate in Islamabad?", "keywords": ["38%", "38 percent", "64%", "64 percent"]},
    {"question": "What is the total federal budget outlay for 2025-26?", "keywords": ["114,692", "11,072", "12,970", "9,805"]},
    {"question": "How does branchless banking contribute to financial inclusion in Pakistan?", "keywords": ["financial inclusion", "access", "mobile", "remote", "unbanked"]},
    {"question": "What are the barriers to financial inclusion for women in Pakistan?", "keywords": ["women", "female", "barrier", "document", "access", "knowledge"]},
]

# Run evaluation
print("Running evaluation...\n")
correct = 0
results = []

for i, test in enumerate(test_set):
    docs = vectorstore.similarity_search(test["question"], k=6)
    context = "\n\n".join([d.page_content for d in docs])

    messages = [
        SystemMessage(content=f"""Answer the question based ONLY on the context below.
Be specific and mention actual numbers when available.
If the answer is not in the context, say "I don't have information on that."

Context:
{context}"""),
        HumanMessage(content=test["question"])
    ]

    response = llm.invoke(messages)
    answer = response.content.lower()

    passed = any(kw.lower() in answer for kw in test["keywords"])
    if passed:
        correct += 1
        status = "PASS ✓"
    else:
        status = "FAIL ✗"

    results.append({
        "question": test["question"],
        "status": status,
        "answer": response.content[:150]
    })

    print(f"Q{i+1}: {status}")
    print(f"     Question: {test['question']}")
    if not passed:
        print(f"     Answer: {response.content[:150]}")
    print()

# Final score
accuracy = (correct / len(test_set)) * 100
print("=" * 50)
print(f"FINAL SCORE: {correct}/{len(test_set)} = {accuracy:.1f}%")
print("=" * 50)