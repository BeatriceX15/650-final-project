from run_baselines import PROJECT_DIR, run


QUERIES = [
    "Neural networks with Python and PyTorch",
    "Node.js projects for building conversational chatbots",
    "PyTorch object detection in images",
    "PyTorch time series forecasting",
    "PDF question-answering chatbots",
    "LangGraph projects for coordinating multiple AI agents",
    "Python projects using PostgreSQL",
    "projects combining LangChain and FastAPI",
    "JavaScript projects for image classification",
    "Python projects for answering database questions",
    "Natural language to SQL",
    "anomaly detection for time series data",
    "Semantic search with embeddings",
    "Text-to-3D generation",
    "LangChain document summarization",
    "Multi-agent debate",
    "Multi-agent task planning",
    "MCP database tools",
    "Python recommendation system",
    "Python web scraping",
]


if __name__ == "__main__":
    run(PROJECT_DIR / "projects.jsonl",
        PROJECT_DIR / "baseline_results_2",
        25, 42, PROJECT_DIR / "stopwords.txt",
        query_texts=QUERIES, query_start=21)
