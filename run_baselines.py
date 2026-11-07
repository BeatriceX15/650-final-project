# 评分公式和检索过程参考hw2ranker.py

import argparse
import csv
import html
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent

QUERIES = [
    "daily job listings",
    "a tool to summarize PDF documents",
    "AI coding assistant",
    "Claude skills for reviewing code",
    "build a landing page with Claude",
    "use Claude to help write research papers",
    "turn a photo into a 3D model",
    "create animated videos with Claude",
    "generate videos with ComfyUI",
    "convert text to speech",
    "make PDF files smaller",
    "test a website with Claude Code",
    "check how much my Steam game library is worth",
    "save Telegram chat history to a file",
    "schedule posts across multiple social media accounts",
    "test a trading strategy with historical data",
    "check a website for security vulnerabilities",
    "connect a smart speaker to Home Assistant",
    "MiniMax coding workflow skills",
    "send API requests from the terminal",
]


def clean_text(text):
    text = html.unescape(text)
    text = re.sub(r"!?\[([^\]]*)\]\([^\s)]*[^)]*\)", r"\1", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split())


def tokenize(text):
    return re.findall(r"[^\W_]+", text.lower())


class RepositoryIndex:
    def __init__(self, documents, stopwords):
        self.postings = defaultdict(list)
        self.lengths = {}
        for doc_id, text in documents.items():
            tokens = tokenize(text)
            self.lengths[doc_id] = len(tokens)
            counts = Counter(token for token in tokens if token not in stopwords)
            for term, frequency in counts.items():
                self.postings[term].append((doc_id, frequency))
        self.n_docs = len(documents)
        self.average_length = sum(self.lengths.values()) / self.n_docs


class BM25:
    def __init__(self, index, b=0.75, k1=1.2, k3=8):
        self.index = index
        self.b, self.k1, self.k3 = b, k1, k3

    def score(self, doc_id, doc_counts, query_counts):
        length_norm = 1 - self.b + self.b * (
            self.index.lengths[doc_id] / self.index.average_length
        )
        score = 0.0
        for term, query_frequency in query_counts.items():
            tf = doc_counts.get(term, 0)
            if not tf:
                continue
            df = len(self.index.postings[term])
            idf = math.log((self.index.n_docs - df + 0.5) / (df + 0.5))
            doc_weight = (self.k1 + 1) * tf / (self.k1 * length_norm + tf)
            query_weight = (self.k3 + 1) * query_frequency / (self.k3 + query_frequency)
            score += idf * doc_weight * query_weight
        return score


class TFIDF:
    def __init__(self, index):
        self.index = index

    def score(self, doc_id, doc_counts, query_counts):
        score = 0.0
        for term in query_counts:
            tf = doc_counts.get(term, 0)
            if not tf:
                continue
            df = len(self.index.postings[term])
            score += math.log(tf + 1) * (math.log(self.index.n_docs / df) + 1)
        return score


def search(query, index, scorer, stopwords):
    query_counts = Counter(t for t in tokenize(query) if t not in stopwords)
    candidates = defaultdict(Counter)
    for term in query_counts:
        for doc_id, frequency in index.postings.get(term, []):
            candidates[doc_id][term] = frequency
    scores = [(doc_id, scorer.score(doc_id, counts, query_counts))
              for doc_id, counts in candidates.items()]
    return sorted(scores, key=lambda item: (-item[1], item[0]))


def write_csv(path, rows, columns):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run(data_path, output_dir, top_k, seed, stopwords_path,
        query_texts=QUERIES, query_start=1):
    with data_path.open(encoding="utf-8") as file:
        repositories = [json.loads(line) for line in file if line.strip()]
    by_id = {int(repo["repo_id"]): repo for repo in repositories}
    stopwords = set(stopwords_path.read_text(encoding="utf-8").splitlines())
    stopwords -= {"no", "not", "without"}
    documents = {}
    for doc_id, repo in by_id.items():
        fields = [repo["name"], repo["description"] or "",
                  " ".join(repo["topics"]), repo["readme_raw"]]
        documents[doc_id] = clean_text("\n".join(fields))
    index = RepositoryIndex(documents, stopwords)
    methods = {"BM25": BM25(index), "TF-IDF": TFIDF(index)}

    queries = [{"query_id": f"q{i:02d}", "query": query}
               for i, query in enumerate(query_texts, start=query_start)]
    results, pool = [], []
    all_ids = sorted(by_id)
    for i, query in enumerate(queries, start=query_start - 1):
        ranked = {name: search(query["query"], index, scorer, stopwords)[:top_k]
                  for name, scorer in methods.items()}
        ranked["Random"] = [(doc_id, None) for doc_id in
                            random.Random(seed + i).sample(all_ids, top_k)]
        pooled_ids = set()
        for method, ranking in ranked.items():
            for rank, (doc_id, score) in enumerate(ranking, start=1):
                repo = by_id[doc_id]
                results.append({**query, "method": method, "rank": rank,
                                "repo_id": doc_id, "full_name": repo["full_name"],
                                "repo_url": repo["repo_url"], "score": score,
                                "description": repo.get("description") or ""})
                pooled_ids.add(doc_id)

        #######
        shuffled_ids = sorted(pooled_ids)
        random.Random(seed + 10000 + i).shuffle(shuffled_ids)
        for doc_id in shuffled_ids:
            repo = by_id[doc_id]
            pool.append({**query, "repo_id": doc_id, "full_name": repo["full_name"],
                         "repo_url": repo["repo_url"],
                         "description": repo.get("description") or "",
                         "readme_excerpt": clean_text(repo["readme_raw"])[:1800],
                         "relevance": "", "notes": ""})

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "retrieval_results.csv", results,
              ["query_id", "query", "method", "rank", "repo_id", "full_name",
               "repo_url", "score", "description"])
    columns = ["query_id", "query", "repo_id", "full_name", "repo_url", "description",
               "readme_excerpt", "relevance", "notes"]
    write_csv(output_dir / "annotation_pool.csv", pool, columns)

    print(f"Repositories: {len(repositories)} | Queries: {len(queries)} | Top k: {top_k}")
    print(f"Result rows: {len(results)} | Unique query-repo pairs to annotate: {len(pool)}")
    print(f"Saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PROJECT_DIR / "projects.jsonl")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "baseline_results")
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stopwords", type=Path, default=PROJECT_DIR / "stopwords.txt")
    args = parser.parse_args()
    run(args.data, args.output, args.top_k, args.seed, args.stopwords)
