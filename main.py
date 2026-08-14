from dataset import load_language, iter_passages, iter_queries

if __name__ == "__main__":
    ds = load_language("hi")
    print(ds)

    # Sample: first 3 passage records (for indexing)
    for p in list(iter_passages(ds["train"], "hi"))[:3]:
        print(p.passage_id, p.is_selected, p.text[:80])

    # Sample: first 3 query records (for evaluation)
    for q in list(iter_queries(ds["validation"], "hi"))[:3]:
        print(q.query_id, q.query[:80])
