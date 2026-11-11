import sys
import config
from store import get_client, get_embedder


def main():
    if len(sys.argv) < 2:
        print('用法：python search.py "你的問題" [top_k]')
        sys.exit(1)
    query = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    client = get_client()
    vec = get_embedder().embed_query(query)
    hits = client.query_points(
        collection_name=config.COLLECTION, query=vec, limit=top_k, with_payload=True
    ).points

    for rank, h in enumerate(hits, 1):
        p = h.payload
        text = p["text"].replace("\n", " ")
        print(f"\n#{rank}  score={h.score:.3f}  {p['source_file']}  第 {p['page']} 頁")
        print(f"    {text[:200]}{'…' if len(text) > 200 else ''}")


if __name__ == "__main__":
    main()
