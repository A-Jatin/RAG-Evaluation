import bs4
from langchain_community.document_loaders import WebBaseLoader

from main import run_query


def test_run_query():
    # Load, chunk and index the contents of the blog.
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    query = "What are some good ways to write the prompt?"
    results = run_query(docs, query)
    print(results)
    assert results
    assert isinstance(results, dict)
    assert results.keys()
