def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def format_texts(texts):
    return "\n\n".join(text for text in texts)