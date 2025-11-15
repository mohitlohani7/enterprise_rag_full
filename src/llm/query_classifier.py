def classify_query(query: str) -> str:
    q = query.lower()
    if q.startswith('define') or 'what is' in q:
        return 'definition'
    if 'compare' in q or 'vs' in q:
        return 'comparison'
    if 'summarize' in q or 'summary' in q:
        return 'summarization'
    if q.strip().endswith('?'):
        return 'question'
    return 'general'
