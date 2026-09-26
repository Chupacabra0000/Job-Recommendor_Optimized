from tfidf_terms import _tokenize, extract_terms


def test_tokenize_removes_stopwords_and_keeps_tech_tokens():
    tokens = _tokenize('Python and SQL в data-science C++ .NET 2025')
    assert 'and' not in tokens
    assert 'в' not in tokens
    assert 'python' in tokens
    assert 'sql' in tokens
    assert 'data-science' in tokens
    assert 'c++' in tokens


def test_extract_terms_handles_empty_and_returns_unique_terms():
    assert extract_terms('', top_k=5) == []
    terms = extract_terms('Python Python SQL analytics dashboards', top_k=3)
    assert len(terms) <= 3
    assert len(set(terms)) == len(terms)
    assert any(term in terms for term in ['python', 'sql', 'analytics'])
