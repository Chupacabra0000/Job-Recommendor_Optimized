import tfidf_terms


def test_tokenize_lowercases_and_removes_stopwords():
    text = "Python and SQL for Data Science in Moscow"
    tokens = tfidf_terms._tokenize(text)

    assert "python" in tokens
    assert "sql" in tokens
    assert "data" in tokens
    assert "science" in tokens
    assert "and" not in tokens
    assert "for" not in tokens
    assert "in" not in tokens


def test_extract_terms_returns_empty_for_blank_input():
    assert tfidf_terms.extract_terms("") == []
    assert tfidf_terms.extract_terms("   ") == []


def test_extract_terms_returns_empty_when_only_stopwords():
    assert tfidf_terms.extract_terms("and the in on of is are") == []


def test_extract_terms_respects_top_k():
    terms = tfidf_terms.extract_terms("python python sql pandas airflow docker kubernetes ml", top_k=3)
    assert len(terms) == 3
    assert len(set(terms)) == 3


def test_extract_terms_handles_multilingual_text():
    terms = tfidf_terms.extract_terms("Python разработка данных анализ sql engineer", top_k=5)
    assert "python" in terms
    assert "sql" in terms
