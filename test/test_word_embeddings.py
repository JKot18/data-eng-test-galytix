import pytest
from src.word_embeddings import WordEmbeddings


@pytest.fixture
def test_similarity_calculation(word_embeddings):
    similarity = word_embeddings.get_similarity("cat", "dog")
    assert isinstance(similarity, tuple)
    assert len(similarity) == 2
    assert all(isinstance(s, float) for s in similarity)

@pytest.fixture
def test_find_closest_match(word_embeddings):
    closest_match, similarity = word_embeddings.find_closest_match("cat")
    assert isinstance(closest_match, str)
    assert isinstance(similarity, float)

