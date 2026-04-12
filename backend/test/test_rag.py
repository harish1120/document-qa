import pytest
from backend.rag import hybrid_search
from dotenv import load_dotenv
from backend.rag_graph import answer_question

load_dotenv()

def test_hybrid_search_returns_documents():
    result = hybrid_search("What is AI Engineering?", k = 5)
    assert len(result)<= 5
    assert all(hasattr(d, 'page_content') for d in result)

                                                                                                                                                         
def test_rag_graph_end_to_end():
    answer, sources = answer_question("What is AI?")
    assert isinstance(answer, str)
    assert len(answer) > 0
    assert len(sources) > 0