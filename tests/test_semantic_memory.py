from app.core import semantic_memory


def test_real_chroma_collection_enforces_user_filter(monkeypatch):
    chromadb = __import__("chromadb")
    from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

    class DeterministicEmbedding(EmbeddingFunction):
        def __init__(self):
            pass

        def __call__(self, input: Documents) -> Embeddings:
            return [[float(sum(ord(char) for char in text) % 97), float(len(text)), 1.0]
                    for text in input]

        def name(self) -> str:
            return "deterministic_test"

        def get_config(self):
            return {}

    collection = chromadb.EphemeralClient().get_or_create_collection(
        "semantic_memory_test", embedding_function=DeterministicEmbedding()
    )
    collection.upsert(
        ids=["u1-memory", "u2-memory"],
        documents=["偏好历史文化博物馆", "偏好海鲜美食"],
        metadatas=[{"user_id": "u1"}, {"user_id": "u2"}],
    )
    monkeypatch.setattr(semantic_memory, "_collection", lambda: collection)
    assert semantic_memory.search_user_memories("u1", "历史文化") == ["偏好历史文化博物馆"]
    assert semantic_memory.search_user_memories("u2", "历史文化") == ["偏好海鲜美食"]
    assert semantic_memory.search_user_memories("u3", "历史文化") == []


class FakeCollection:
    def __init__(self):
        self.rows = []

    def upsert(self, ids, documents, metadatas):
        self.rows.extend(zip(ids, documents, metadatas))

    def query(self, query_texts, n_results, where, include):
        docs = [doc for _, doc, meta in self.rows if meta["user_id"] == where["user_id"]]
        return {"documents": [docs[:n_results]]}


def test_semantic_memory_is_isolated_by_user(monkeypatch):
    store = FakeCollection()
    monkeypatch.setattr(semantic_memory, "_collection", lambda: store)
    store.upsert(["a", "b"], ["偏好微辣", "每天少走路"], [{"user_id": "u1"}, {"user_id": "u2"}])

    assert semantic_memory.search_user_memories("u1", "重庆美食") == ["偏好微辣"]
    assert semantic_memory.search_user_memories("u2", "带父母旅行") == ["每天少走路"]


def test_semantic_memory_status_distinguishes_empty_from_unavailable(monkeypatch):
    class EmptyCollection:
        def query(self, **kwargs):
            return {"documents": [[]]}

    monkeypatch.setattr(semantic_memory, "_collection", lambda: EmptyCollection())
    assert semantic_memory.search_user_memories_with_status("u1", "museum") == ([], "ok")
    monkeypatch.setattr(semantic_memory, "_collection", lambda: None)
    assert semantic_memory.search_user_memories_with_status("u1", "museum") == ([], "degraded")


def test_semantic_memory_status_marks_query_failure(monkeypatch):
    class BrokenCollection:
        def query(self, **kwargs):
            raise RuntimeError("vector index unavailable")

    monkeypatch.setattr(semantic_memory, "_collection", lambda: BrokenCollection())
    assert semantic_memory.search_user_memories_with_status("u1", "museum") == ([], "degraded")


def test_semantic_memory_deletion_is_user_scoped(monkeypatch):
    class Collection:
        def __init__(self):
            self.filters = []

        def delete(self, **kwargs):
            self.filters.append(kwargs)

    collection = Collection()
    monkeypatch.setattr(semantic_memory, "_collection", lambda: collection)
    assert semantic_memory.delete_user_memories("u1") == (True, "ok")
    assert collection.filters == [{"where": {"user_id": "u1"}}]


def test_semantic_memory_deletion_reports_unavailable_store(monkeypatch):
    monkeypatch.setattr(semantic_memory, "_collection", lambda: None)
    assert semantic_memory.delete_user_memories("u1") == (False, "degraded")


def test_semantic_memory_prompt_marks_current_request_as_higher_priority():
    prompt = semantic_memory.format_semantic_memories(["带父母时每天少走路", "不吃太辣"])
    assert "当前用户明确需求优先" in prompt
    assert "不吃太辣" in prompt


def test_semantic_memory_cleans_duplicates_and_long_values():
    values = semantic_memory._clean_memories(["偏好慢节奏", "偏好慢节奏", "x" * 300])
    assert values == ["偏好慢节奏", "x" * 240]
