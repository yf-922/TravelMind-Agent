from app.core import semantic_memory


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


def test_semantic_memory_prompt_marks_current_request_as_higher_priority():
    prompt = semantic_memory.format_semantic_memories(["带父母时每天少走路", "不吃太辣"])
    assert "当前用户明确需求优先" in prompt
    assert "不吃太辣" in prompt


def test_semantic_memory_cleans_duplicates_and_long_values():
    values = semantic_memory._clean_memories(["偏好慢节奏", "偏好慢节奏", "x" * 300])
    assert values == ["偏好慢节奏", "x" * 240]
