import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from rag.vector_store import EducationVectorStore


class InMemoryVectorStore:
    def __init__(self):
        self.documents = {}

    def add_documents(self, documents, ids):
        self.documents.update(dict(zip(ids, documents)))
        return ids

    def delete(self, ids):
        for document_id in ids:
            self.documents.pop(document_id, None)


class VectorStoreTests(unittest.TestCase):
    def test_add_update_and_remove_file(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            knowledge = root / "knowledge"
            knowledge.mkdir()
            source = knowledge / "数学测试.txt"
            source.write_text("长方形面积等于长乘宽。", encoding="utf-8")

            store = EducationVectorStore()
            store.data_path = knowledge
            store.manifest_path = root / "manifest.json"
            store._vector_store = InMemoryVectorStore()

            first = store.sync()
            self.assertEqual(first.added_files, 1)
            self.assertGreater(first.indexed_chunks, 0)

            second = store.sync()
            self.assertEqual(second.unchanged_files, 1)

            source.write_text(
                "长方形面积等于长乘宽，周长等于长与宽的和乘二。",
                encoding="utf-8",
            )
            third = store.sync()
            self.assertEqual(third.updated_files, 1)

            source.unlink()
            fourth = store.sync()
            self.assertEqual(fourth.removed_files, 1)


if __name__ == "__main__":
    unittest.main()
