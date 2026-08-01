import unittest
from pathlib import Path

from langchain_core.documents import Document

from rag.qa_service import EducationQAService
from utils.config_handler import app_conf
from utils.file_handler import calculate_sha256, list_knowledge_files
from utils.path_tool import get_abs_path
from utils.prompt_loader import load_rag_answer_prompt, load_system_prompt


class ProjectContentTests(unittest.TestCase):
    def test_subject_knowledge_files_exist(self):
        knowledge_path = Path(get_abs_path("data/knowledge"))
        files = list_knowledge_files(knowledge_path, ("txt", "pdf"))
        stems = {path.stem for path in files}
        for subject in ("语文", "数学", "英语", "科学", "安全与品德"):
            self.assertTrue(
                any(stem.startswith(subject) for stem in stems),
                f"缺少 {subject} 知识文件",
            )
        self.assertEqual(len(files), 5)

    def test_prompts_are_education_specific(self):
        combined = load_system_prompt() + load_rag_answer_prompt()
        self.assertIn("小学生", combined)
        self.assertIn("search_knowledge", combined)
        self.assertIn("教育知识问答", combined)
        self.assertIn("综合", app_conf["supported_subjects"])

    def test_hash_is_stable(self):
        path = Path(get_abs_path("data/knowledge/数学基础.txt"))
        self.assertEqual(calculate_sha256(path), calculate_sha256(path))
        self.assertEqual(len(calculate_sha256(path)), 64)

    def test_document_formatting(self):
        documents = [
            Document(
                page_content="三角形内角和是 180°。",
                metadata={"source": "数学基础.txt", "subject": "数学"},
            )
        ]
        result = EducationQAService.format_documents(documents)
        self.assertIn("数学基础.txt", result)
        self.assertIn("180°", result)


if __name__ == "__main__":
    unittest.main()
