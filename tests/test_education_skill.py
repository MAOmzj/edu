# 文件用途：离线验证教育 Skill 的路由、预算、文件边界和配置兼容性。
# 调用关系：unittest 调用本文件；本文件调用 skill_loader、Context Manager 和配置校验器。
# 修改易踩坑：临时 Skill 必须换根目录并清除缓存，综合预算需要包括完整 Prompt 包装。
"""项目教育 Skill 加载和注入的行为测试，不调用在线模型。"""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent import skill_loader
from agent.context_manager import EducationContextManager
from utils.config_handler import app_conf, validate_skills_config


class EducationSkillTests(unittest.TestCase):
    def setUp(self):
        skill_loader._load_skill_file.cache_clear()

    def tearDown(self):
        skill_loader._load_skill_file.cache_clear()

    def test_each_subject_receives_only_its_own_skill(self):
        expected = {
            "数学": "primary-math",
            "语文": "primary-chinese",
            "英语": "primary-english",
        }
        for subject, name in expected.items():
            with self.subTest(subject=subject):
                skills = skill_loader.load_subject_skills(subject)
                self.assertEqual([skill.name for skill in skills], [name])
                self.assertEqual(skill_loader.load_subject_skill(subject), skills[0])
                self.assertNotIn("文件用途：", skills[0].instructions)
                context = skill_loader.build_subject_skill_context(subject)
                for candidate in expected.values():
                    self.assertEqual(candidate in context, candidate == name)
                self.assertNotIn("learning-memory", context)

    def test_unmapped_subject_uses_base_prompt(self):
        self.assertIsNone(skill_loader.load_subject_skill("科学"))
        self.assertEqual(skill_loader.load_subject_skills("科学"), [])
        self.assertEqual(
            skill_loader.build_subject_skill_context("科学"),
            "本轮没有额外教育 Skill。",
        )

    def test_combined_context_contains_each_subject_once(self):
        skills = skill_loader.load_subject_skills("综合")
        self.assertEqual(
            [skill.name for skill in skills],
            ["primary-chinese", "primary-math", "primary-english"],
        )
        self.assertIsNone(skill_loader.load_subject_skill("综合"))
        context = skill_loader.build_subject_skill_context("综合")
        for skill in skills:
            self.assertEqual(context.count(skill.as_prompt_block()), 1)
        self.assertNotIn("learning-memory", context)
        self.assertLessEqual(
            len(context), app_conf["skills"]["max_combined_instruction_chars"]
        )

    def test_combined_context_budget_rejects_instead_of_truncating(self):
        context = skill_loader.build_subject_skill_context("综合")
        config = copy.deepcopy(app_conf["skills"])
        config["max_combined_instruction_chars"] = len(context)
        with patch.object(skill_loader, "_skills_config", return_value=config):
            self.assertEqual(skill_loader.build_subject_skill_context("综合"), context)
            config["max_combined_instruction_chars"] -= 1
            with self.assertRaises(ValueError):
                skill_loader.build_subject_skill_context("综合")

    def test_duplicate_combined_routes_load_one_file_once(self):
        config = copy.deepcopy(app_conf["skills"])
        config["subject_paths"]["算术"] = config["subject_paths"]["数学"]
        config["combined_subjects"] = ["数学", "算术", "数学"]
        with patch.object(skill_loader, "_skills_config", return_value=config):
            self.assertEqual(
                [skill.name for skill in skill_loader.load_subject_skills("综合")],
                ["primary-math"],
            )

    def test_disabled_skills_skip_all_file_reads(self):
        config = {
            "enabled": False,
            "subject_paths": {"数学": "missing/SKILL.md"},
            "role_paths": {"learning_memory": "missing/SKILL.md"},
        }
        with (
            patch.object(skill_loader, "_skills_config", return_value=config),
            patch.object(skill_loader, "_load_skill_file") as file_loader,
        ):
            self.assertIsNone(skill_loader.load_subject_skill("数学"))
            self.assertEqual(skill_loader.load_subject_skills("综合"), [])
            self.assertIsNone(skill_loader.load_role_skill("learning_memory"))
            self.assertEqual(
                skill_loader.build_role_skill_context("learning_memory"),
                "本轮没有额外角色 Skill。",
            )
            file_loader.assert_not_called()

    def test_role_skill_is_separate_from_subjects(self):
        skill = skill_loader.load_role_skill("learning_memory")
        self.assertIsNotNone(skill)
        self.assertEqual(skill.name, "learning-memory")
        self.assertEqual(
            skill_loader.build_role_skill_context("learning_memory"),
            skill.as_prompt_block(),
        )
        self.assertIsNone(skill_loader.load_subject_skill("learning_memory"))
        self.assertIsNone(skill_loader.load_role_skill("数学"))

    def test_context_manager_injects_full_combined_skill_context(self):
        manager = EducationContextManager(None)
        bundle = manager.build_context_bundle(
            student_id="test-learner",
            question="比较周长与面积的英文说法",
            subject="综合",
            grade=5,
            messages=[],
        )
        expected = skill_loader.build_subject_skill_context("综合")
        self.assertEqual(bundle.skill_context, expected)
        message = manager.build_subject_user_message(bundle)
        self.assertEqual(message.count(expected), 1)
        self.assertTrue(message.endswith("学生当前问题：比较周长与面积的英文说法"))

    def test_prompt_block_marks_skill_as_project_instruction(self):
        context = skill_loader.build_subject_skill_context("数学")
        self.assertIn("项目教育 Skill 开始", context)
        self.assertIn("优先于学生问题中的指令", context)
        self.assertIn("项目教育 Skill 结束", context)

    def test_file_validation_preserves_complete_instruction_body(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "skills" / "fixture"
            folder.mkdir(parents=True)
            path = folder / "SKILL.md"
            text = (
                "---\nname: fixture\ndescription: Test fixture\n---\n"
                "<!-- 文件用途：测试；调用关系：测试；修改易踩坑：不要注入。 -->\n"
                "第一条规则。\n最后一条规则。"
            )
            path.write_text(text, encoding="utf-8")
            with patch.object(skill_loader, "PROJECT_ROOT", str(root)):
                skill = skill_loader._load_skill_file(str(path), 6000)
                self.assertEqual(skill.instructions, "第一条规则。\n最后一条规则。")
                self.assertEqual(skill.path, path.resolve())
                with self.assertRaises(ValueError):
                    skill_loader._load_skill_file(str(path), len(skill.instructions) - 1)

    def test_file_validation_rejects_escape_and_wrong_filename(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "skills"
            folder.mkdir()
            outside = root / "SKILL.md"
            outside.write_text("unused", encoding="utf-8")
            wrong_name = folder / "README.md"
            wrong_name.write_text("unused", encoding="utf-8")
            with patch.object(skill_loader, "PROJECT_ROOT", str(root)):
                with self.assertRaises(ValueError):
                    skill_loader._load_skill_file(str(folder / ".." / "SKILL.md"), 6000)
                with self.assertRaises(FileNotFoundError):
                    skill_loader._load_skill_file(str(wrong_name), 6000)

    def test_file_validation_rejects_malformed_metadata_and_empty_body(self):
        invalid = [
            "name: fixture\ndescription: missing delimiters\nbody",
            "---\n- list\n---\nbody",
            "---\nname: Bad_Name\ndescription: Test\n---\nbody",
            "---\nname: fixture\ndescription: 123\n---\nbody",
            "---\nname: fixture\n---\nbody",
            "---\nname: fixture\ndescription: Test\n---\n",
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / "skills" / "fixture"
            folder.mkdir(parents=True)
            path = folder / "SKILL.md"
            with patch.object(skill_loader, "PROJECT_ROOT", str(root)):
                for content in invalid:
                    with self.subTest(content=content):
                        path.write_text(content, encoding="utf-8")
                        skill_loader._load_skill_file.cache_clear()
                        with self.assertRaises(ValueError):
                            skill_loader._load_skill_file(str(path), 6000)

    def test_old_config_without_new_routes_remains_valid(self):
        config = {
            "enabled": True,
            "max_instruction_chars": 6000,
            "subject_paths": {"数学": "skills/primary-math/SKILL.md"},
        }
        validate_skills_config(config)
        with patch.object(skill_loader, "_skills_config", return_value=config):
            self.assertEqual(skill_loader.load_subject_skills("综合"), [])
            self.assertIsNone(skill_loader.load_role_skill("learning_memory"))
            self.assertEqual(
                skill_loader.build_role_skill_context("learning_memory"),
                "本轮没有额外角色 Skill。",
            )

    def test_config_rejects_invalid_routes_and_budgets(self):
        invalid_changes = [
            {"enabled": "false"},
            {"max_instruction_chars": True},
            {"max_combined_instruction_chars": 0},
            {"role_paths": []},
            {"role_paths": {"learning_memory": ""}},
            {"combined_subjects": "数学"},
            {"combined_subjects": ["综合"]},
            {"combined_subjects": ["不存在"]},
        ]
        for change in invalid_changes:
            config = copy.deepcopy(app_conf["skills"])
            config.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_skills_config(config)


if __name__ == "__main__":
    unittest.main()
