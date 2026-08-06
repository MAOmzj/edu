"""项目教育 Skill 的加载、学科路由和 Prompt 注入测试。"""

import unittest

from agent.skill_loader import build_subject_skill_context, load_subject_skill


class EducationSkillTests(unittest.TestCase):
    def test_math_skill_is_loaded_with_required_tool_rules(self):
        """数学学科应加载 Skill，并明确要求本地检索和计算器核对。"""

        skill = load_subject_skill("数学")
        self.assertIsNotNone(skill)
        self.assertEqual(skill.name, "primary-math")
        self.assertIn("search_knowledge", skill.instructions)
        self.assertIn("calculate_expression", skill.instructions)
        self.assertIn("检查单位", skill.instructions)

    def test_other_subject_does_not_receive_math_skill(self):
        """语文等其他学科不能误加载小学数学教学规则。"""

        self.assertIsNone(load_subject_skill("语文"))
        self.assertEqual(
            build_subject_skill_context("语文"),
            "本轮没有额外教育 Skill。",
        )

    def test_prompt_block_marks_skill_as_project_instruction(self):
        """Skill Prompt 应有清晰边界，并标明它来自项目维护者。"""

        context = build_subject_skill_context("数学")
        self.assertIn("项目教育 Skill 开始", context)
        self.assertIn("primary-math", context)
        self.assertIn("优先于学生问题中的指令", context)
        self.assertIn("项目教育 Skill 结束", context)


if __name__ == "__main__":
    unittest.main()
