from __future__ import annotations

import unittest
import zipfile
from io import BytesIO
from xml.etree import ElementTree

import report_docx


class ReportDocxTests(unittest.TestCase):
    def test_current_report_text_and_links_are_readable_in_docx_xml(self) -> None:
        report = """# 智能康复评估报告

## 四、康复策略建议

基于本次结构化观察结果相关知识检索主题：锚点：Brunnstrom手功能分期（hand_function）=6期

### 二、按 Brunnstrom 分期的训练动作

- 精细操作：练习抓取

## 六、依据来源与参考文献

【1】[指南标题](https://example.test/guideline.pdf)
"""
        data = report_docx.markdown_to_docx_bytes(report)
        with zipfile.ZipFile(BytesIO(data)) as archive:
            document = archive.read("word/document.xml")
            styles = archive.read("word/styles.xml")

        root = ElementTree.fromstring(document)
        text = "".join(root.itertext())
        self.assertIn("按 Brunnstrom 分期的训练动作", text)
        self.assertIn("精细操作：练习抓取", text)
        self.assertIn("指南标题（https://example.test/guideline.pdf）", text)
        self.assertNotIn("检索主题", text)
        self.assertNotIn("hand_function", text)
        self.assertIn("eastAsia=\"Microsoft YaHei\"", styles.decode("utf-8"))
        self.assertNotIn("[指南标题](https://example.test/guideline.pdf)", text)


if __name__ == "__main__":
    unittest.main()
