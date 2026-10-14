from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReadingUIContractTests(unittest.TestCase):
    def test_existing_routes_controls_and_escaping_stay_present(self):
        text = (ROOT / "static/research.html").read_text(encoding="utf-8")
        for route in ("'/history'", "'/research/import'", "'/research/adaptive'"):
            self.assertIn(route, text)
        for control in ("token", "paper", "strategy", "depth", "pdf", "confirmEmbedding", "question", "submit", "sources"):
            self.assertIn(f'id="{control}"', text)
        self.assertIn("item.textContent", text)
        self.assertNotIn("innerHTML", text)

    def test_access_memory_is_optional_tab_scoped_and_clearable(self):
        script = (ROOT / "static/research-access.js").read_text(encoding="utf-8")
        self.assertIn("sessionStorage.setItem", script)
        self.assertIn("sessionStorage.removeItem", script)
        self.assertIn("remember.checked && token.value.trim()", script)
        self.assertNotIn("localStorage.setItem", script)
        for page in ("research.html", "admin-traces.html"):
            text = (ROOT / "static" / page).read_text(encoding="utf-8")
            self.assertIn('id="rememberToken"', text)
            self.assertIn('id="clearToken"', text)
            self.assertIn('/research-access.js?v=20261014', text)
