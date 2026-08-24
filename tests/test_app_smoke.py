from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"


class TestAppSmoke:
    def test_app_runs_without_exception(self, monkeypatch):
        monkeypatch.chdir(APP_PATH.parent)
        at = AppTest.from_file(str(APP_PATH), default_timeout=60)
        at.run()
        assert not at.exception

    def test_diagnosis_button_produces_results(self, monkeypatch):
        monkeypatch.chdir(APP_PATH.parent)
        at = AppTest.from_file(str(APP_PATH), default_timeout=60)
        at.run()
        at.button[0].click().run()
        assert not at.exception
        subheaders = [s.value for s in at.subheader]
        assert any("貯蓄の立ち位置" in s for s in subheaders)
        assert any("同年代×同収入" in s for s in subheaders)
        assert any("同じ収入階級" in s for s in subheaders)
        assert any("家計タイプマップ" in s for s in subheaders)
