import json

from rich.console import Console

from phonoflow.resolved_settings import ResolvedSetting, ResolvedSettings


def test_resolved_setting_serializes_value_source_and_note():
    setting = ResolvedSetting("backend_resolved", "calorine", "auto", "Calorine available")
    assert setting.to_dict() == {
        "value": "calorine",
        "source": "auto",
        "note": "Calorine available",
    }


def test_resolved_settings_write_json_yaml_and_print_table(tmp_path, capsys):
    settings = ResolvedSettings()
    settings.add("backend_requested", "auto", "default")
    settings.add("backend_resolved", "calorine", "auto", "Calorine CPUNEP available")
    settings.add("supercell_dim_resolved", [3, 3, 3], "auto")

    json_path = tmp_path / "resolved_settings.json"
    yaml_path = tmp_path / "resolved_settings.yaml"
    settings.write_json(json_path)
    settings.write_yaml(yaml_path)
    settings.print_table(Console(record=True))
    capsys.readouterr()
    table_text = settings.write_table(tmp_path / "resolved_settings_table.txt")
    captured = capsys.readouterr()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["backend_requested"]["source"] == "default"
    assert data["backend_resolved"]["source"] == "auto"
    assert data["supercell_dim_resolved"]["value"] == [3, 3, 3]
    assert "backend_resolved" in yaml_path.read_text(encoding="utf-8")
    assert "Resolved PhonoFlow settings" in table_text
    assert captured.out == ""
