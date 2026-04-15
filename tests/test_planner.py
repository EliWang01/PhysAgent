from physagent.agents.planner import _parse_planner_output, _format_history


def test_parse_valid_json():
    text = '{"composition": "CsSnI3", "space_group": 221, "reasoning": "Sn lowers bandgap"}'
    result = _parse_planner_output(text)
    assert result["composition"] == "CsSnI3"
    assert result["space_group"] == 221
    assert "Sn" in result["reasoning"]


def test_parse_json_in_markdown_code_block():
    text = '```json\n{"composition": "CsPbBr3", "space_group": 221, "reasoning": "test"}\n```'
    result = _parse_planner_output(text)
    assert result["composition"] == "CsPbBr3"


def test_parse_invalid_json_fallback():
    text = "I think we should try CsPbI3 because..."
    result = _parse_planner_output(text)
    assert result["composition"] == "CsPbI3"  # Safe default
    assert result["space_group"] == 221


def test_format_history_empty():
    assert _format_history([]) == "无历史记录"


def test_format_history_entries():
    history = [
        {"iteration": 0, "composition": "CsPbI3", "verdict": "REVISE",
         "score": 0.5, "suggestion": "Try Sn"},
    ]
    result = _format_history(history)
    assert "CsPbI3" in result
    assert "REVISE" in result
