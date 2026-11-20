"""P0 命令组契约。"""
import pytest

from llm_matgen.__main__ import main


@pytest.mark.parametrize(
    ("group", "names"),
    [
        ("mof", "bond-graph clean graph-diff extract-subnet scan-placement node-substitute remove-fragment replace-fragment"),
        ("hetero", "measure build check"),
    ],
)
def test_group_help(group, names, capsys):
    with pytest.raises(SystemExit) as exc:
        main([group, "--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    for name in names.split():
        assert name in help_text
