import hashlib
import json
import re

from pymatgen.core import Lattice, Structure


def test_viewer_preserves_coordinates_and_escapes_labels(tmp_path):
    from llm_matgen.viewer import write_viewer

    structure = Structure(Lattice.cubic(4), ['Si', 'Si'], [[0, 0, 0], [.25, .25, .25]])
    before = structure.as_dict()
    artifact = write_viewer([('</script><script>alert(1)</script>', structure)], tmp_path / 'viewer.html')
    text = artifact.path.read_text(encoding='utf-8')
    payload = json.loads(re.search(r'<script id="structure-data" type="application/json">(.*?)</script>', text, re.S).group(1))
    assert payload['entries'][0]['atoms'][1]['position'] == [1.0, 1.0, 1.0]
    assert payload['entries'][0]['atoms'][1]['number'] == 2
    assert '</script><script>alert(1)</script>' not in text
    assert not re.search(r'<script[^>]+src=', text)
    assert hashlib.sha256(artifact.path.read_bytes()).hexdigest() == artifact.sha256
    assert structure.as_dict() == before


def test_generation_exports_registered_viewer(tmp_path):
    from llm_matgen.generators.vacancy import VacancyGenerator, VacancyParams
    from llm_matgen.io.exporters import ExportOptions
    from llm_matgen.pipeline import GenerationPipeline

    structure = Structure(Lattice.cubic(5), ['Si', 'Si'], [[0, 0, 0], [.5, .5, .5]])
    result = GenerationPipeline(tmp_path).run(VacancyGenerator(), structure,
        VacancyParams(target_elements=['Si'], counts=[1], seed=1), ExportOptions())
    assert result.ok
    assert result.viewer_path.is_file()
    manifest = json.loads(result.manifest_path.read_text(encoding='utf-8'))
    assert any(a['format'] == 'html' and a['path'] == 'viewer.html' for a in manifest['artifacts'])


def test_mcp_can_read_generated_html(tmp_path):
    from llm_matgen.mcp.server import MCPServer
    from llm_matgen.orchestration.tools import ToolRegistry
    root = tmp_path / 'run'
    root.mkdir()
    (root / 'viewer.html').write_text('<html>结构</html>', encoding='utf-8')
    (root / 'manifest.json').write_text(json.dumps({'artifacts': [{'path': 'viewer.html'}]}))
    server = MCPServer(ToolRegistry(), output_root=tmp_path)
    response = server.handle({'jsonrpc': '2.0', 'id': 1, 'method': 'resources/read', 'params': {'uri': 'artifact://run/viewer.html'}})
    assert response['result']['contents'][0]['mimeType'] == 'text/html'


def test_mcp_generate_returns_viewer(tmp_path, monkeypatch):
    from llm_matgen.orchestration.tools import default_tool_registry, ToolExecutor
    monkeypatch.chdir(tmp_path)
    Structure(Lattice.cubic(5), ['Si', 'Si'], [[0, 0, 0], [.5, .5, .5]]).to(filename='input.cif')
    result = ToolExecutor(default_tool_registry(output_root=tmp_path / 'out')).call(
        'generate', {'generator': 'vacancy', 'arguments': ['--input', 'input.cif', '--target-element', 'Si', '--count', '1']})
    assert result.ok, result.summary
    assert result.structured_content['runs'][0]['viewer'].endswith('viewer.html')
    assert any('viewer.html' in ref for ref in result.artifact_refs)


def test_viewer_failure_does_not_lose_structures(tmp_path, monkeypatch):
    import llm_matgen.viewer
    from llm_matgen.generators.vacancy import VacancyGenerator, VacancyParams
    from llm_matgen.io.exporters import ExportOptions
    from llm_matgen.pipeline import GenerationPipeline
    def fail(*args, **kwargs):
        raise ValueError('preview budget exceeded')
    monkeypatch.setattr(llm_matgen.viewer, 'write_viewer', fail)
    structure = Structure(Lattice.cubic(5), ['Si', 'Si'], [[0, 0, 0], [.5, .5, .5]])
    result = GenerationPipeline(tmp_path).run(VacancyGenerator(), structure,
        VacancyParams(target_elements=['Si'], counts=[1], seed=1), ExportOptions())
    assert result.ok and result.viewer_path is None
    assert result.artifacts[0].path.is_file()
    manifest = json.loads(result.manifest_path.read_text(encoding='utf-8'))
    assert any('preview budget exceeded' in w for w in manifest['warnings'])


def test_generate_open_uses_created_page(tmp_path, monkeypatch, capsys):
    import webbrowser
    from llm_matgen.__main__ import main
    monkeypatch.chdir(tmp_path)
    Structure(Lattice.cubic(5), ['Si', 'Si'], [[0, 0, 0], [.5, .5, .5]]).to(filename='input.cif')
    opened = []
    monkeypatch.setattr(webbrowser, 'open', lambda uri: opened.append(uri))
    assert main(['generate', 'vacancy', '--input', 'input.cif', '--target-element', 'Si', '--count', '1', '--open']) == 0
    payload = json.loads(capsys.readouterr().out)
    from pathlib import Path
    assert opened == [Path(payload['runs'][0]['viewer']).as_uri()]


def test_mcp_rejects_browser_flag_abbreviation(tmp_path, monkeypatch):
    from llm_matgen.orchestration.tools import default_tool_registry, ToolExecutor
    result = ToolExecutor(default_tool_registry(tmp_path)).call('generate', {
        'generator': 'vacancy', 'arguments': ['--input', 'input.cif', '--target-element', 'Si', '--count', '1', '--op']})
    assert not result.ok
    assert result.error_code == 'invalid_arguments'
