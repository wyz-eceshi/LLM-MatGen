import json
from pathlib import Path
from types import SimpleNamespace

from pymatgen.core import Lattice, Structure


def test_mcp_search_download_then_generate(tmp_path, monkeypatch):
    from llm_matgen.orchestration.tools import ToolExecutor, default_tool_registry
    import llm_matgen.sources.mp as mp
    events = []

    class FakeCollector:
        def search(self, query):
            events.append(('search', query.formula))
            return [SimpleNamespace(material_id='mp-149', formula_pretty='Si', band_gap=1.1,
                                    formation_energy_per_atom=0., classification=None)]

        def download(self, ids, output_dir):
            events.append(('download', ids))
            output_dir.mkdir(parents=True, exist_ok=True)
            p = output_dir / 'mp-149.cif'
            Structure(Lattice.cubic(5), ['Si', 'Si'], [[0, 0, 0], [.5, .5, .5]]).to(filename=str(p))
            return SimpleNamespace(successes=[SimpleNamespace(source_reference='mp-149', local_path=p,
                                   structure_hash='fixture')], failures=[])

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mp, 'MPCollector', FakeCollector)
    executor = ToolExecutor(default_tool_registry(tmp_path / 'out'))
    search = executor.call('search', {'formula': 'Si', 'limit': 5})
    assert search.ok, search.summary
    assert search.structured_content['results'][0]['material_id'] == 'mp-149'
    download = executor.call('download', {'material_ids': ['mp-149']})
    assert download.ok, download.summary
    path = download.structured_content['successes'][0]['path']
    assert Path(path).is_file()
    generated = executor.call('generate', {'generator': 'vacancy', 'arguments': [
        '--input', path, '--target-element', 'Si', '--count', '1']})
    assert generated.ok, generated.summary
    assert Path(generated.structured_content['runs'][0]['viewer']).is_file()
    assert events == [('search', 'Si'), ('download', ['mp-149'])]


def test_mp_search_missing_key_is_explicit(tmp_path, monkeypatch):
    from llm_matgen.orchestration.tools import ToolExecutor, default_tool_registry
    monkeypatch.delenv('MP_API_KEY', raising=False)
    result = ToolExecutor(default_tool_registry(tmp_path)).call('search', {'formula': 'Si'})
    assert not result.ok
    assert 'MP_API_KEY' in result.summary


def test_prompt_prefers_mp_without_choosing_first_result():
    from llm_matgen.orchestration.prompts import build_system_prompt
    prompt = build_system_prompt()
    assert 'Materials Project' in prompt
    assert 'search' in prompt and 'download' in prompt
    assert 'first result' in prompt
