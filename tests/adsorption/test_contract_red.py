import importlib.util


def test_adsorption_case_contract_module_is_available():
    """Break caught: shipping the subsystem without its public contracts."""
    assert importlib.util.find_spec("llm_matgen.adsorption.models") is not None
