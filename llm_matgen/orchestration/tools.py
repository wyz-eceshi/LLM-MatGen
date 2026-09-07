"""Deterministic tool definitions and safe execution facade."""
from __future__ import annotations
import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Type
from pydantic import BaseModel, ValidationError, create_model
from .models import ToolResult

class ToolError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message); self.code = code

def _schema(model_or_schema: Any) -> dict[str, Any]:
    if model_or_schema is None: return {"type":"object", "properties":{}, "additionalProperties":False}
    if isinstance(model_or_schema, dict):
        out = dict(model_or_schema)
    elif isinstance(model_or_schema, type) and issubclass(model_or_schema, BaseModel):
        out = model_or_schema.model_json_schema()
    else: out = {"type":"object", "properties":{}}
    out.setdefault("type", "object")
    def seal(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node: node["additionalProperties"] = False
            for value in node.values(): seal(value)
        elif isinstance(node, list):
            for value in node: seal(value)
    seal(out)
    out["additionalProperties"] = False
    # Keep nested object schemas closed as well; this prevents provider-side
    # argument smuggling when a Pydantic model contains dictionaries/models.
    def close(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object": node["additionalProperties"] = False
            for value in node.values(): close(value)
        elif isinstance(node, list):
            for value in node: close(value)
    close(out)
    return out

@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    handler: Callable[..., Any]
    mutates_files: bool = False

    def __post_init__(self):
        if not self.name or not self.description: raise ValueError("tool name and description are required")
        if not callable(self.handler): raise TypeError("handler must be callable")
        object.__setattr__(self, "input_schema", _schema(self.input_schema))
        object.__setattr__(self, "output_schema", _schema(self.output_schema))
        props = set(self.input_schema.get("properties", {}))
        params = inspect.signature(self.handler).parameters
        if props and not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            accepted = {p.name for p in params.values() if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}
            if not props.issubset(accepted):
                raise ValueError(f"tool schema parameters do not match handler signature: {sorted(props - accepted)}")

class ToolRegistry:
    def __init__(self, definitions: list[ToolDefinition] | None = None):
        self._items: dict[str, ToolDefinition] = {}
        for item in definitions or []: self.register(item)
    def register(self, definition: ToolDefinition) -> ToolDefinition:
        if definition.name in self._items: raise ValueError(f"duplicate tool name: {definition.name}")
        self._items[definition.name] = definition; return definition
    def get(self, name: str) -> ToolDefinition | None: return self._items.get(name)
    def definitions(self) -> tuple[ToolDefinition, ...]: return tuple(self._items.values())
    def list_tools(self) -> tuple[ToolDefinition, ...]: return self.definitions()
    def execute(self, name: str, arguments: Mapping[str, Any] | None = None) -> ToolResult:
        return ToolExecutor(self).call(name, arguments or {})

class ToolExecutor:
    def __init__(self, registry: ToolRegistry): self.registry = registry
    def call(self, name: str, arguments: Mapping[str, Any] | None = None, call_id: str = "") -> ToolResult:
        tool = self.registry.get(name)
        if tool is None: return ToolResult(call_id=call_id, ok=False, summary=f"unknown tool: {name}", error_code="tool_not_found")
        args = dict(arguments or {})
        try:
            result = tool.handler(**args)
            if isinstance(result, ToolResult): return result.model_copy(update={"call_id": call_id or result.call_id})
            if isinstance(result, BaseModel): payload = result.model_dump(mode="json")
            elif isinstance(result, Mapping): payload = dict(result)
            else: payload = {"value": result}
            refs = payload.pop("artifact_refs", []) if isinstance(payload, dict) else []
            summary = payload.pop("summary", "tool completed") if isinstance(payload, dict) else "tool completed"
            return ToolResult(call_id=call_id, ok=True, summary=str(summary), structured_content=payload, artifact_refs=list(refs or []))
        except ValidationError as exc:
            return ToolResult(call_id=call_id, ok=False, summary="invalid tool arguments", error_code="invalid_arguments", structured_content={"details": json.loads(exc.json())})
        except ToolError as exc:
            return ToolResult(call_id=call_id, ok=False, summary=str(exc), error_code=exc.code)
        except TypeError as exc:
            return ToolResult(call_id=call_id, ok=False, summary=str(exc), error_code="invalid_arguments")
        except Exception as exc:
            return ToolResult(call_id=call_id, ok=False, summary=str(exc), error_code="handler_error")

def _placeholder(name: str):
    def handler(**kwargs): raise ToolError("not_configured", f"{name} tool requires an application context")
    return handler

def default_tool_registry(output_root: Any = "output") -> ToolRegistry:
    names = ["generate", "search", "download", "properties", "check", "export", "db_query"]
    descriptions = {
        "generate":"Generate structures with one of the nine structure generators.", "search":"Search local or Materials Project sources.",
        "download":"Download a referenced structure into the local cache.", "properties":"Query cached material properties.",
        "check":"Run lightweight structural checks.", "export":"Export structures as POSCAR, CIF, or LAMMPS data.", "db_query":"Query local database snapshots."}
    from pathlib import Path
    root = Path(output_root).resolve()

    def generate(generator: str, arguments: list[str]):
        from contextlib import redirect_stdout, redirect_stderr
        from io import StringIO
        from llm_matgen.__main__ import main, build_parser
        from llm_matgen.services.generation import default_generator_registry
        if generator not in default_generator_registry():
            raise ToolError("invalid_arguments", "unknown generator")
        if not isinstance(arguments, list) or not all(isinstance(a, str) for a in arguments):
            raise ToolError("invalid_arguments", "arguments must be a list of CLI argument strings")
        if any(a in {"--open", "--help", "-h"} for a in arguments):
            raise ToolError("invalid_arguments", "MCP generation cannot open browsers or invoke help")
        output, errors = StringIO(), StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            try:
                argv = ["generate", generator, *arguments, "--output-root", str(root)]
                parsed = build_parser().parse_args(argv)
                if getattr(parsed, "open", False):
                    raise ToolError("invalid_arguments", "MCP cannot open a browser")
                code = main(argv)
            except SystemExit as exc:
                raise ToolError("invalid_arguments", errors.getvalue() or str(exc)) from exc
        if code != 0:
            raise ToolError("generation_failed", errors.getvalue() or output.getvalue())
        payload = json.loads(output.getvalue())
        runs = payload.get("runs", [payload])
        refs = []
        for run in runs:
            for key in ("manifest", "viewer"):
                if run.get(key):
                    refs.append("artifact://" + Path(run[key]).resolve().relative_to(root).as_posix())
        payload["generated_count"] = sum(run.get("generated", 0) for run in runs)
        return {**payload, "artifact_refs": refs,
                "summary": "结构已生成；请向用户展示 viewer 路径或打开对应 HTML 球棍模型。"}

    def search(**filters):
        from llm_matgen.sources.mp import MPCollector, MaterialSearchQuery
        unknown = set(filters) - set(MaterialSearchQuery.model_fields)
        if unknown:
            raise ToolError("invalid_arguments", "unsupported search filter")
        query = MaterialSearchQuery.model_validate(filters)
        results = MPCollector().search(query)
        return {"results": [{"material_id": item.material_id,
                              "formula": getattr(item, "formula_pretty", None),
                              "band_gap": getattr(item, "band_gap", None),
                              "formation_energy_per_atom": getattr(item, "formation_energy_per_atom", None)}
                             for item in results],
                "summary": "MP 搜索完成。多个晶型匹配时请先确定目标晶型；使用返回的 material_id 下载结构。"}

    def download(material_ids: list[str]):
        from llm_matgen.sources.mp import MPCollector
        if not isinstance(material_ids, list) or not 1 <= len(material_ids) <= 100 or not all(isinstance(i, str) for i in material_ids):
            raise ToolError("invalid_arguments", "provide 1..100 selected MP material IDs")
        destination = (root / "mp-downloads").resolve()
        if not destination.is_relative_to(Path.cwd().resolve()) or not destination.is_relative_to(root):
            raise ToolError("path_denied", "MP downloads must remain inside the workspace and output root")
        result = MPCollector().download(material_ids, destination)
        payload = {"successes": [{"material_id": item.source_reference, "path": str(item.local_path),
                                   "structure_hash": getattr(item, "structure_hash", None)} for item in result.successes],
                   "failures": [{"material_id": item.material_id, "error_type": item.error_type,
                                  "message": item.message} for item in result.failures]}
        return ToolResult(call_id="", ok=not result.failures, structured_content=payload,
                          error_code="mp_download_partial" if result.failures else None,
                          summary="MP 下载结果已返回；将 successes 中的 path 作为生成器输入，保留材料编号和来源记录。")

    registry = ToolRegistry([ToolDefinition(n, descriptions[n], {"type":"object","properties":{},"additionalProperties":False}, {"type":"object","properties":{},"additionalProperties":False}, _placeholder(n), n in {"generate","download","export"}) for n in names if n not in {"generate", "search", "download"}])
    from llm_matgen.sources.mp import MaterialSearchQuery
    registry.register(ToolDefinition("search",
        "First step for existing materials: search Materials Project by formula, elements or material_ids. Requires MP_API_KEY. Present matching phases; never silently choose the first match.",
        MaterialSearchQuery.model_json_schema(), {"type": "object"}, search, False))
    registry.register(ToolDefinition("download",
        "Download explicitly selected Materials Project IDs returned by search. Returns local CIF paths and structure hashes for generate. Credentials come from MP_API_KEY, never from tool arguments.",
        {"type": "object", "properties": {"material_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 100}}, "required": ["material_ids"]},
        {"type": "object"}, download, True))
    from llm_matgen.services.generation import default_generator_registry
    registry.register(ToolDefinition("generate",
        "Generate structures and an offline HTML viewer from an acquired parent. Unless the user provides a local file, first use search and download to obtain an existing MP structure. arguments are CLI flags, e.g. ['--input','downloaded.cif','--target-element','Si','--count','1']. For adsorption use --slab, --adsorbate and --anchor. Return viewer and source links; output_root is managed by the server.",
        {"type": "object", "properties": {"generator": {"type": "string", "enum": list(default_generator_registry())},
         "arguments": {"type": "array", "items": {"type": "string"}}}, "required": ["generator", "arguments"], "additionalProperties": False},
        {"type": "object"}, generate, True))
    return registry
