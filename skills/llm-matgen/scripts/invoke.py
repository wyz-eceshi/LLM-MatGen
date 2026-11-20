import argparse
import contextlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys

BASE = Path(__file__).resolve().parents[1]
if (BASE / 'vendor').is_dir():
    sys.path.insert(0, str(BASE / 'vendor'))

COMMANDS = {
    'build-info': ['build-info'],
    'mp-search': ['search'],
    'mp-download': ['download'],
    'inspect': ['structure', 'inspect'],
    'validate': ['structure', 'validate'],
    'adjust-coordinate': ['structure', 'adjust-coordinate'],
    'revision-import': ['revision', 'import'],
    'export': ['export'],
}
GENERATORS = {'vacancy', 'interstitial', 'doping', 'solid-solution', 'surface',
              'grain-boundary', 'interface', 'stacking-fault', 'dislocation',
              'adsorption', 'symmetry-crystal'}
PATH_KEYS = {'input', 'film', 'substrate', 'slab', 'adsorbate', 'gas-reference', 'atom-map', 'guest',
             'archive-db', 'recipe', 'output-dir', 'output-root', 'output', 'sidecar', 'parent', 'revised'}
DENIED = {'open', 'history-policy', 'run-id'}

def checked_path(value, root, *, new=False):
    if not isinstance(value, str) or not value or any(ord(c) < 32 for c in value):
        raise ValueError('路径必须非空且不含C0控制字符')
    # Reject Windows device namespaces, alternate streams and special filenames.
    value = value.replace('\\', '/')
    if value.startswith('//') or ':' in value[2:] or re.match(r'^[^/]:[^/]', value):
        raise ValueError('拒绝设备路径、UNC或备用数据流')
    for part in value.split('/'):
        if part in {'.', '..'}:
            continue
        if part.endswith((' ', '.')) or re.match(r'(?i)^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(\.|$)', part):
            raise ValueError('拒绝Windows特殊文件名')
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_relative_to(root):
        raise ValueError('路径必须位于显式选择的工作项目内')
    if new and path.exists():
        raise ValueError('输出已存在，不覆盖')
    return path

def scalar(v):
    if isinstance(v, bool) or not isinstance(v, (str, int, float)):
        raise ValueError('参数值必须是字符串或有限数字')
    if isinstance(v, float) and not math.isfinite(v):
        raise ValueError('参数值必须有限')
    if isinstance(v, str) and (any(ord(c) < 32 for c in v) or v.startswith('-')):
        raise ValueError('拒绝控制字符和选项形式的值')
    return str(v)

def leaf_parser(parser, command):
    for part in command:
        actions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
        parser = actions[0].choices[part]
    return parser

def make_argv(req, root):
    from llm_matgen.__main__ import build_parser
    op = req['operation']
    if op in {'mp-search', 'mp-download'} and req.get('allow_mp') is not True:
        raise ValueError('MP联网必须有本次用户明确请求并设置allow_mp=true')
    command = COMMANDS.get(op)
    if op == 'mof':
        subcommand = req.get('subcommand')
        if subcommand not in ('bond-graph', 'clean', 'graph-diff', 'extract-subnet', 'scan-placement', 'node-substitute', 'remove-fragment', 'replace-fragment'):
            raise ValueError('MOF 操作不在 MOF 白名单')
        command = ['mof', subcommand]
    elif op == 'hetero':
        subcommand = req.get('subcommand')
        if subcommand not in ('measure', 'build', 'check'):
            raise ValueError('未知异质结操作')
        command = ['hetero', subcommand]
    elif 'subcommand' in req:
        raise ValueError('该操作不接收 subcommand')
    if op == 'generate':
        name = req.get('generator')
        if name not in GENERATORS:
            raise ValueError('不支持的生成器')
        command = ['generate', name]
    if command is None:
        raise ValueError('操作不在白名单')
    parser = leaf_parser(build_parser(), command)
    allowed = {a.option_strings[0][2:]: a for a in parser._actions
               if a.option_strings and a.option_strings[0].startswith('--')
               and a.option_strings[0][2:] not in DENIED | {'help'}}
    args = req.get('arguments', {})
    if not isinstance(args, dict) or set(args) - set(allowed):
        raise ValueError('未知或被禁止的参数')
    if op == 'mp-download' and 'output-dir' not in args:
        raise ValueError('MP下载必须显式指定output-dir')
    if op in {'generate', 'export', 'revision-import'} and 'output-root' not in args:
        raise ValueError('必须显式指定output-root')
    if op == 'adjust-coordinate':
        index = args.get('index')
        if type(index) is not int or index < 1:
            raise ValueError('index必须是从1开始的整数；本接口不转换为0-based')
        if not isinstance(args.get('reason'), str) or not args['reason'].strip():
            raise ValueError('reason不能为空')
    argv = list(command)
    if op == 'mp-download':
        ids = req.get('material_ids')
        if not isinstance(ids, list) or not ids or any(not isinstance(v,str) or not re.fullmatch(r'mp-[0-9]+',v) for v in ids):
            raise ValueError('material_ids必须是非空mp-数字数组')
        argv.extend(ids)
    if op == 'export':
        paths = req.get('paths')
        if not isinstance(paths, list) or not paths:
            raise ValueError('export需要非空paths数组')
        argv.extend(str(checked_path(v, root)) for v in paths)
    for key, value in args.items():
        action = allowed[key]
        flag = '--' + key
        if isinstance(action, argparse._StoreTrueAction):
            if type(value) is not bool:
                raise ValueError('开关必须为JSON布尔值')
            if value:
                argv.append(flag)
            continue
        values = value if isinstance(action, argparse._AppendAction) else [value]
        if key == "displace" and isinstance(value, (str, int)) and not isinstance(value, bool):
            values = [value]
        if not isinstance(values, list) or not values:
            raise ValueError('可重复参数需非空数组')
        for item in values:
            if key in PATH_KEYS:
                if isinstance(item, str) and item.startswith('mp:'):
                    raise ValueError('本安装包仅接本地文件；MP检索需单独明确授权的公共入口')
                argv += [flag, str(checked_path(item, root, new=key in {'output','output-root','output-dir'} or (key=='sidecar' and op=='adjust-coordinate')))]
            elif key == 'bond-path':
                if not isinstance(item,list) or len(item)!=2 or any(type(i) is not int or i<1 for i in item):
                    raise ValueError('bond-path 端点必须是两个从 1 开始的整数')
                argv += [flag, *map(str,item)]
            elif action.nargs in (3, 2):
                if not isinstance(item, list) or len(item) != action.nargs:
                    raise ValueError('坐标必须是正确长度的数组')
                argv += [flag, *map(scalar, item)]
            else:
                argv += [flag, scalar(item)]
    if op == 'generate' and req.get('generator') == 'adsorption':
        anchor = args.get('anchor')
        if type(anchor) is not int or anchor < 1:
            raise ValueError('anchor必须是从1开始的整数')
        argv += ['--history-policy', 'off']
    return argv

def execute(req, workspace):
    if not isinstance(req, dict) or set(req) - {'version','operation','subcommand','generator','arguments','paths','structures','output','allow_mp','material_ids'}:
        raise ValueError('请求包含未知字段；不是任意shell或Harness官方JSON RPC')
    if req.get('version') != 1:
        raise ValueError('version必须为1')
    root = Path(workspace).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('工作项目必须是目录')
    old = Path.cwd()
    os.chdir(root)
    try:
        if req.get('operation') == 'viewer':
            from llm_matgen.io.readers import read_structure
            from llm_matgen.viewer import write_viewer
            structures = req.get('structures')
            if not isinstance(structures, list) or not structures or len(structures) > 16:
                raise ValueError('viewer需1至16个本地结构路径')
            output = checked_path(req.get('output'), root, new=True)
            loaded = [(Path(s).name, read_structure(checked_path(s, root))) for s in structures]
            artifact = write_viewer(loaded, output)
            return 0, json.dumps({'path': str(artifact.path), 'sha256':artifact.sha256}, ensure_ascii=False), ''
        argv = make_argv(req, root)
        from llm_matgen.__main__ import main
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = main(argv)
            except SystemExit as exc:
                code = exc.code
        return int(code or 0), out.getvalue(), err.getvalue()
    finally:
        os.chdir(old)

def main():
    parser = argparse.ArgumentParser(description='MatGen本地白名单JSON适配器（自有合同v1）')
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--request', required=True)
    args = parser.parse_args()
    try:
        raw = Path(args.request).read_bytes()
        if len(raw) > 65536:
            raise ValueError('请求上限64KiB')
        req = json.loads(raw.decode('utf-8-sig'), parse_constant=lambda v: (_ for _ in ()).throw(ValueError('非有限数字')))
        code, out, err = execute(req, args.workspace)
        print(out, end='')
        print(err, end='', file=sys.stderr)
        return code
    except Exception as exc:
        print(json.dumps({'ok':False,'error':str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2

if __name__ == '__main__':
    sys.exit(main())

