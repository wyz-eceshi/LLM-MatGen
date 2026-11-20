"""发布前按输入配方逐位核对临时目录中的两份结构文件。"""
from collections import Counter
import json
from pathlib import Path

from .cif import read_model


def charge_ledger(parent, removed, added):
    values = (parent, removed, added)
    if any(value is not None and type(value) is not int for value in values):
        raise ValueError("形式电荷须为整数或 null（未知）")
    child = parent - removed + added if None not in values else None
    return dict(parent=parent, removed=removed, added=added, child=child)


def verify_output(out, parent_symbols, removed, added, charges, *, expected_sites=None):
    parent = Counter(parent_symbols)
    expected = parent.copy()
    expected.subtract(row["element"] for row in removed)
    expected.update(row["element"] for row in added)
    if any(count < 0 for count in expected.values()):
        raise ValueError("组成收支出现负数")
    expected = +expected
    for name in ("structure.cif", "POSCAR"):
        try:
            snapshot = read_model(out / name)
            sites = [site.specie.symbol for site in snapshot.structure]
        except (KeyError, IndexError, TypeError, OSError, ValueError) as exc:
            raise ValueError(f"无法回读并核对 {name}：{exc}") from exc
        actual = Counter(sites)
        if (any(o != 1 for o in snapshot.occupancy) or actual != expected
                or (expected_sites is not None and list(expected_sites) != sites)):
            raise ValueError(f"{name} 回读与输入原子账不符，预期 {dict(expected)}，实际 {dict(actual)}")
    return dict(composition_ledger=dict(parent=dict(parent), removed=removed, added=added,
                                        expected_child=dict(expected), actual_child=dict(actual)),
                formal_charge_ledger=charges)


def _template(path):
    from pymatgen.core import Molecule
    from .fragment_ops import sdf_template
    path = Path(path)
    if path.suffix.lower() == ".sdf":
        symbols, _, _ = sdf_template(path)
        lines = path.read_text("utf-8").splitlines()
        # V2000 的 M CHG 记录优先于原子块中的电荷编码。
        charge_codes = {0:0, 1:3, 2:2, 3:1, 4:0, 5:-1, 6:-2, 7:-3}
        charges = [charge_codes[int(line[36:39].strip() or 0)] for line in lines[4:4+len(symbols)]]
        records = [line.split() for line in lines if line.startswith("M  CHG")]
        if records:
            charges = [0] * len(symbols)
            for fields in records:
                for index, value in zip(fields[3::2], fields[4::2], strict=True):
                    charges[int(index)-1] = int(value)
        return symbols, charges, None
    if path.suffix.lower() in (".json", ".mson"):
        data = json.loads(path.read_text("utf-8"))
        mol = Molecule.from_dict(data)
        total = data.get("charge")
        if total is not None and float(total).is_integer():
            total = int(total)
    else:
        mol = Molecule.from_file(path)
        total = None  # XYZ 坐标块本身没有形式电荷字段。
    return [s.specie.symbol for s in mol], None, total


def _removed_charge(parent, removed):
    if not removed:
        return 0
    fragments = [f for f in parent.fragments() if removed.intersection(f["atoms"])]
    if any(f["formal_charge"] is None or not removed.issuperset(f["atoms"]) for f in fragments):
        return None
    return sum(f["formal_charge"] for f in fragments)


def source_charge(structure, source):
    from .sidecar import sha256
    path = Path(source)
    meta = path.with_name("sidecar.json")
    if meta.is_file():
        data = json.loads(meta.read_text("utf-8"))
        if sha256(source) in (data.get("product_sha256"), data.get("poscar_sha256")):
            value = data.get("formal_charge_ledger", {}).get("child")
            if value is not None:
                return value
    if all(hasattr(s.specie, "oxi_state") for s in structure):
        total = sum(s.specie.oxi_state for s in structure)
        if float(total).is_integer():
            return int(total)
    return None


def _template_rows(path, params, report, new_sites, role):
    from .sidecar import sha256
    if path is None or sha256(path) != params.get("guest_sha256"):
        raise ValueError("缺少与配方一致的模板")
    symbols, charges, total = _template(path)
    ids = report.get("template_indices", list(range(1, len(symbols)+1)))
    omitted = report.get("template_discarded", [])
    if any(not isinstance(row.get("reason"), str) or not row["reason"].strip() for row in omitted):
        raise ValueError("丢弃模板原子须逐项注明原因")
    partition = list(ids) + [row["index"] for row in omitted]
    if (any(type(i) is not int for i in partition) or
            sorted(partition) != list(range(1, len(symbols)+1)) or len(new_sites) != len(ids)):
        raise ValueError("模板原子未全部写入或未显式声明丢弃")
    params["template_discarded"] = list(omitted)
    added = [dict(index=child, element=symbols[i-1], source=role, template_index=i)
             for child, i in zip(new_sites, ids, strict=True)]
    provenance = dict(sha256=params["guest_sha256"], atom_count=len(symbols),
                      atom_map={str(child): i for child, i in zip(new_sites, ids, strict=True)},
                      discarded=omitted)
    return added, provenance, (sum(charges[i-1] for i in ids) if charges is not None else total)


def _declared(charges, key, known):
    value = charges.get(key, known)
    if known is not None and value != known:
        raise ValueError(f"声明的 {key} 形式电荷与输入不符")
    return value


def _planned_cap_rows(parent, gone, recipe, report, mapping):
    """用父图断开的周期边逐项重算封端，核对位点与实际写入次序。"""
    from .cap_plan import describe_plan
    sites, charges = describe_plan(parent, {i-1 for i in gone}, recipe)
    actual = report.get("cap_sites", [])
    if len(actual) != len(sites) or report.get("cap_charge") != charges:
        raise ValueError("封端空位或电荷收支与父结构不符")
    paired_caps = {}
    rows = []
    new_indices = [i for i,p in enumerate(mapping,1) if p is None]
    for site, written in zip(sites, actual, strict=True):
        if any(site[key] != written.get(key) for key in ("metal_parent","removed_parent","removed_image")):
            raise ValueError("封端的金属、原配位原子或周期像不符")
        cap = written.get("cap")
        if cap not in ("OH","H2O"):
            raise ValueError("未知封端类型")
        if recipe["cap_plan"] == "auto-charge" and "carboxyl_parent" in site:
            carbon = site["carboxyl_parent"]
            if carbon != written.get("carboxyl_parent") or site["carboxyl_image"] != written.get("carboxyl_image"):
                raise ValueError("封端对应的原羧酸位点不符")
            paired_caps.setdefault(carbon, []).append(cap)
        elif cap != site["cap"]:
            raise ValueError("写出的封端类型与配方不符")
        symbols = {"OH": ("O","H"), "H2O": ("O","H","H")}[cap]
        indices = new_indices[len(rows):len(rows)+len(symbols)]
        if (indices != written.get("cap_indices") or len(indices) != len(symbols)
                or written.get("metal_child") != mapping.index(site["metal_parent"])+1):
            raise ValueError("封端原子的子编号与父金属映射不符")
        rows.extend(dict(index=i, element=s, source="cap", cap=cap,
                         metal_parent=site["metal_parent"], removed_parent=site["removed_parent"],
                         removed_image=list(site["removed_image"])) for i,s in zip(indices,symbols,strict=True))
    if len(rows) != len(new_indices):
        raise ValueError("封端配方未覆盖所有新增位点")
    if any(sorted(caps) != ["H2O","OH"] for caps in paired_caps.values()):
        raise ValueError("每个原羧酸位点须分别补一个 OH 和一个 H2O")
    return rows, charges


def _guest_fragment_ledger(template, recipe, integrity):
    """去水等操作发生于客体进胞前；保留独立的客体进出账，避免从宿主扣除客体原子。"""
    from .donor import resolve_displacements
    fragments = resolve_displacements(template, recipe)
    symbols, site_charges, template_charge = _template(template)
    expected = sorted((idx,f["name"]) for f in fragments for idx in f["template_indices"])
    omitted = sorted((row["index"],row.get("fragment","")) for row in integrity["discarded"])
    if expected != omitted:
        raise ValueError("客体实际丢弃的原子与 displace 指定片段不符")
    removed = [dict(index=i, element=symbols[i-1], source="guest", fragment=name) for i,name in expected]
    retained = Counter(symbols[i-1] for i in integrity["atom_map"].values())
    if retained + Counter(r["element"] for r in removed) != Counter(symbols):
        raise ValueError("客体片段的移除与保留组成不守恒")
    fragment_charges = []
    for part in fragments:
        q = part.get("formal_charge")
        if site_charges is not None:
            q = _declared({"removed":q} if q is not None else {},"removed",
                          sum(site_charges[i-1] for i in part["template_indices"]))
        fragment_charges.append(q)
    removed_charge = None if None in fragment_charges else sum(fragment_charges)
    if site_charges is not None:
        template_charge = sum(site_charges)
    template_charge = _declared(recipe,"guest_charge",template_charge)
    remaining_charge = charge_ledger(template_charge,removed_charge,0)["child"]
    return dict(parent=dict(Counter(symbols)),removed=removed,retained=dict(retained),
                template_charge=template_charge,removed_charge=removed_charge,retained_charge=remaining_charge)


def mof_output(out, source, operation, parent, mapping, params, report, template):
    recipe = params["recipe"]
    declared = recipe.get("formal_charges", {})
    if not isinstance(declared, dict):
        raise ValueError("formal_charges 须为分项电荷")
    source_q = _declared(declared, "parent", source_charge(parent.structure, source))
    count = len(parent.symbols)
    new_sites = [j for j, p in enumerate(mapping, 1) if p is None]
    retained = [p for p in mapping if p is not None]
    gone = set(report.get("removed_atoms", []))
    if (any(type(i) is not int or not 1 <= i <= count for i in retained)
            or len(set(retained)) != len(retained)
            or gone != set(range(1, count+1)).difference(retained)):
        raise ValueError("父位点映射与声明的删除清单不符")
    expected = [parent.symbols[p-1] if p is not None else None for p in mapping]
    removed = [dict(index=i, element=parent.symbols[i-1]) for i in sorted(gone)]
    added = []
    integrity = None
    guest_ledger = None
    cap_charges = None
    removed_q = _removed_charge(parent, gone)
    if operation in ("replace-fragment", "scan-placement"):
        added, integrity, added_q = _template_rows(template, params, report, new_sites,
                  "template" if operation == "replace-fragment" else "guest")
        if operation == "scan-placement" and recipe.get("displace") is not None:
            guest_ledger = _guest_fragment_ledger(template, recipe, integrity)
            added_q = guest_ledger["retained_charge"]
        elif operation == "scan-placement" and integrity["discarded"]:
            raise ValueError("放置时丢弃客体原子须指定 displace")
        elif operation == "scan-placement":
            added_q = _declared(recipe,"guest_charge",added_q)
    elif operation == "remove-fragment" and "cap_plan" in recipe:
        added, cap_charges = _planned_cap_rows(parent, gone, recipe, report, mapping)
        added_q = cap_charges["added"]
    elif operation == "remove-fragment":
        cap = recipe.get("cap", "none")
        group = {"none": [], "OH": ["O","H"], "H2O": ["O","H","H"]}[cap]
        ends = sum(parent.elements[j].is_metal and j+1 not in gone
                   for i in gone for j, _ in parent.neighbors[i-1]) if group else 0
        symbols = group * ends
        if len(new_sites) != len(symbols):
            raise ValueError("封端原子数与原配位空位不符")
        added = [dict(index=j, element=s, source="cap", cap=cap)
                 for j, s in zip(new_sites, symbols, strict=True)]
        added_q = -ends if cap == "OH" else 0
    elif operation == "node-substitute":
        if new_sites or gone or retained != list(range(1,count+1)):
            raise ValueError("节点替换改变了位点映射")
        for i, s in zip(params["sites"], params["assignment"], strict=True):
            old = expected[i-1]
            if old != s:
                removed.append(dict(index=i, element=old))
                added.append(dict(index=i, element=s, source="substitution", parent_index=i))
        added_q = None if added else 0
        removed_q = None if removed else 0
    else:
        raise ValueError(f"缺少 {operation} 的组成核对")
    for row in added:
        expected[row["index"]-1] = row["element"]
    removed_q = _declared(declared, "removed", removed_q)
    added_q = _declared(declared, "added", added_q)
    charges = charge_ledger(source_q, removed_q, added_q)
    if cap_charges is not None:
        if any(charges[k] != cap_charges[k] for k in charges):
            raise ValueError("封端计划与回读形成的电荷账不一致")
        charges["uncapped"] = cap_charges["uncapped"]
    if operation == "remove-fragment" and recipe.get("cap", "none") != "none":
        bare = recipe["uncapped_charge"]
        if source_q is not None and removed_q is not None and source_q - removed_q != bare:
            raise ValueError("裸结构电荷与删除部分电荷不符")
        charges["uncapped"] = bare
        charges["child"] = bare + added_q
    result = verify_output(out, parent.symbols, removed, added, charges, expected_sites=expected)
    if integrity is not None:
        result["template_integrity"] = integrity
    if guest_ledger is not None:
        if guest_ledger["retained_charge"] is None:
            guest_ledger["retained_charge"] = added_q
        result["guest_fragment_ledger"] = guest_ledger
    if cap_charges is not None:
        result["cap_sites"] = report["cap_sites"]
    return result
