"""用一致的位点顺序写出结构与父子映射。"""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from pymatgen.io.vasp.inputs import Poscar
from llm_matgen.viewer import write_viewer
from .diff import compare


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2)+"\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bonded_cif(graph):
    structure = graph.structure
    lines = ["data_MatGen", "_symmetry_space_group_name_H-M 'P 1'",
             "_symmetry_Int_Tables_number 1","loop_","_symmetry_equiv_pos_as_xyz","'x,y,z'"]
    for key,val in zip(("length_a","length_b","length_c","angle_alpha","angle_beta","angle_gamma"),
                       (*structure.lattice.abc,*structure.lattice.angles),strict=True):
        lines.append(f"_cell_{key} {val:.12f}")
    lines.extend(["loop_","_atom_site_label","_atom_site_type_symbol","_atom_site_fract_x",
                  "_atom_site_fract_y","_atom_site_fract_z","_atom_site_occupancy"])
    labels = [f"{s}{i}" for i,s in enumerate(graph.symbols,1)]
    for label,s,site,occ in zip(labels,graph.symbols,structure,graph.model.occupancy,strict=True):
        coords = " ".join(f"{x:.12f}" for x in site.frac_coords)
        lines.append(f"{label} {s} {coords} {occ:.9g}")
    lines.append(f"_matgen_explicit_bond_count {len(graph.edges)}")
    if graph.edges:
        lines.extend(["loop_","_geom_bond_atom_site_label_1","_geom_bond_atom_site_label_2",
                      "_geom_bond_distance","_geom_bond_site_symmetry_2"])
    for i,j,t in sorted(graph.edges):
        if any(v < -5 or v > 4 for v in t):
            raise ValueError("跨胞偏移超出 CIF 对称编码范围")
        code = "1_"+ "".join(str(x+5) for x in t) if any(t) else "."
        lines.append(f"{labels[i]} {labels[j]} {graph.distance(i,j,t):.9f} {code}")
    return "\n".join(lines)+"\n"


def publish(destination, source, operation, parent, product, mapping, params, report, assumptions,
            *, template=None):
    dest = Path(destination).resolve()
    if dest.exists():
        raise ValueError("输出目录已存在")
    if any(occ != 1 for occ in product.model.occupancy):
        raise ValueError("POSCAR 无法表达部分占位，请先使用 clean 明确选择")
    dest.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=dest.parent) as temp:
        out = Path(temp)/"product"
        out.mkdir()
        (out/"structure.cif").write_text(bonded_cif(product),encoding="utf-8")
        (out/"POSCAR").write_text(Poscar(product.structure).get_str(),encoding="utf-8")
        accounting = {}
        if operation in {"remove-fragment", "replace-fragment", "scan-placement", "node-substitute"}:
            from .ledger import mof_output
            accounting = mof_output(out, source, operation, parent, mapping, params, report, template)
        write_viewer([(operation,product.structure)],out/"preview.html")
        frame=report.get("comparison_frame")
        if frame is not None:
            from .subnet import subnet_diff
            diff=subnet_diff(parent,product,mapping,frame)
        else:
            diff=compare(parent,product,mapping)
        info = dict(operation=operation,params=params,parent_sha256=sha256(source),
                    product_sha256=sha256(out/"structure.cif"),poscar_sha256=sha256(out/"POSCAR"),
                    atom_map=mapping,atom_map_basis="子位点 → 读取并按需展开后的父位点，均从 1 开始",
                    parent_sites=[dict(label=label,frac_coords=site.frac_coords.tolist()) for label,site
                                  in zip(parent.model.labels,parent.structure,strict=True)],
                    assumptions=assumptions,graph_diff=diff,
                    status="unoptimized_candidate", **accounting)
        if frame is not None:
            info["comparison_frame"]=frame
        if operation=="clean":
            info["removed_atoms"] = report["removed_atoms"]
        recipe = params.get("recipe", {})
        donor = operation=="scan-placement" and recipe.get("site",{}).get("kind")=="donor"
        mixed_caps = operation=="remove-fragment" and "cap_plan" in recipe
        if donor:
            info["placement"] = report
        if mixed_caps:
            info["cap_geometry"] = report["cap_geometry"]
        write_json(out/"sidecar.json",info)
        if not (donor or mixed_caps):
            write_json(out/"report.json",report)
        out.rename(dest)
    return info
