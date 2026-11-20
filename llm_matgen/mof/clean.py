"""按显式选择处理无序位点和游离客体，保留父结构编号。"""
from collections import defaultdict
from .graph import BondGraph


def inspect_clean(graph):
    model = graph.model
    alternatives = defaultdict(lambda: defaultdict(list))
    partial = []
    for i,(assembly,group,occ) in enumerate(zip(model.assemblies,model.groups,model.occupancy),1):
        if group not in {".","?","0"}:
            alternatives[assembly][group].append(i)
        if occ < 1:
            partial.append(dict(atom=i,label=model.labels[i-1],occupancy=occ))
    # 缺 H 只能标疑点。坐标不能区分水、OH、氧化物或未解析的 H。
    oxygen = [i+1 for i,s in enumerate(graph.symbols) if s=="O" and graph.neighbors[i]
              and all(graph.elements[j].is_metal for j,_ in graph.neighbors[i])]
    return dict(disorder_groups=dict(alternatives),partial_sites=partial,
                free_guests=[f for f in graph.fragments() if f["type"]=="free_guest"],
                possible_missing_h=oxygen,borderline=graph.borderline,
                hydrogen_note="只列出仅连金属的 O 作人工判读，未提供质子化状态。")


def clean(graph, choices):
    report = inspect_clean(graph)
    if not choices:
        return report, None
    model = graph.model
    removed, ordered = set(), set()
    selected = {}
    for choice in choices:
        if choice=="remove-guests":
            uncertain = {n for b in graph.borderline for n in b["edge"][:2]}
            for f in report["free_guests"]:
                if uncertain.intersection(f["atoms"]):
                    raise ValueError("临界键涉及候选客体，须先明确成键再删除")
                removed.update(f["atoms"])
        elif choice.startswith("group:"):
            parts = choice.split(":")[1:]
            groups = report["disorder_groups"]
            if len(parts)==1 and len(groups)==1:
                assembly, group = next(iter(groups)),parts[0]
            elif len(parts)==2:
                assembly, group = parts
            else:
                raise ValueError("请用 group:组装编号:构象 指定唯一无序组")
            if group not in groups.get(assembly,{}) or assembly in selected:
                raise ValueError("无序组不存在或选择重复")
            selected[assembly] = group
            for key,sites in groups[assembly].items():
                (ordered if key==group else removed).update(sites)
        elif choice=="keep-coordinated-H":
            for i,s in enumerate(graph.symbols):
                if s=="H" and any(graph.symbols[j] in {"O","N"} and
                    any(graph.elements[k].is_metal for k,_ in graph.neighbors[j])
                    for j,_ in graph.neighbors[i]):
                    ordered.add(i+1)
        elif choice.startswith(("keep-label:","drop-label:")):
            action,label = choice.split(":",1)
            matches = {s["atom"] for s in report["partial_sites"] if s["label"]==label}
            if not matches:
                raise ValueError("未找到部分占位标签 "+label)
            (ordered if action=="keep-label" else removed).update(matches)
        else:
            raise ValueError("未知清洗选择 "+choice)
    if removed & ordered:
        raise ValueError("同一位点同时被选中保留与删除")
    unresolved = {s["atom"] for s in report["partial_sites"]}-removed-ordered
    for assembly, alternatives in report["disorder_groups"].items():
        if assembly not in selected:
            unresolved.update(i for ids in alternatives.values() for i in ids if i not in removed)
    if unresolved:
        report["unresolved_atoms"] = sorted(unresolved)
        return report,None
    keep = [i for i in range(len(model.structure)) if i+1 not in removed]
    if not keep:
        raise ValueError("选择会删除全部原子")
    new_index = {old:i for i,old in enumerate(keep)}
    edges = {(new_index[i],new_index[j],offset) for i,j,offset in graph.edges if i in new_index and j in new_index}
    child = model.slice(keep,edges)
    report["removed_atoms"] = [dict(atom=i,label=model.labels[i-1]) for i in sorted(removed)]
    return report,(BondGraph(child),[i+1 for i in keep])
