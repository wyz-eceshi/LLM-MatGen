"""金属节点定点枚举与对称轨道。"""
from collections import Counter,defaultdict
from dataclasses import replace
from math import factorial,prod
import numpy as np
from pymatgen.core import Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from llm_matgen.generators.doping import DopingGenerator,DopingParams
from .graph import BondGraph


def arrangements(counts, prefix=()):
    if not any(counts.values()):
        yield prefix
    for s in counts:
        if counts[s]:
            counts[s]-=1
            yield from arrangements(counts,prefix+(s,))
            counts[s]+=1


def enumerate_nodes(g,recipe,limit):
    metals={i for f in g.fragments() if f["type"]=="metal_node" for i in f["atoms"]}
    sites=recipe.get("sites",sorted(metals))
    if not sites or any(type(i)!=int or i not in metals for i in sites) or len(set(sites))!=len(sites):
        raise ValueError("只能选取互不重复的金属节点位点")
    if "fragments" in recipe:
        selected=set(recipe["fragments"])
        nodes={f["id"]:f["atoms"] for f in g.fragments() if f["type"]=="metal_node"}
        if not selected.issubset(nodes):
            raise ValueError("所选片段必须是金属节点")
        allowed={i for k in selected for i in nodes[k]}
        sites=[i for i in sites if i in allowed]
    if "coordination" in recipe:
        sites=[i for i in sites if g.coordination()[i]==recipe["coordination"]]
    species=recipe["species"]
    if len(sites)!=len(species) or not sites or not all(Element(s).is_metal for s in species):
        raise ValueError("所选节点与金属分配数不符")
    counts=Counter(species)
    total=factorial(len(sites))//prod(factorial(k) for k in counts.values())
    report=dict(total=total,sites=sites,max_structures=limit)
    if total>limit:
        return report,[]
    tol=float(recipe.get("symprec",.01))
    if tol<=0 or not np.isfinite(tol):
        raise ValueError("对称容差必须为正")
    coords=g.coords[np.array(sites)-1]
    perms=set()
    for op in SpacegroupAnalyzer(g.structure,symprec=tol).get_symmetry_operations():
        d=g.structure.lattice.get_all_distances(op.operate_multi(coords),coords)
        perm=tuple(map(int,d.argmin(axis=1)))
        if len(set(perm))==len(sites) and d.min(axis=1).max()<=tol:
            perms.add(perm)
    choices=list(arrangements(counts))
    classes={}
    for choice in choices:
        orbit=min(tuple(choice[k] for k in p) for p in perms)
        classes.setdefault(orbit,len(classes)+1)
    report.update(symprec=tol,inequivalent=len(classes),assignments=[
        dict(species=c,orbit=classes[min(tuple(c[k] for k in p) for p in perms)]) for c in choices])
    return report,choices


def generate(g,sites,assignment):
    grouped=defaultdict(list)
    for i,s in zip(sites,assignment,strict=True):
        if g.symbols[i-1]!=s:
            grouped[g.symbols[i-1],s].append(i)
    struct=g.structure.copy()
    for (original,dopant),subset in grouped.items():
        params=DopingParams(target_elements=[original],dopant_elements=[dopant],dopant_counts=[len(subset)],
                            target_site_indices=subset,allow_repeated_dopant=True,seed=0,max_structures=1)
        struct=DopingGenerator().generate(struct,params).generated[0].structure
    return BondGraph(replace(g.model,structure=struct,explicit=set(g.edges))),list(range(1,len(g.symbols)+1))
