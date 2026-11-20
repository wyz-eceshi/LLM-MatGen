"""按晶面法线测层与端接，周期边界两侧同一层合并。"""
from collections import Counter
import numpy as np
from .check import continuous,height,fingerprint


def layers(structure,axis,tol=.14):
    period=height(structure.lattice.matrix,axis)
    values=structure.frac_coords[:,axis]%1
    # 从最大空隙之后开始，可以将零点两侧属于同一原子层的位点合并。
    z=continuous(values)*period
    groups=[]
    for i in np.argsort(z):
        if not groups or z[i]-z[groups[-1][0]]>tol:
            groups.append([])
        groups[-1].append(int(i))
    data=[]
    for ids in groups:
        center=float(z[ids].mean()%period)
        data.append(dict(position=center,composition=dict(Counter(structure[i].specie.symbol for i in ids)),
                         boundary=bool(min(center,period-center)<=tol),indices=[i+1 for i in ids]))
    return sorted(data,key=lambda row:row["position"])


def measure(model,axis,settings):
    s=model.structure
    layer=layers(s,axis,settings.get("layer_tol",.14))
    lz=height(s.lattice.matrix,2)
    z=continuous(s.frac_coords[:,2])*lz
    bottom=np.flatnonzero(z<=z.min()+settings.get("layer_tol",.14))
    top=np.flatnonzero(z>=z.max()-settings.get("layer_tol",.14))
    positions=[r["position"] for r in layer]
    gaps=np.diff(np.r_[positions,positions[0]+height(s.lattice.matrix,axis)])
    result=dict(atom_count=len(s),in_plane=[s.lattice.a,s.lattice.b,s.lattice.gamma],
                layers=layer,spacings=gaps.tolist(),boundary_layer=any(r["boundary"] for r in layer),
                thickness=float(np.ptp(z)),vacuum=float(lz-np.ptp(z)),
                terminations=[dict(Counter(s[i].specie.symbol for i in ids)) for ids in (bottom,top)],
                coordination=fingerprint(s,settings.get("cutoffs")))
    if settings.get("mof", "C" in [site.specie.symbol for site in s] and any(site.specie.is_metal for site in s)):
        from llm_matgen.mof.graph import BondGraph
        g=BondGraph(model)
        result["fragments"]=g.fragments()
        result["borderline"]=g.borderline
    return result
