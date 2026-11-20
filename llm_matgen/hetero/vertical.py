"""二维整数超胞匹配与面面堆垛；主应变取面内变形的奇异值减一。"""
from itertools import product
import numpy as np
from pymatgen.core import Lattice,Structure


def planar(s):
    a,b,c=s.lattice.matrix
    x=a/np.linalg.norm(a)
    n=np.cross(a,b);n/=np.linalg.norm(n)
    if abs(c@n)/np.linalg.norm(c)<.999999:
        raise ValueError("板片真空矢量须垂直于面内晶格")
    return np.array([[np.linalg.norm(a),0.],[b@x,b@np.cross(n,x)]])


def bases(cell,max_det):
    unique={}
    for det in range(1,max_det+1):
        for a in range(1,det+1):
            if det%a:
                continue
            b=det//a
            for k in range(b):
                m=np.array([[a,k],[0,b]])
                for _ in range(100):
                    v=m@cell
                    if v[1]@v[1]+1e-10<v[0]@v[0]:
                        m=m[[1,0]]
                        continue
                    mult=round((v[0]@v[1])/(v[0]@v[0]))
                    if not mult:
                        break
                    m[1]-=mult*m[0]
                if np.linalg.det(m)<0:
                    m[1]*=-1
                if (m@cell)[0]@(m@cell)[1]<0:
                    m[1]*=-1
                v=m@cell
                key=tuple((v@v.T).round(7).flat)
                if key not in unique or tuple(m.flat)<tuple(unique[key].flat):
                    unique[key]=m
    return list(unique.values())


def principal(cell1,cell2):
    return np.sort(np.linalg.svd(np.linalg.solve(cell2,cell1),compute_uv=False)-1)


def matches(a,b,recipe):
    ca,cb=planar(a),planar(b)
    limit=recipe.get("max_atoms",500)
    def row(m,n):
        strain=principal(m@ca,n@cb)
        return dict(substrate=m.tolist(),film=n.tolist(),strain=strain.tolist(),maximum=float(max(abs(strain))),
                    atoms=round(abs(np.linalg.det(m)))*len(a)+round(abs(np.linalg.det(n)))*len(b))
    rows=[]
    if "matrices" in recipe:
        m,n=[np.array(x) for x in recipe["matrices"]]
        if any(x.shape!=(2,2) or not np.issubdtype(x.dtype,np.integer) or np.linalg.det(x)<=0 for x in (m,n)):
            raise ValueError("二维超胞矩阵必须是正行列式整数矩阵")
        rows.append(row(m,n))
    if recipe.get("search",not rows):
        ma=bases(ca,(limit-len(b))//len(a))
        mb=bases(cb,(limit-len(a))//len(b))
        for m,n in product(ma,mb):
            if round(abs(np.linalg.det(m)))*len(a)+round(abs(np.linalg.det(n)))*len(b)>limit:
                continue
            for perm in (np.eye(2,dtype=int),np.array([[0,1],[1,0]])):
                for signs in product((1,-1),repeat=2):
                    q=np.diag(signs)@perm@n
                    if np.linalg.det(q)*np.linalg.det(m)>0:
                        rows.append(row(m,q))
    return [r for r in rows if r["atoms"]<=limit]


def expand(s,matrix,block):
    m=np.eye(3,dtype=int)
    m[:2,:2]=matrix
    cell=s.copy()
    cell.add_site_property("source_site",list(range(1,len(s)+1)))
    cell.make_supercell(m,to_unit_cell=True)
    mapping=[]
    for site in cell:
        i=site.properties["source_site"]
        t=site.frac_coords@m-s[i-1].frac_coords
        if not np.allclose(t,np.rint(t),atol=1e-7):
            raise ValueError("超胞映射不完整")
        mapping.append(dict(parent=block,site=i,image=np.rint(t).astype(int).tolist()))
    return cell,mapping


def build(left,right,recipe):
    a,b=left.structure,right.structure
    rows=matches(a,b,recipe)
    threshold=recipe.get("accept_strain",.05)
    if not np.isfinite(threshold) or threshold<0:
        raise ValueError("主应变门槛须为非负有限数")
    allowed=[r for r in rows if r["maximum"]<=threshold]
    ordered=sorted(rows,key=lambda r:r["maximum"])
    report=dict(limit=threshold,examined=len(rows),closest=ordered[:5])
    if not allowed:
        report["infeasible"]=ordered
        return report,None
    # 显式登记的匹配通过门槛时优先，其他情形使用最小主应变。
    match=allowed[0] if "matrices" in recipe and allowed[0]["substrate"]==recipe["matrices"][0] and allowed[0]["film"]==recipe["matrices"][1] else min(allowed,key=lambda r:(r["maximum"],r["atoms"]))
    base,one=expand(a,match["substrate"],1)
    film,two=expand(b,match["film"],2)
    cell2=np.asarray(match["substrate"])@planar(a)
    gap=recipe.get("gap",3.2);vac=recipe.get("vacuum",20.)
    if gap<=0 or vac<=0:
        raise ValueError("层间距和真空必须为正")
    heights=[]
    for s in (base,film):
        z=s.frac_coords[:,2]*s.lattice.c
        # make_supercell 可能把 -0 舍入成 1，端点应归回零。
        z[np.isclose(z,s.lattice.c,atol=1e-8,rtol=0)]=0
        heights.append(z-z.min())
    za,zb=heights
    zb+=za.max()+gap
    cell=np.zeros((3,3));cell[:2,:2]=cell2;cell[2,2]=zb.max()+vac
    xyz=np.vstack([base.frac_coords,film.frac_coords])
    registry=recipe.get("registry",[0,0])
    if "registry_site" in recipe:
        if "registry" in recipe or "registry_grid" in recipe:
            raise ValueError("配位位置、平移与扫描互斥")
        site=recipe["registry_site"]
        ids=site["substrate_sites"]
        anchor=site["film_site"]
        if len(ids) not in (1,2,3) or len(set(ids))!=len(ids) or not all(type(i) is int and 1<=i<=len(base) for i in ids) or not 1<=anchor<=len(film):
            raise ValueError("配准位点编号无效")
        face=base.frac_coords[np.array(ids)-1]
        if np.ptp(face[:,2]*base.lattice.c)>.15 or np.any(za.max()-za[np.array(ids)-1]>.15):
            raise ValueError("指定的衬底原子必须位于同一顶面")
        images=[base.lattice.get_distance_and_image(face[0],f)[1] for f in face]
        center=np.mean(face+np.array(images),axis=0)[:2]
        registry=((center-film[anchor-1].frac_coords[:2]+.5)%1-.5).tolist()
        report["registry_site"]=dict(kind=("top","bridge","hollow")[len(ids)-1],**site)
    if "registry_grid" in recipe:
        if "registry" in recipe:
            raise ValueError("指定配准与扫描网格互斥")
        from .scan import vertical_registry
        registry,report["registry_scan"]=vertical_registry(base,film,cell2,za,zb,recipe["registry_grid"])
    xyz[len(base):,:2]+=registry
    xyz[:,2]=(np.r_[za,zb]+vac/2)/cell[2,2]
    constrained=base.site_properties.get("selective_dynamics",[[True]*3 for _ in base])
    flags=np.vstack([constrained,[[True]*3 for _ in film]])
    struct=Structure(cell,[s.species for s in list(base)+list(film)],xyz,to_unit_cell=True,
                     site_properties={"selective_dynamics":flags.tolist()})
    for i,entry in enumerate(one+two):
        entry["wrap"]=np.rint(xyz[i]-struct[i].frac_coords).astype(int).tolist()
    report.update(match=match,registry=registry,fixed=int(np.sum(~np.any(flags,axis=1))),
                  gap=gap,vacuum=vac)
    return report,(struct,one+two)
