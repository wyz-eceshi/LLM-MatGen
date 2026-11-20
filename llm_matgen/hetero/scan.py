"""用历史板片扫描的排斥、接触与距离评分选择面内平移。"""
from itertools import product
import numpy as np
from ase.data import vdw_radii, covalent_radii


def vertical_registry(base,film,cell,za,zb,grid):
    if type(grid) is not int or grid < 1:
        raise ValueError("配准网格须为正整数")
    def radii(s):
        numbers=np.array([site.specie.Z for site in s])
        cov=covalent_radii[numbers]
        vd=vdw_radii[numbers]
        return np.where(np.isfinite(vd),vd,1.5*cov),cov
    av,ac=radii(base)
    bv,bc=radii(film)
    vd=av[:,None]+bv
    cov=ac[:,None]+bc
    xy=film.frac_coords[None,:,:2]-base.frac_coords[:,None,:2]
    z=zb[None,:]-za[:,None]
    best=None
    for i,j in product(range(grid),repeat=2):
        shift=np.array([i,j])/grid
        diff=xy+shift
        cart=(diff-np.rint(diff))@cell
        ds=np.sqrt((cart*cart).sum(axis=2)+z*z)
        collisions=int(np.sum(ds<.78*cov))
        score=float(1e6*collisions+1000*np.sum(np.maximum(.85*vd-ds,0)**2)
                    -np.sum((ds>=.85*vd)&(ds<=1.15*vd))- .01*np.exp(-((ds-vd)/.45)**2).sum())
        if best is None or score<best[0]:
            best=(score,shift.tolist(),collisions)
    return best[1],dict(grid=grid,scanned=grid**2,score=best[0],collisions=best[2])
