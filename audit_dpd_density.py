import json, math, re
from pathlib import Path
p=json.loads(Path("ProjectParametersDEM.json").read_text())
m=json.loads(Path(p["solver_settings"]["material_import_settings"]["materials_filename"]).read_text())
V=math.prod(p["BoundingBoxMax"+a]-p["BoundingBoxMin"+a] for a in "XYZ")
path=Path(p["problem_name"]+"DEM.mdpa")
text=path.read_text()
N=0
for kind,body in re.findall(r"Begin Elements ([^\n]+)\n(.*?)End Elements",text,re.S):
    if kind.strip()=="DPDParticle":
        N+=sum(bool(line.split("//")[0].strip()) for line in body.splitlines())
rc=next(r["Variables"]["DPD_CUTOFF_RADIUS"] for r in m["material_relations"] if r["material_ids_list"]==[1,1])
if N==0: raise RuntimeError("No DPDParticle elements found; inspect element types.")
n=N/V
out={"input_file":str(path),"N_DPD_initial":N,"V_box":V,"rc":rc,"n":n,"V_per_particle":1/n,"r_cell":(3/(4*math.pi*n))**(1/3),"spacing":n**(-1/3),"eta":n*4*math.pi/3*rc**3,"n_rc3":n*rc**3,"N_for_eta4":math.ceil(4*V/(4*math.pi/3*rc**3))}
print(json.dumps(out,indent=2))
Path("dpd_density_audit.json").write_text(json.dumps(out,indent=2))