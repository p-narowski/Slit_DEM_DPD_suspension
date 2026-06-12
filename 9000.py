import random, math
random.seed(42)
N = 9000
spacing = (1.0*1.0*0.5/N)**(1/3)
nx = math.ceil(1.0/spacing); ny = math.ceil(1.0/spacing); nz = math.ceil(0.5/spacing)
pts = []
for ix in range(nx):
    for iy in range(ny):
        for iz in range(nz):
            x = max(0.01, min(0.99, (ix+0.5+random.uniform(-0.3,0.3))/nx))
            y = max(0.01, min(0.99, (iy+0.5+random.uniform(-0.3,0.3))/ny))
            z = max(0.01, min(0.49, (iz+0.5+random.uniform(-0.3,0.3))/nz*0.5))
            pts.append((x,y,z))
random.shuffle(pts); pts = pts[:N]

with open("SlitDEM.mdpa", "w") as f:
    f.write("Begin ModelPartData\n//  VARIABLE_NAME value\nEnd ModelPartData\n\n")
    f.write("Begin Properties 0\nEnd Properties\n\n")
    f.write("Begin Nodes\n")
    for i,(x,y,z) in enumerate(pts,1): f.write(f"  {i:6d}  {x:.6f}  {y:.6f}  {z:.6f}\n")
    f.write("End Nodes\n\nBegin Elements DPDParticle\n")
    for i in range(1,N+1): f.write(f"  {i:6d}  0  {i}\n")
    f.write("End Elements\n\nBegin NodalData RADIUS // GUI group identifier: ParticlesPart\n")
    for i in range(1,N+1): f.write(f"  {i:6d}   0  0.02\n")
    f.write("End NodalData\n\nBegin SubModelPart DEMParts_ParticlesPart\n")
    f.write("  Begin SubModelPartNodes\n")
    for i in range(1,N+1): f.write(f"    {i}\n")
    f.write("  End SubModelPartNodes\n  Begin SubModelPartElements\n")
    for i in range(1,N+1): f.write(f"    {i}\n")
    f.write("  End SubModelPartElements\nEnd SubModelPart\n")
print("Done")