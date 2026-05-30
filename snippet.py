from pathlib import Path
import re

mdpa_in  = Path("Slit_full.mdpa")
mdpa_out = Path("Slit_full_fixed.mdpa")

text = mdpa_in.read_text()

n_particles = 500  # change if you ever adjust nx*ny*nz

# Build a clean ParticlesPart block
subpart = []
subpart.append("Begin SubModelPart ParticlesPart")
subpart.append("  Begin SubModelPartNodes")
for i in range(1, n_particles+1):
    subpart.append(f"    {i}")
subpart.append("  End SubModelPartNodes")
subpart.append("")
subpart.append("  Begin SubModelPartElements")
for i in range(1, n_particles+1):
    subpart.append(f"    {i}")
subpart.append("  End SubModelPartElements")
subpart.append("End SubModelPart")

subpart_new = "\n".join(subpart)

# Replace the existing ParticlesPart block (with the # ... in it)
pattern = r"Begin SubModelPart ParticlesPart.*?End SubModelPart"
text_new = re.sub(pattern, subpart_new, text, flags=re.S)

mdpa_out.write_text(text_new)
print("Wrote", mdpa_out)