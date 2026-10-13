import os
import pyvista as pv

from svv.domain.domain import Domain
from svv.tree.tree import Tree


# --------------------------------------------------
# Domain
# --------------------------------------------------

cube = Domain(pv.Cube())

cube.create()
cube.solve()
cube.build()


# --------------------------------------------------
# Tree
# --------------------------------------------------

tree = Tree(preallocation_step=100000)

tree.set_domain(cube)
tree.set_root([0.0, 0.0, 0.0])


# --------------------------------------------------
# Grow in chunks
# --------------------------------------------------

os.makedirs("trees", exist_ok=True)

total = 100000
chunk = 5000

for start in range(0, total, chunk):

    tree.n_add(chunk)

    current = start + chunk

    filename = f"trees/network_{current}.tree"

    tree.save(filename)

    print(
        f"Checkpoint saved: {filename} | "
        f"segments={tree.segment_count}, "
        f"terminals={tree.n_terminals}"
    )


print(f"Finished generating {total:,} vessels!")