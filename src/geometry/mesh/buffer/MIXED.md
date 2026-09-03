# Mixed-element buffer layer

Scoping document. No implementation on this branch.

## Motivation

`Mesh::buffer` grows an all-hexahedral layer off a hexahedral core and fits it
to a `Tessellation`. One hexahedron per boundary quadrilateral means the four
outer nodes of every buffer cell must stay a near-planar quadrilateral while
also landing on the target. Where the target has a crease or a corner crossing
the interior of a core face, no placement of four nodes satisfies both, and the
cell inverts. Everything downstream of that — the regularized fitting energy,
the epsilon schedule, the min-SJ ratchet, feature snapping — exists to trade
one failure mode against another, and the recorded results are mostly negative: the min-SJ ratchet, the periodic
Laplace shake, crease snapping and geometric pillowing were each measured and
rejected.

`Mesh::buffer_tets` removes the constraint entirely by making both the core and
the shell simplicial, but that gives up the hexahedral core, which is the point
of the marching/MCHex arc.

The mixed shell keeps the core untouched and spends the freedom only in the
transition band: the core-side face of every shell cell stays a quadrilateral,
and the outer side is triangulated, so the surface is fitted by simplices that
cannot be forced non-planar.

## What already exists

Verified against `main` (the `tet` branch is merged; `origin/tet` is an
ancestor of `main`).

| Piece | Location | State |
| --- | --- | --- |
| `Connectivity::Pyramidal`, `Connectivity::Wedge` | `connectivity/mod.rs` | present, with `TryFrom<Connectivity>` |
| local faces, Abaqus sides, edge adjacency, element counts | `connectivity/base/mod.rs`, `connectivity/primitive/mod.rs` | present for 5- and 6-node cells |
| `exterior_faces` on mixed blocks | `mesh/base/mod.rs` | generic over `local_faces`, already correct |
| `retain_elements` | `mesh/retain/mod.rs` | already handles `Pyramidal`/`Wedge` |
| `peel`, `prism`, `merge`, `manifold_boundary`, `project` | `buffer/mod.rs` | reusable verbatim |
| linear + quadratic pyramid and wedge elements | `fem/block/element/{linear,quadratic}/{pyramid,wedge}/` | present |
| cohesive wedge | `fem/block/element/cohesive/linear/wedge/` | present |
| Exodus / VTU / Abaqus / Medit read **and** write | `mesh/{read,write}/` | `pyramid5`/`wedge6`, `C3D5`/`C3D6`, `Pyramids`/`Prisms`, VTK 14/13 — both directions |
| crease and corner extraction, binned lookup | `tessellation/features/mod.rs` | `Features`, `FeatureIndex::{nearest_corner,nearest_crease}` |

So the analysis side and the I/O side are not blockers. The gaps are quality
metrics, the fitting energy, and the shell construction itself.

## Templates

Fixed constraint: the core-side face of a shell cell is the core hexahedron's
boundary quadrilateral, unmodified. Diagonalizing it would either break
conformity with the core hexahedron or force the core to be split, which is the
whole reason a pyramid is used at all.

Notation: `n0..n3` are the core boundary quadrilateral's nodes in face order,
`m0..m3` their peel duplicates, `m` a new node.

### Template B — feature-crossed face (recommended default)

Insert one new node `m` above the face, free to sit on the crease or corner.

```
  1 pyramid  base (n0,n1,n2,n3)          apex m
  4 pyramids base (ni,ni+1,mi+1,mi)      apex m      i = 0..3
```

5 pyramids, 1 new node beyond the peel duplicates. The decomposition is exact:
coning the hexahedral shell region from `m` over each of its faces not
containing `m` gives the base quadrilateral and the four side quadrilaterals,
and nothing else. The outer surface is the four triangles `(mi, mi+1, m)`.

Optional split of the four side pyramids into two tetrahedra each gives the
`1 pyramid + 8 tets` form.

**Template B in its 5-pyramid form has no diagonal-parity problem at all.**
Every internal side quadrilateral `(ni, ni+1, mi+1, mi)` stays a quadrilateral,
shared as a whole face between the side pyramids of two adjacent core faces. No
global diagonal choice is needed anywhere. This is the property that makes B
the right first slice rather than the exceptional case.

It also satisfies the one-element-thick requirement for the rational pyramid
shape functions: every one of the five cells has a face on the core boundary or
on the target surface, and none is stacked behind another.

### Template A — clean quadrilateral patch

No new node; the apex is one of the four layer nodes, chosen deterministically
(lowest global node number, matching the ordering trick in `prism`).

Coning the shell region from `m0` over the faces not containing `m0`:

```
  1 pyramid  base (n0,n1,n2,n3)          apex m0
  1 pyramid  base (n1,n2,m2,m1)          apex m0
  1 pyramid  base (n2,n3,m3,m2)          apex m0
```

3 pyramids, 0 new nodes; or `1 pyramid + 4 tets` if the two side pyramids are
split. **Correction to the earlier sketch:** the residual above the base
pyramid is two side pyramids, not a triangular prism, so it is 4 tetrahedra,
not 2. There is no `pyramid + wedge` variant, because a wedge would need a
triangulated core-side face.

Template A does impose diagonals. The two side quadrilaterals that contain
`m0` — `(n0,n1,m1,m0)` and `(n3,n0,m0,m3)` — are each covered by two triangles
meeting on the diagonal through `m0`, so the neighbouring core face's template
must agree on that diagonal. That is the same parity problem as the
Kuhn/Freudenthal split solved in `from_lattice_tets`, and it must additionally
be reconciled against Template B's neighbours, which want the same face left
undiagonalized.

### Selection policy

- Build `Features::of(target).index(radius)` once, `radius` from the local
  element size.
- A core boundary face is *feature-crossed* if `nearest_corner` or
  `nearest_crease` returns a hit within the face's own size of the face
  centroid, or if the four layer nodes' closest-point normals disagree by more
  than the crease threshold (`CREASE_COSINE`).
- Feature-crossed → Template B. Clean → Template A, once A exists.
- P0 uses Template B unconditionally. It is uniform, parity-free, and correct
  everywhere; A is purely an element-count optimization for the flat majority.

### Conformance across templates

If A is introduced, a side quadrilateral is shared by two core faces and must
be either a whole quad face (both sides B, or both sides A-with-that-quad-as-a-
base) or split on the same diagonal.

Recommended rule, in order:

1. If either neighbour is B, the shared quadrilateral stays a quadrilateral;
   the A side must then use the 3-pyramid form and pick its apex so that the
   shared face is a base, not a split pair. This is not always satisfiable for
   an arbitrary apex choice, so:
2. Simpler and always satisfiable: an A face adjacent to any B face is demoted
   to B. B-ness is then grown one face outward from the feature set. The cost
   is one extra node and two extra cells per demoted face.
3. Between two A faces, split the shared quadrilateral on the diagonal joining
   the lower-numbered of its two core nodes to the duplicate of the higher —
   exactly the rule `prism` already uses, which is well defined from the node
   numbers alone and therefore automatically consistent.

Rule 2 plus rule 3 is the plan. Rule 1 is recorded only to say it was
considered and rejected as fragile.

At the core boundary itself there is nothing to reconcile: every template's
core-side face is the untouched quadrilateral.

## Quality metrics

`quality/metrics/mod.rs` returns `Scalar::NAN` for `Pyramidal` and `Wedge` in
all five `Verdict` methods, and `Kind::of` returns `None` for them, so
`Incidence::of` — used by `untangle` and `smart_laplace` — *panics* on any mesh
containing them (`expect("unsupported element type")`). That panic is a hard
blocker, not just a missing number.

Concrete work, in `quality/metrics/{pyramid,wedge}/mod.rs` mirroring
`tetrahedron/mod.rs`:

1. `wedge::CORNERS: [[usize; 3]; 6]` — each of the six wedge corners meets
   exactly three edges, so the existing `min_jacobian` / `min_scaled_jacobian`
   machinery applies unchanged. Normalizer: the ideal wedge's triangle edges
   meet at 60°, so the scale constant is `2.0 / SQRT_3`.
2. `wedge::EDGES: [[usize; 2]; 9]`, `wedge::FACES` (2 triangles, 3 quads) for
   `maximum_edge_ratio` and `maximum_skew`; skew takes the triangle skew on the
   two triangular faces and the hexahedron-style axis-pair cosine on the three
   quadrilateral ones.
3. `wedge::volume` — sum of three tetrahedra, or the exact trilinear form.
4. `pyramid::CORNERS` — the four base corners each meet three edges and are
   handled by the existing table shape. The apex meets four edges and is not.
   Handle it by fanning the apex into four triples `(4, [i, i+1, ...])`, which
   requires the corner table to carry its origin node explicitly rather than
   being indexed by position (see below). Scale constant for the base corners:
   the ideal pyramid's apex edge is at 45° to the base, giving `SQRT_2`;
   the apex triples take the tetrahedral `SQRT_2` as well. Check both against
   Verdict/CUBIT before locking the numbers in — this is the one place the
   design is guessing.
5. `pyramid::EDGES: [[usize; 2]; 8]`, `FACES` (4 triangles + 1 quad), `volume`
   (two tetrahedra, or base-area times height / 3 for the planar-base case).
6. Extend `Kind` with `Pyramid` and `Wedge`, wire both into `Kind::of`,
   `minimum_jacobian`, `minimum_scaled_jacobian`, and all five `Verdict`
   methods, and remove `Pyramidal`/`Wedge` from the `NAN` arms.
7. Unit tests per element: reference cell gives exactly 1.0 scaled Jacobian,
   a known-inverted cell gives a negative value, degenerate gives 0.

Only the tables and constants are new; `corners`, `corner_measure`,
`min_jacobian`, `min_scaled_jacobian`, `maximum_edge_ratio`, `triangle_skew`
and `tet_volume` are all reused.

The `[[usize; 3]; N]` corner-table shape is what forces the pyramid apex to be
special-cased. Change it to `&'static [(usize, [usize; 3])]` — corner origin
plus its three neighbours — decoupling the corner count from the node count.
Existing tables become mechanical rewrites; the pyramid gets 8 entries for 5
nodes.

## Fitting energy

`fit_elements::<N, F>` is monomorphic in the element arity `N` and the boundary
face arity `F`, collects every block into a single `Vec<[usize; N]>`, and
errors on any other block kind. A mixed mesh breaks all three assumptions.

Generalization:

- Reuse the same corner-table change: `energy`, `scatter` and `determinant`
  iterate `(corner, adjacent)` pairs instead of enumerating a per-node array,
  so they stop being generic in `N` and take `element: &[usize]` with a
  `&'static [(usize, [usize; 3])]` table.
- `Sweep` holds `elements: Vec<(&'static CornerTable, &[usize])>` — or an
  index into a per-kind table plus a flat node slice — rather than
  `&[[usize; N]]`. Local gradient accumulation uses a fixed `[Slope; 8]` with a
  live length, since no supported cell has more than eight nodes.
- `sizes` averages nodal lengths over `element.len()` instead of `N`.
- The `F` parameter goes away with it. The shell's exterior faces under
  Template B are all triangles `(mi, mi+1, m)`; under Template A they are the
  triangles produced by the top-quad diagonal. So the fitted boundary is
  simplicial in both cases and `F = 3` would in fact suffice, but a mixed core
  boundary is not worth assuming — carry `&[usize]` faces and divide by
  `face.len()` for the centroid.
- New entry point `Mesh::fit_mixed(&mut self, nodes, target)`, dispatching per
  block to the right corner table. `fit` and `fit_tets` become thin wrappers,
  or are deleted in favour of it once the numbers are shown to be unchanged.

Everything else in `fit/mod.rs` — the L-BFGS loop, the Armijo backtracking, the
epsilon schedule, the `weight` floor, the target oracle, the stagnation window,
the threading — is untouched. That matters: the tuned constants
(`SWEEPS = 50`, `BALANCE`, `STAGNATION`) carry a lot of accumulated evidence
and must not be re-tuned as part of this work.

Note on the apex corner: each of the four base corners of a pyramid has the
apex among its three neighbours, so the base triples alone already constrain
the apex position and forbid it from crossing the base plane. Including the
four apex triples is still preferred because it penalizes an apex that slides
laterally off the base, which is precisely the motion a crease pulls it into.

## Architecture

New submodule `buffer/mixed/`, sibling to `buffer/fit/` and `buffer/restrict/`:

```
buffer/
  mod.rs        Fitting, Peeled, peel, prism, merge, manifold_boundary, project
  mixed/
    mod.rs      Mesh::buffer_mixed, template selection, shell assembly
    template.rs Template A and B cell emission, apex choice, parity rule
    test.rs
  fit/
  restrict/
```

`buffer/mod.rs` keeps everything shared and gains nothing but a `mod mixed;`.
`Peeled`, `peel`, `merge`, `project` and `manifold_boundary` are used verbatim;
`prism` is used only if the tetrahedral split forms are enabled.

Driver, mirroring `buffer` and `buffer_tets`:

```rust
pub fn buffer_mixed(mut self, target: &Tessellation, fitting: Fitting)
    -> Result<Self, &'static str>
```

1. `self.restrict()?` — unchanged, still requires the single hexahedral core
   block, and still runs before any peeling.
2. `let boundary = self.exterior_faces();`
3. `self.peel(&boundary, 4, "non-quadrilateral boundary face")?`
4. classify each boundary face against the `FeatureIndex`, grow B outward one
   face, place the Template B apexes at the closest point on the target from
   the face-centroid duplicate (so `m` starts on the surface, not above it).
5. emit cells per template, `merge` them into a `Pyramidal` block (plus a
   `Tetrahedral` block only if a split form is enabled).
6. `fit_mixed` over layer nodes plus core nodes; on `Fitting::Snap`, `project`
   the layer then `fit_mixed` over the core only.

Under P0 the result has exactly two blocks: the hexahedral core and the
pyramidal shell.

## FEM integration

`Blocks<B1, B2>` is right-nested and every solver trait
(`ElasticElements`, `HyperelasticElements`, `ViscoelasticElements`,
`ElasticViscoplasticElements`, `HyperelasticViscoplasticElements`,
`HyperviscoelasticElements`, `ElasticHyperviscousElements`,
`ThermalConductionElements`, `Elements`) has a blanket recursive impl over
`B1, B2`, so arbitrary nesting depth already works in the solver. Confirmed by
inspection of `src/domain/fem/{mod.rs,solid/*,thermal/*}`.

What does **not** exist is the constructor. `src/domain/fem/from/mod.rs` has
`TryFrom<(Mesh<3>, C)>` for one block and `TryFrom<(Mesh<3>, (C1, C2))>` for
two, and nothing beyond. So:

- P0's two-block output (hex + pyramid) is constructible today with
  `Model::<Blocks<Block<_, Hexahedron, ...>, Block<_, Pyramid, ...>>, 3>::try_from((mesh, (model, model)))`.
- Any tetrahedral split form yields three blocks and needs a
  `TryFrom<(Mesh<3>, (C1, C2, C3))>` for `Blocks<B1, Blocks<B2, B3>>`, and four
  if wedges are added. Write these as a macro over the arity; the two-block
  impl is already a mechanical unrolling.
- Block ordering must be deterministic — `merge` appends by kind, so the core
  hexahedral block stays first and the shell blocks follow in emission order.
  The test suite should pin that ordering, since the tuple position selects the
  element type.

`LinearElement<8, 5>` (`Pyramid`, `G = 8`) and `LinearElement<6, 6>` (`Wedge`)
both implement `FiniteElement` and `From<ElementNodalReferenceCoordinates<N>>`,
so no element work is needed.

The rational pyramid shape functions are integrated at 8 Gauss points with the
`bottom(xi_3)` collapse factor; accuracy is lower than for a hexahedron, which
is the standing reason to keep the pyramid band one element thick. Template B
satisfies that by construction. If the band ever needs to be load-bearing
rather than transitional, the split forms plus the quadratic pyramid are the
escape hatch, not a thicker linear band.

## I/O

Nothing to build. Verified on `main`:

| Format | Read | Write |
| --- | --- | --- |
| Exodus | `pyramid5`, `wedge6` (`read/exodus/mod.rs:161-162`) | via `exodus_element_type` (`primitive/mod.rs:74-75`) |
| VTU | cell types 14, 13 (`read/vtk/unstructured/mod.rs:165-166`) | 13, 14 (`write/vtk/unstructured/mod.rs:30-31`) |
| Abaqus | `C3D5`, `C3D6` (`read/abaqus/mod.rs:142,145`) | `C3D5`, `C3D6` (`write/abaqus/mod.rs:26-27`) |
| Medit | `Pyramids`, `Prisms`/`Pentahedra` | `Pyramids`, `Prisms` |

Required work is a round-trip test only: write a two-block hex+pyramid mesh to
each of the four formats, read it back, assert block kinds, ordering, node
count and connectivity are identical. Node ordering conventions across formats
for pyramid and wedge are the thing that will actually break, so the test must
compare connectivity element-wise, not just counts. Note that the `io` coverage
gate is 100/100/99, so any new I/O branch needs its own test.

## Reuse from the tetrahedral arc, verbatim

- `peel` — arity-parameterized already, called with 4.
- `merge` — kind-agnostic, appends to the last block of a kind or opens a new
  one. Called once per emitted kind.
- `project` — moves layer nodes onto the target's closest point. Used for
  `Fitting::Snap` and for placing the Template B apexes.
- `retain_elements` — already handles `Pyramidal` and `Wedge`.
- `prism` — only for the split forms; its "diagonal from the lower-numbered
  base node to the duplicate of the higher" rule is reused directly as the
  Template A parity rule, and the reasoning in its doc comment (a total order
  on node numbers cannot wind a diagonal around a prism) transfers unchanged.
- `manifold_boundary` — **not** reused. It is tetrahedron-specific (`TET_FACES`,
  `[usize; 4]`) and exists because `trim` can pinch a simplicial background. A
  hexahedral core rarely pinches and `buffer` already skips it. If a
  generalization is ever wanted it should be written against
  `Connectivity::local_faces` rather than a hard-coded table.
- `Fitting`, `Peeled` — unchanged.

## Test strategy

Unit, in `buffer/mixed/test.rs`:

- Template B on one core face in isolation: 5 pyramids, 1 new node, all volumes
  positive, the union's volume equals the shell region's volume, every internal
  side quadrilateral appears exactly twice across the whole mesh.
- Template A on one core face: 3 pyramids (or 1 + 4 tets), 0 new nodes,
  same volume and face-pairing assertions.
- Mixed A/B patch: assert `exterior_faces` is manifold and that every internal
  face is shared by exactly two cells — this is the assertion that actually
  catches a parity bug, and it must be written to fail against a deliberately
  flipped diagonal before it is believed.
- Apex placement: with a target carrying a crease through the face, the
  Template B apex lands on the crease within tolerance.

End-to-end, reusing the existing fixtures in `buffer/test.rs`:

- `core()` (single hexahedron) buffered to `tessellation()` (the unit cube):
  the eight cube corners must be captured, as `buffer_captures_corners`
  already asserts for the hexahedral shell, and the minimum scaled Jacobian
  over the *whole* mixed mesh must exceed the same 0.2 threshold. The cube is
  the sharpest available crease-and-corner target and is the discriminating
  case for this feature.
- Sphere target via `sphere(12, 16, 1.0)` with a trimmed hexahedral background:
  smooth surface, so every face should classify clean; assert Template B count
  is zero once A exists, and non-zero for the cube.
- `Fitting::Snap`: layer node deviation from the target below `1e-12`, as
  `buffer_snaps_to_surface` asserts today.
- Comparison against `buffer` on the same input: the mixed shell's minimum
  scaled Jacobian must be strictly better on the cube. If it is not, the
  feature has not earned its keep and should not merge.

The bone model is not in the repository; it is the out-of-repo manual check, and a
weak discriminator: `buffer` timings are non-deterministic because of the
stagnation window, so use the bone for min-SJ and visual inspection only, never
for timing regressions.

Gates: `cargo test --features geometry`, never `--features all` or `fem`. Coverage for `geometry` is 50/50/50, so new
modules need reasonable but not exhaustive coverage; the I/O round-trip test
lands under the `io` feature where the gate is 100/100/99.

## Staged plan

**P0 — quality metrics.** Pyramid and wedge scaled Jacobian, Jacobian, edge
ratio, skew, volume; the corner-table reshape to `&[(usize, [usize; 3])]`;
`Kind::{Pyramid, Wedge}` wired everywhere, removing the `Incidence` panic. No
buffer changes. Independently useful and independently reviewable — this is the
smallest shippable slice and it should be its own PR.

**P1 — fitting energy generalization.** `fit_mixed` with per-kind corner
tables, `fit`/`fit_tets` rewritten as wrappers over it. Assert byte-identical
results on the existing `buffer` and `buffer_tets` tests before and after, so
the tuned constants are provably untouched.

**P2 — Template B shell, unconditional.** `Mesh::buffer_mixed` emitting five
pyramids per boundary face. No feature classification, no Template A, no
parity machinery. Two output blocks. This is the first slice that produces a
mesh, and it is already the interesting one, because B is exactly the template
that handles creases.

**P3 — Template A and selection.** `FeatureIndex`-driven classification, the
B-grown-one-face-outward demotion rule, the `prism` parity rule between A
faces. Element count drops by roughly 40% on smooth targets.

**P4 — FEM constructors.** `TryFrom<(Mesh<3>, (C1, C2, C3))>` and the
four-tuple, as a macro. Only needed once a split form or wedges are emitted;
P2's two-block output needs nothing.

**P5 — split forms and wedges.** `1 pyramid + 8 tets` / `1 pyramid + 4 tets`
behind an option, for consumers that reject pyramids. Wedges enter here if a
graded two-sub-layer variant is ever wanted; they are not needed for P2 or P3.

## Risks and open questions

- **Pyramid scaled-Jacobian normalizer.** The base-corner constant `SQRT_2` and
  the apex-triple treatment are derived here, not looked up. If they disagree
  with Verdict/CUBIT the absolute numbers in every quality gate shift. Settle
  this against a reference implementation in P0, before any threshold is tuned
  against it.
- **Element count.** Template B is 5 cells and 1 node per boundary face against
  the hexahedral shell's 1 and 0. On a bone-scale model that is a large
  absolute increase in the shell. P3 recovers most of it on smooth regions but
  not near features, which is where the faces are.
- **Does the fit actually behave better?** The premise is that a simplicial
  outer surface cannot be forced non-planar, so the regularized energy has a
  feasible minimum where the hexahedral shell has none. That is an argument,
  not a measurement. The cube min-SJ comparison in P2 is the falsification
  test, and a negative result should be recorded and the branch abandoned
  rather than patched, given how many negative results this arc already has.
- **`restrict` clearance.** The clearance pre-pass is defined on hexahedral
  boundary quadrilaterals and runs before peeling, so it is unaffected. But its
  premise — that a node with no feasible direction should be pruned — is
  weaker when the shell has more freedom. It may prune faces the mixed shell
  could have handled. Worth measuring in P2, not worth changing blind.
- **Template A apex choice and parity interaction.** Rule 2 (demote A next to
  B) is chosen for being always satisfiable, not for being tight. If the
  demotion cascades — B spreading across a large fraction of the boundary on a
  feature-rich target — the element-count win from P3 evaporates. Measure the
  demotion fraction before investing in rule 1.
- **`manifold_boundary` for hexahedral cores.** Assumed unnecessary, matching
  `buffer`'s existing behaviour. If a trimmed marching/MCHex core turns out to
  pinch, this becomes a P2 blocker and needs a `local_faces`-based rewrite.
- **Quadratic pyramid.** `G = 27, N = 13`. Present but untouched by this plan;
  noted only so that a future decision to make the band load-bearing has a
  path that does not involve thickening it.
