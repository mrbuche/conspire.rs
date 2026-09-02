# `cad` branch review log

Vibe-coded geometry: ~15k lines under `src/geometry/{cad,csg,solid}` + edits to
`geometry/mesh/buffer`. Reviewed in module-scoped passes against a standing
harness; the physical PR split is deferred to a stabilization freeze.

## Harness

| Piece | Location | Runs | Checks |
|---|---|---|---|
| Oracle cross-check | `cad/brep/oracle/harness.rs` | always (`cargo test -F geometry`) | `BrepOracle` vs closed-form `csg::Primitive` oracle: sign everywhere outside a surface band, magnitude to 3e-3·diag |
| SDF invariants | `cad/brep/oracle/harness.rs` | always | on closed solids, where the field is smooth (2nd-difference kink filter): `\|∇sdf\|≈1`, `sdf(project foot)≈0`, projection normal outward |
| STEP corpus parse sweep | `cad/read/step/brep/test.rs::corpus_parse_snapshot` | `#[ignore]`, `STEP_CORPUS_DIR` | every file parses without panic; per-file `{solids, faces, surface histogram, primitive count, oracle ok/n, assemble}` diffed vs `snapshots/corpus_parse.txt` |
| Golden mesh stats | `cad/read/step/brep/test.rs::corpus_mesh_snapshot` | `#[ignore]`, `STEP_CORPUS_DIR` | octree→dual→trim (no fit) at a fixed sizing; per-file `{hexes, bbox spans, boundary-quad count, non-manifold flag, worst SJ}` or the pipeline error, diffed vs `snapshots/corpus_mesh.txt` |

External corpora (not vendored): `~/Downloads/steptools/` (AS1 sample + `boxy_*`
GD&T), `~/Downloads/NIST-PMI-STEP-Files/` (NIST CTC/FTC/STC, US-gov public
domain). Point `STEP_CORPUS_DIR` at a dir holding both (symlinks fine); walked
recursively. `UPDATE_SNAPSHOT=1` rewrites a snapshot after a reviewed change.
The mesh sweep bounds work up front — skips files over `STEP_CORPUS_MAX_BYTES`
(2 MB) or `STEP_CORPUS_MAX_FACES` (160) and caps the octree at
`STEP_CORPUS_LEVELS` (6) — because one runaway mesh SIGKILLs the whole run;
all three are env-overridable for a deliberate full sweep.

Run both: `STEP_CORPUS_DIR=… cargo test --profile release-dev -F geometry --lib
corpus_ -- --ignored`. Both verified deterministic across re-runs.

Reading `corpus_mesh.txt`: `worst SJ 1.000` = the part refined to a single
octree level (every dual hex a cube); `worst SJ 0.258` = the part spans a 2:1
balance transition, and 0.258 is the characteristic worst dual hex at a level
jump — a stable fingerprint, not a defect (this is the pre-fit trimmed mesh;
fit quality is a separate concern). A *change* in that number is the signal.

## Cadence

- Weekly, or every ~10 commits: `/code-review high` on the range since each
  module's last-reviewed SHA (table below); bump the SHA.
- Every pass: re-check the risk register regardless of whether its SHA moved.
- Every new harness assertion: delete the code it guards, confirm it fails
  (`[[vacuous_test_mutation_check]]`).
- Freeze + stacked-PR split when passes stop finding structural issues.

## Module matrix

| Module | Reviewed through | Notes |
|---|---|---|
| `cad/part_21` | — | ISO-10303-21 tokenizer/parser; fuzz it |
| `cad/read/step/brep` | — | STEP entity graph → `Brep` |
| `cad/brep/{curve,surface}` | — | primitive curve/surface eval |
| `cad/brep/oracle/{mod,patch,sampled}` | — | **highest risk**; harness covers primitives + closed quadrics |
| `cad/brep/{classify,inside,orient,planar,tessellate,primitive,features}` | — | trimming/topology |
| `cad/sizing` | — | feature-size field |
| `cad/{assemble,mesh}` | — | orchestration |
| `csg/*` | `bb39c8cf` (Sonnet high; Opus pass pending) | analytic primitives + boolean ops |
| `solid` | `bb39c8cf` (Sonnet high; Opus pass pending) | shared octree→dual→trim→fit driver |
| `geometry/mesh/buffer` edits | — | regression surface on existing code |

## Risk register (re-check every pass)

- Cone/chamfer distance — see FINDING cone-distance below.
- `BrepOracle` fillet/chamfer faces still error (`primitive()` deferred).
- Mismatched-arc / two-edge planar trimming loops (recent commits).
- Cone apex: trim-ring carry across the apex singularity.
- Ray-parity sign determination: graze handling, majority vote fallback.
- Partial (open-shell) quadric patches: `signed_distance` is undefined on them;
  only `project` is meaningful. No standing invariant covers partial-face
  trimming accuracy yet — harness gap.

## Findings

### FINDING cone-distance — open

`BrepOracle` reports the **radial gap** to a conical face, not the perpendicular
distance to the slant. Two independent detections:

- `sdf_invariants_cone`: `|∇ signed_distance| ≈ 1.118` at an interior point by
  the wall of `cone(3,1,4)`. `1/cos(atan(0.5)) = 1.11803` — exact match, so the
  wall distance is scaled by `1/cos(semi_angle)`.
- `oracle_matches_primitive_cone`: exterior point past the top rim,
  `BrepOracle` −4.998 vs `csg::Cone` −4.496 (drift 0.5).

Impact: over-refinement and a biased boundary fit near conical/chamfer faces;
worse as the semi-angle grows. Fix: project onto the cone's generator line in
the axial (r, z) half-plane rather than differencing radii. Tests are
`#[ignore]`d with this tag until fixed; un-ignore to verify.

(`oracle_matches_primitive_cone` also depends on `csg::Cone`'s own SDF being
exact — confirm that separately when fixing.)

### csg/solid pass 1 (Sonnet `/code-review high`, `bb39c8cf`) — 4 open

All verified against the code. None block; fix opportunistically.

1. **`TorusOracle::project` off-surface for a query exactly on the tube centre
   circle** (`csg/torus/mod.rs:100`). When `offset` is zero the normal falls
   back to `perpendicular(axis)` — a fixed global direction, not radial at the
   ring point — so the returned "projection" is ~`minor·(1−cos)` *inside* the
   solid, not on the surface (e.g. query `[0,3,0]` on `Torus(0,+z,3,1)` →
   `[1,3,0]`, `signed_distance ≈ 0.84`). Fix: reuse `radial_unit` from
   `tube_frame` (or `axis`) as the fallback — both are genuine tube-frame
   directions and land on the surface. Severity low (measure-zero input,
   `1e-30` guard) but a real `project` correctness bug.

2. **`Solid::mesh` evaluates corner SDFs twice** (`solid/mod.rs:434`).
   `dual_background` → `classify` already does one `signed_distance` per node
   (plus per non-`Cut` centroid); `mesh` then keeps only the `Outside` bit and
   recomputes corner SDFs via `signed_distances` at :452. The centroid work and
   one full corner pass are wasted. `mesh` only needs an "entirely outside"
   predicate. Costs most on the ray-casting B-rep oracle this driver is built
   to share.

3. **`classify_by_signed_distance` is single-threaded** (`solid/mod.rs:105`).
   The default `Solid::classify` for every CSG primitive evaluates the oracle in
   a plain `.iter().map()` for nodes and again per centroid, while the sibling
   `classify_by_flood_fill` uses the threaded `signed_distances` helper (whose
   own doc calls SDF eval "the expensive part"). Route both passes through
   `signed_distances`.

4. **CSG combinators silently use the naive classifier** (`solid/mod.rs:347`).
   `classify_by_flood_fill` (thin-wall centroid probe, detached-`Cut`-island
   handling) is wired only into `impl Solid for Brep`; `Primitive` /
   `Difference` / `Union` / … fall through to `classify_by_signed_distance`,
   which has no sub-cell-feature guard. A `Difference(box, thin_pore)` whose
   pore is thinner than the local octree cell — all 8 corners in material,
   centroid missing the pore — is classified `Inside` and never trimmed. Real
   for the box−⋃pores use case unless the caller sizes the octree fine enough.
