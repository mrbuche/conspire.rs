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
| `csg/*` | — | analytic primitives + boolean ops |
| `solid` | — | shared octree→dual→trim→fit driver |
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
