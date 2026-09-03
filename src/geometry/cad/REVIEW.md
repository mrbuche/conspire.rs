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
| `cad/part_21` | `756f042d` (Sonnet high; Opus pass 529'd, retry) | 6 findings, no panics/loops found; see below |
| `cad/read/step/brep` | — | STEP entity graph → `Brep` |
| `cad/brep/{curve,surface}` | — | primitive curve/surface eval |
| `cad/brep/oracle/{mod,patch,sampled}` | — | **highest risk**; harness covers primitives + closed quadrics |
| `cad/brep/{classify,inside,orient,planar,tessellate,primitive,features}` | — | trimming/topology |
| `cad/sizing` | — | feature-size field |
| `cad/{assemble,mesh}` | — | orchestration |
| `csg/*` | `5422e853` (Sonnet high + Opus deep) | 10 findings; 1, 5, 6 fixed; 8, 10 open (notes) |
| `solid` | `5422e853` (Sonnet high + Opus deep) | 3, 7 fixed; 2 (not cheap), 4, 9 open |
| `geometry/mesh/buffer` edits | — | regression surface on existing code |

## Risk register (re-check every pass)

- CSG `project` near an overlap lens / CSG edge returns an operand foot, not the
  true nearest edge point — csg/solid finding 6 (predicate fixed, edge-candidate
  generation still open).
- `EllipsoidOracle` on a symmetry plane inside the focal set returns a
  stationary point, not the global nearest — csg/solid finding 5 residual.
- CSG classify path lacks the flood-fill thin-wall/void guards — csg/solid
  findings 4, 9 (7 fixed).
- `Solid::mesh` double-evaluates corner SDFs — csg/solid finding 2 (needs a
  classify→mesh contract change).
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
   **FIXED `702daceb`** — `tube_frame` returns the ring point's `radial_unit`;
   `project` uses it as the on-centre-circle fallback.

2. **`Solid::mesh` evaluates corner SDFs twice** (`solid/mod.rs:434`).
   `dual_background` → `classify` already does one `signed_distance` per node
   (plus per non-`Cut` centroid); `mesh` then keeps only the `Outside` bit and
   recomputes corner SDFs via `signed_distances` at :452. The centroid work and
   one full corner pass are wasted. `mesh` only needs an "entirely outside"
   predicate. Costs most on the ray-casting B-rep oracle this driver is built
   to share. **Not cheap** — the real fix threads the node SDFs `classify`
   computed out through `dual_background`, a `pub(crate)` signature + trait
   change; deferred to a dedicated pass.

3. **`classify_by_signed_distance` is single-threaded** (`solid/mod.rs:105`).
   The default `Solid::classify` for every CSG primitive evaluates the oracle in
   a plain `.iter().map()` for nodes and again per centroid, while the sibling
   `classify_by_flood_fill` uses the threaded `signed_distances` helper (whose
   own doc calls SDF eval "the expensive part"). Route both passes through
   `signed_distances`. **FIXED `ff56840e`** — node pass + masked centroid pass,
   both via `signed_distances`.

4. **CSG combinators silently use the naive classifier** (`solid/mod.rs:347`).
   `classify_by_flood_fill` (thin-wall centroid probe, detached-`Cut`-island
   handling) is wired only into `impl Solid for Brep`; `Primitive` /
   `Difference` / `Union` / … fall through to `classify_by_signed_distance`,
   which has no sub-cell-feature guard. A `Difference(box, thin_pore)` whose
   pore is thinner than the local octree cell — all 8 corners in material,
   centroid missing the pore — is classified `Inside` and never trimmed. Real
   for the box−⋃pores use case unless the caller sizes the octree fine enough.

### csg/solid pass 2 (Opus deep review, `5422e853`) — 3 confirmed + 3 notes

Opus verified each against a concrete input; the cuboid/cylinder/cone SDFs, all
tilted-primitive AABBs, `circumradius`'s Lipschitz bound, `HEX_FACES`
adjacency, and the `needed`/`keep_hexes` index alignment all checked out clean.

5. **CONFIRMED — `EllipsoidOracle::closest_local` returns an off-surface point
   for an interior query with a (near-)zero principal coordinate**
   (`csg/ellipsoid/mod.rs:112,126-138`). `TOLERANCE = 1e-12` is used both to
   nudge a zero coordinate (`y = p.abs().max(TOLERANCE)`) and as the *absolute*
   bisection stop (`hi - lo <= TOLERANCE*(1+|hi|)`), so when the root sits at
   `t ≈ -e_min² + O(1e-12)` it is never resolved and `closest[min] =
   e_min²·y_min/(t+e_min²)` becomes `O(1)` noise. `Ellipsoid::new(0,[1,2,3])`,
   query `(0,1,0)` (true closest `(0,2,0)`, dist 1) → the recogniser returns a
   point with `Σ(xᵢ/eᵢ)² ≈ 1.4` and a wrong distance. **Reachable, not
   exotic:** `refine_octree` deliberately snaps a grid plane through the world
   origin (`solid/mod.rs:503-511`), so a centred ellipsoid gets a whole plane
   of nodes with an exactly-zero coordinate, and for an oblate ellipsoid most
   of that plane lies inside the evolute. `Fit` then pulls those boundary nodes
   off the surface. Fix: separate the coordinate-nudge epsilon from a
   *relative* bisection stop, or special-case a zero coordinate (drop that axis
   and solve the lower-D problem).
   **FIXED `59a2df88`** — drops any axis within `AXIS_EPSILON` of zero and
   solves the remaining-axes Eberly problem via `eberly_root`, foot pinned to
   that plane. Residual limitation (pre-existing, out of the near-surface
   regime, not fixed): a query *inside the focal set* on a symmetry plane lands
   on an on-surface stationary point, not the global nearest.

6. **CONFIRMED — every boolean-op `project` tests the "patch survives the
   boolean" flag at the query, not at the candidate point**
   (`csg/ops/{union,intersection,difference,union_all}/mod.rs`). The correct
   test for `a.project(q) = p` is `b.signed_distance(&p) ⋚ 0`; the code passes
   `query`. Union, interior query `q=(0.93,0,0)` with `A=Sphere(0,1)`,
   `B=Sphere((1.9,0,0),1)`: `sa,sb > 0` so *both* candidates are flagged
   off-surface, `best_candidate` falls through to `any` and returns `(0.9,0,0)`
   — 0.1 *inside* the union, not on its boundary. Difference, exterior query
   `q=(3,0,0)` with `outer=Sphere(0,2)`, `inner=Sphere((1.9,0,0),1)`: the outer
   candidate `(2,0,0)` is flagged valid (`carved(q) ≤ 0`) and wins, though it
   was carved away (0.9 inside `inner`); the real boundary is the rim circle.
   Intersection/UnionAll share it. Directly degrades the fit for the box−⋃pores
   case. Fix: evaluate the survival predicate at `p`.
   **FIXED `a7075224`** — every combinator's survival test is now at the
   candidate point, and `best_candidate` carries a penalty (distance onto the
   wrong side of the boolean) so the fallback returns the least-buried
   candidate. Residual limitation (not fixed): a query deep in an overlap lens
   or beside a CSG edge still gets an operand foot, not the edge point — needs
   CSG-edge candidate generation.

7. **CONFIRMED — `Solid::mesh`'s trim rule deletes every cell the flood-fill
   thin-wall rescue just saved** (`solid/mod.rs:457-461` vs `190-209`). The
   rescue promotes a cell to `Cut` only in the `maximum < 0` branch (all
   corners in air, centroid in solid). `mesh` then keeps a non-`Outside` cell
   iff `minimum + 0.1·maximum ≥ 0`; with `minimum ≤ maximum < 0` that is always
   negative, so the rescued cell is dropped immediately. The rescue only
   survives via `trim()` (keeps all non-`Outside`). The `Plate` case
   `solid/test.rs:154` asserts kept would mesh to nothing through `mesh()`.
   Fix: exempt `Cut` cells with all corners outside from the trim rule, or
   record the probed centroid value and use it as `maximum`.
   **FIXED `e4d9debb`** — per-cell decision factored into `survives_trim`,
   which keeps a `Cut` cell with `maximum < 0`.

8. **PLAUSIBLE / doc — the Tong `TRIM_RATIO` rule assumes a true Euclidean
   distance, which min/max CSG composition is not** (`solid/mod.rs:461`). Signs
   are always right (classification safe) but magnitudes are understated near
   creases: at an `Intersection`'s convex edge a corner 0.3+0.3 outside reports
   −0.3 not −0.424, biasing ~40% of a cell toward keeping protruding cells;
   symmetric over-trim at a `Union`'s reentrant crease. At least note that
   `TRIM_RATIO` is calibrated for exact fields.

9. **PLAUSIBLE — the thin-feature probe is one-sided** (`solid/mod.rs:190-201`).
   Only cells with `maximum < 0` (thin solid wall) are probed. The mirror — all
   eight corners inside, a sub-cell-thin air slot crossing the cell — passes
   the straddle test, is never probed, and is labelled `Inside`, filling the
   slot. Symmetric guard: `minimum > 0 && maximum - circumradius < 0` plus a
   centroid probe for a negative value. (Dual of finding 4.)

10. **NOTE — `CuboidOracle` exterior normal in the edge/corner Voronoi region
    is the bevel direction, not either adjacent face** (`csg/cuboid/mod.rs:64`).
    `normal = unit(query − clamped)`, so `fit::project`'s `(x−p)·n` deviation
    pulls boundary-layer quads straddling a box edge toward a chamfer — the same
    feature-preservation failure mode recorded for the tessellation path.
    Snapping the normal to the dominant clamped axis keeps `p` identical while
    restoring a face normal.

### part_21 pass 1 (Sonnet `/code-review high`, `756f042d`) — 6 open

No panics or infinite loops found: every `from_utf8().unwrap()` is on an
ASCII-only span, every loop makes progress or returns `Err`, `position` never
runs past `bytes.len()`. The findings are robustness / spec-coverage gaps.
(Opus pass 529'd mid-run — retry for a second opinion.)

P1. **Unbounded recursion in `parameter()`/`parameters()`** (`part_21/mod.rs:274`).
    `parameter → List → parameters → parameter` and `Typed → Box::new(parameter)`
    carry no depth limit. A crafted/corrupt `.stp` with `((((…))))` nested tens
    of thousands deep overflows the thread stack → uncatchable `SIGABRT`, a bad
    file crashes the whole meshing run. Fix: thread a depth counter, `Err` past
    a bound (~256).

P2. **Exactly one HEADER + one DATA section** (`part_21/mod.rs:72`). Valid Part 21
    files with multiple `DATA;` blocks, or edition-3 `ANCHOR`/`REFERENCE`/
    `SIGNATURE` sections, are rejected at the `literal("END-ISO-10303-21")` after
    the first `ENDSEC`. (No corpus file hits this — NIST e3 files parse — so
    lower priority.)

P3. **`string()` doesn't decode STEP control directives** (`part_21/mod.rs:302`).
    Only `''` un-doubling; `\X\`, `\X2\…\X0\`, `\X4\…\X0\`, `\S\`, `\P…` pass
    through literally, so any non-ASCII string value is returned corrupted (e.g.
    `'caf\X2\00E9\X0\'` → `caf\X2\00E9\X0\`). Latent — today's reader only reads
    ASCII fields — but the module is billed as the shared Part 21 parser.

P4. **`number()` maps out-of-range reals to ±inf, rejects >i64 integers**
    (`part_21/mod.rs:366`). `1.0E400` → `Ok(f64::INFINITY)` with no error → an
    infinite coordinate flows into geometry. A STEP INTEGER beyond i64 (spec
    says unbounded) aborts the whole file parse. Fix: reject non-finite reals.

P5. **`trivia()` doesn't skip a UTF-8 BOM** (`part_21/mod.rs:62`). A file saved
    with a leading `EF BB BF` fails on the first token (`expected ISO-10303-21
    at byte 0`). Fix: strip a leading BOM in `parse()`.

P6. **No EOF check after `END-ISO-10303-21;`** (`part_21/mod.rs:85`). Trailing
    garbage or a second concatenated document is silently ignored and the file
    reported valid. Fix: after the trailing terminator, error if `trivia()`
    leaves anything.
