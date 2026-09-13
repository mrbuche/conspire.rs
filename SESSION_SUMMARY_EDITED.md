# Crease-tangle investigation — session summary

**Date:** 2026-09-11
**Repo:** `/projects/conspire.rs` (conspire 0.7.8), meshed via `/projects/automesh`
**Model file:** `/projects/???.stp`
**Constraint honored throughout:** no git commits/branches/pushes; all edits left uncommitted
in the working tree. `automesh/Cargo.toml` left pointing at `../conspire.rs` (user handles).

---

## The problem

Meshing the section `.stp` to all-hex through the CAD buffer-fit path tangles the mesh along a
long internal sharp crease — **B-rep edge ??** — and worse at its two ends.
Visually: boundary elements are pulled *elongated through* their own wall onto surfaces farther
away, and the two crease ends are visibly disturbed.

### Root cause (verified, not hypothesized)

The buffer fit's per-quad surface target is the **global-nearest B-rep patch**
(`BrepOracle::nearest_scan`). Near the crease, the crease's own wall runs micron-close to unrelated features.
Query points near the **medial axis** are near-equidistant, so the winning patch **flips**
between a plane and a cylinder under microns of node motion during the fit sweeps. That flip
drags a boundary node across the gap onto the far surface — the elongation seen in the mesh.

Evidence gathered:
- `probe_crease_quad_flips` on the real trimmed mesh: **6374/18456 (34.5%)** of boundary
  quads near the crease flip their nearest patch; every flip pair is plane↔cylinder/cone.
- `probe_face_info` dumps confirmed the flipping surfaces are **legitimate distinct geometry**
  — this is **not** a CAD-file defect.
- The crease ends are dense **multi-surface junctions**: within microns, Winner flips between adjacent quads no matter
  what target rule is used.

---

## The metric harness

To judge fixes objectively, several `#[ignore]`d probes in
`src/geometry/cad/read/step/brep/test.rs` were used/extended (driven by `STEP_*` env vars):

- **`probe_crease_damage`** — full `brep.mesh(..., Fitting::Soft)`, then buckets every hex
  within `STEP_CREASE_DAMAGE_BAND` (3e-4 m) of edge #?? into 20 t-buckets along the crease,
  reporting per-bucket **max edge-ratio, max skew, min scaled-Jacobian (SJ)**. *This session
  added* `STEP_CREASE_DAMAGE_INTERIOR` (a t-margin) that discards hexes whose perpendicular
  foot falls off the segment ends — without it, buckets t=0.00 and t=0.95 are polluted by the
  whole far mesh clamping to the endpoints (39906 / 17604 hexes), masking the real endpoint
  quality. With the guard those buckets carry ~1600 genuinely-near-end hexes and become
  meaningful. **This guard is a keeper.**
- **`probe_crease_quad_flips`** — counts nearest-patch flips between adjacent crease-band
  quads on the trimmed (pre-fit) mesh; reports each quad's top-2 patches + gap + frozen owner.
- **`probe_mesh_real_file`** — full mesh, reports node/element count + global worst SJ, writes
  VTU stages to `STEP_MESH_OUT`.
- **`probe_holes` / `probe_trim_spur`** — global census: interior-hole faces and out-of-bbox
  spur cells on the trimmed mesh (both must stay 0).

**Standard mesh env** (baseline sizing): `STEP_MESH_MIN=1e-5 CELL=none SEGMENTS=24
GRADATION=0.25 PROXIMITY=4 CURVATURE=15 LEVELS=15`, plus `STEP_DISABLE_CREASE_PROXIMITY=1`
(temporary sizing gate, see below). Full mesh ≈ **457,982 nodes / 400,141 elements**.

**Success bar:** whole-crease damage measurably reduced, **zero new inversions** (global worst
SJ ≥ baseline +0.0175), interior-hole census 0, spur census 0, full test suite green.

---

## What we tried, in order

### 0. Baseline (pre-session state)
- Global worst SJ **+0.0175**; mid-crease max edge-ratio **7.81**; crease ends (interior-guarded)
  t=0.00 SJ **+0.0255** / ratio 22.8, t=0.95 SJ **+0.0388** / ratio 12.5.
- VTU: `section_baseline_fitted.vtu`.

### 1. Crease-*proximity sizing* term (abandoned early)
- **Idea:** a new `FeatureSizing` term forcing N elements across a narrow gap between two sharp
  edges (ported from the STL `separation` algorithm). Wired behind `STEP_DISABLE_CREASE_PROXIMITY`.
- **Outcome:** abandoned. The tangle is a *fit-target instability*, not a *sizing gap* — more
  elements don't stop a node being pulled through its wall. Superseded by the oracle-level work.
  The gate + partial scaffolding remain in `src/geometry/cad/sizing/mod.rs` (+437 lines) and
  `sizing/test.rs`; **should be reverted/cleaned** if not pursued.

### 2. Visibility gate (option 1 — "rule out walls you'd cross another wall to reach")
- **Idea:** in the oracle, reject a candidate nearest patch if the straight segment from query
  to it passes through another surface first (occlusion test), with tie-break machinery.
- **Outcome:** **half-worked, then rejected.** Global worst SJ *regressed* +0.0175 → **+0.01385**
  (`section_gate_fitted.vtu`, `mesh_gate.log`). Mid-crease slightly better but new edge-ratio
  spikes (11.7/13.4), **endpoints worse**. User confirmed visually the endpoints were bad and
  far-side pull persisted. The gate is too timid where a node already escaped to the far side,
  too aggressive at corners. **All gate machinery was removed.**

### 3. Ownership carry-through (option 2 — the main fix)
- **Idea:** freeze each boundary quad's owning B-rep patch **once**, from its pre-fit centroid
  (before any sweep drifts it toward a near-tie), then constrain every fit sweep to project
  only onto that frozen patch. A node can't be pulled onto an unrelated far surface because
  that surface isn't in its candidate set.
- **Plumbing (still in tree):**
  - fit `Oracle` trait (`buffer/fit/mod.rs`): added `owner(query)->Option<usize>` and
    `project_owned(query, owner)` with backward-compatible defaults (STL/CSG unaffected);
    `Mesh::fit` computes owners once (`owners()` helper) from pre-fit coords and threads them
    through `project()`.
  - `SolidOracle` trait (`solid/mod.rs`) + `Fit` bridge: same two methods forwarded.
  - `BrepOracle` (`oracle/mod.rs`): `owner` = nearest patch index; `project_owned` = closest
    point on that frozen patch.
- **Outcome (ungated):** **fixed the mid-crease** — max edge-ratio **7.81 → 4.90**, mid min-SJ
  steady +0.175. **But inverted the t=0 corner**: SJ **−0.0267**, edge-ratio 405
  (`crease_damage_owned_interior.log`). A frozen owner at a multi-surface junction pins the
  node to the wrong wall. Fails the zero-inversion bar.

### 4. Near-tie guard on the carry-through (kept)
- **Idea:** freeze the owner *only where unambiguous*. `owner()` uses `nearest_scan_gap`
  (winner + runner-up distance); if runner-up is within `OWNER_GAP` (10%) of the winner it's a
  near-tie (corner/junction) → return `None`, falling back to free projection.
- **Outcome — best measured state:**
  - Global worst SJ **+0.0175 → +0.03589** (>2×) — `section_gap_fitted.vtu`, `mesh_gap.log`.
  - Mid-crease max edge-ratio **7.81 → 4.68**; mid min-SJ +0.172.
  - Crease ends (interior-guarded): t=0.00 SJ **+0.0523** / ratio 11.2 (better than baseline!),
    t=0.95 SJ **+0.0498** / ratio 11.2.
  - Interior-hole census **0**, spur census **0** (over 388,760 kept cells).
  - Full lib suite **1423 passed, 0 failed** (25 ignored probes).
  - No bucket negative anywhere in the crease band.
- **But:** user reports the crease ends **still visually tangle**, and "elements pulled through"
  persists. The guard cured the *inversion* but not the *tangle* — the `probe_crease_quad_flips`
  census shows the ends flip just as much whether frozen or free (**12779 gated vs 11702 plain
  flips over 29414 quads**), because at a genuine 4+-surface junction *no single-surface target
  is stable*. The guard just disables stabilization exactly where it's needed.

### 5. Edge-curve / corner-vertex snap (option chosen after #4, ultimately insufficient)
- **Idea (standard mesher approach):** a boundary node near a crease should target the
  topological **edge curve** (1D) or **corner vertex** (0D), not any single face's closest
  point. A 1D/0D target can't flip between the two contested surfaces.
- **Implementation (still in tree, `oracle/mod.rs`):**
  - `BrepOracle` gains `creases: Vec<CreaseCurve>` (each crease edge's exact-endpoint chord
    polyline via `curve::chords`, plus a padded AABB) and `corner_points: Vec<Coordinate>`,
    built in `oracle()` from `self.features()`.
  - Owner indices `≥ patches.len()` **encode a crease** (`crease_owner`/`crease_index`).
  - `owner()`: if the quad centroid is within `CREASE_SNAP_RADIUS · nearest_patch_distance` of
    a crease, return the encoded crease owner (checked *before* the face freeze); else the
    near-tie-guarded face owner.
  - `project_owned()`: for a crease owner, closest point on the crease polyline
    (`nearest_crease` / `crease_target` / `closest_on_segment`), snapped exactly onto a corner
    vertex if within `CORNER_SNAP_RADIUS · patch_distance`.
- **Outcome — bracketed the design, no good operating point:**
  - `CREASE_SNAP_RADIUS = 0.6`: **no-op** — quad centroids sit ~1× wall-distance off the
    crease, so nothing snaps; census **identical** to the gap-gated carry-through (worst SJ
    +0.03589, `section_edge_fitted.vtu` ≡ `section_gap_fitted.vtu`).
  - `CREASE_SNAP_RADIUS = 1.5`: **over-snaps** — pulls whole quads onto the line, making rows
    perpendicular to the crease collinear/bunched. Mid-crease max edge-ratio **4.68 → ~14**,
    global worst SJ **+0.03589 → +0.02643** (`section_edge15_fitted.vtu`, `mesh_edge15.log`).
    User: this variant has "new bad problems too."
  - **Diagnosis:** per-**quad** centroid snapping is the wrong granularity. Only the mesh
    **nodes shared with the crease edge** should lie on the curve; snapping a whole quad's
    target drags its off-seam nodes onto the line and elongates the perpendicular rows. Doing
    it right needs a per-**node** hook in `Mesh::fit`, not per-quad ownership — not attempted.

### Verdict
Neither the carry-through (#4) nor the edge-snap (#5) removed the tangle to the user's
satisfaction. **Investigation stopped here** at the user's direction. The tree is left as-is
(edge-snap present, `CREASE_SNAP_RADIUS = 1.5`) for the user to save on a branch.

---

## Results table (interior-guarded crease census, edge #39)

| State | worst SJ (global) | mid-crease ratio | t=0.00 SJ / ratio | t=0.95 SJ / ratio |
|---|---|---|---|---|
| Baseline | +0.0175 | 7.81 | +0.0255 / 22.8 | +0.0388 / 12.5 |
| Visibility gate (#2) | +0.01385 | — (spikes 11.7/13.4) | worse | worse |
| Carry-through ungated (#3) | (mid fixed) | 4.90 | **−0.0267 / 405** ❌ | +0.0395 / 33.8 |
| **Carry-through + near-tie guard (#4)** | **+0.03589** | **4.68** | **+0.0523 / 11.2** | **+0.0498 / 11.2** |
| Edge-snap r=1.5 (#5) | +0.02643 | ~14 | +0.0396 / 13.0 | +0.0394 / 13.9 |

Best *measured* state = **#4** (all metrics improved over baseline, no inversions, censuses
clean, suite green) — but still visually tangled per the user, so not a true fix.

---

## Files modified this session (all uncommitted; `git diff --stat`)

```
 src/geometry/cad/brep/oracle/mod.rs     |  649 ++   # carry-through owner/project_owned,
                                                     #   near-tie guard, crease-snap machinery,
                                                     #   nearest_scan_gap, STEP_DISABLE_OWNER_CARRY,
                                                     #   STEP_NEAREST_TRACE, fit_target accessor
 src/geometry/cad/brep/oracle/patch.rs   |   33 ++  # (supporting patch closest/report tweaks)
 src/geometry/cad/read/step/brep/test.rs | 1470 ++  # probe_crease_damage (+INTERIOR guard),
                                                     #   probe_crease_quad_flips, probe_crease_tie,
                                                     #   probe_face_info, and other diagnostics
 src/geometry/cad/sizing/mod.rs          |  437 ++  # abandoned crease-proximity SIZING term
                                                     #   + STEP_DISABLE_CREASE_PROXIMITY gate
 src/geometry/cad/sizing/test.rs         |   53 ++  # tests for the abandoned sizing term
 src/geometry/mesh/buffer/fit/mod.rs     |  133 ++  # Oracle::owner/project_owned + owners()
                                                     #   helper, threaded through project()
 src/geometry/solid/mod.rs               |   34 ++  # SolidOracle::owner/project_owned + Fit bridge
```

### Temporary diagnostic gates left in code (remove if any of this is kept)
- `STEP_DISABLE_OWNER_CARRY` (`oracle/mod.rs`) — disables carry-through to regenerate a true
  baseline under identical bucketing.
- `STEP_DISABLE_CREASE_PROXIMITY` (`sizing/mod.rs`) — from the abandoned #1 sizing term.
- `STEP_CREASE_DAMAGE_INTERIOR` (`test.rs`) — legitimate probe improvement; a **keeper**.

### VTU artifacts (all 400,141 elements)
- `section_baseline_fitted.vtu` — pre-fix baseline (worst SJ +0.0175)
- `section_gate_fitted.vtu` — visibility gate (+0.01385, rejected)
- `section_gap_fitted.vtu` / `section_edge_fitted.vtu` — gap-gated carry-through (+0.03589)
- `section_edge15_fitted.vtu` — edge-snap radius 1.5 (+0.02643)

---

## If picking this back up

The unfinished lead is **per-node crease snapping**: instead of freezing a whole boundary
quad's owner to a crease, add a hook in `Mesh::fit` (`buffer/fit/mod.rs`) that, for the
specific boundary **nodes** coincident with a crease edge, constrains them to the crease
curve (and crease endpoints to the corner vertex) — leaving their neighboring off-seam nodes
free. This targets the flipping seam without the row-elongation that per-quad snapping caused.
The oracle-side geometry to support it (`creases`, `corner_points`, `nearest_crease`,
`crease_target`, `closest_on_segment`) is already built and can be reused; what's missing is
the per-node application point in the fit, and a way to identify crease-incident boundary
nodes (the exact-endpoint crease polylines + a distance threshold, or B-rep edge→node topology
carried through the buffer inflation).

Whether that fully cures the *ends* (genuine 4+-surface junctions) is unproven — the flip
census suggests the ends may be intrinsically hard and might additionally need explicit
corner-node pinning.
