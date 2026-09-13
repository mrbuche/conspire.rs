//! Volume sizing fields derived from B-rep geometry.

#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate,
        cad::brep::{
            Brep, Edge, Face,
            curve::Curve,
            oracle::BrepOracle,
            surface::{Cone, Cylinder, Sphere, Surface, Torus},
        },
        solid::Sizing,
    },
    math::{FxHashMap, FxHashSet, Quantity, Scalar, Tensor},
    units::Length,
};
use std::{
    array::from_fn,
    f64::consts::{PI, TAU},
};

const D: usize = 3;

/// How many places along a pair of creases the surface between them is
/// sampled to decide whether they bound a ribbon of surface (see `joined`).
const JOINED_SAMPLES: usize = 5;

/// How far, as a fraction of a pair's own gap, the surface joining two creases
/// may stray from the straight line between them and still count as joining
/// them (see `joined`).
const JOINED_FRACTION: Scalar = 0.25;

/// A binary AABB tree over sizing primitives: proximity slabs (a surface tile
/// pushed inward along its face normal by the local wall thickness, so a cell
/// buried mid-wall still overlaps it) or crease chords. Each node caches
/// `floor`, a lower bound on the size any primitive below it can ask for, so a
/// query that already has a smaller answer skips the whole subtree.
struct Bvh<T> {
    low: [Scalar; D],
    high: [Scalar; D],
    floor: Scalar,
    node: BvhNode<T>,
}

enum BvhNode<T> {
    Leaf(T),
    Split(Box<Bvh<T>>, Box<Bvh<T>>),
}

/// `(aabb low, aabb high, size floor, payload)` for one primitive.
type Item<T> = ([Scalar; D], [Scalar; D], Scalar, T);

impl<T> Bvh<T> {
    fn build(mut items: Vec<Item<T>>) -> Bvh<T> {
        let (mut low, mut high) = ([Scalar::INFINITY; D], [Scalar::NEG_INFINITY; D]);
        let mut floor = Scalar::INFINITY;
        for (item_low, item_high, item_floor, _) in &items {
            for k in 0..D {
                low[k] = low[k].min(item_low[k]);
                high[k] = high[k].max(item_high[k]);
            }
            floor = floor.min(*item_floor);
        }
        if items.len() == 1 {
            let (low, high, floor, payload) = items.pop().unwrap();
            return Bvh {
                low,
                high,
                floor,
                node: BvhNode::Leaf(payload),
            };
        }
        let axis = (0..D)
            .max_by(|&a, &b| (high[a] - low[a]).total_cmp(&(high[b] - low[b])))
            .unwrap();
        items.sort_by(|a, b| (a.0[axis] + a.1[axis]).total_cmp(&(b.0[axis] + b.1[axis])));
        let right = items.split_off(items.len() / 2);
        Bvh {
            low,
            high,
            floor,
            node: BvhNode::Split(Box::new(Bvh::build(items)), Box::new(Bvh::build(right))),
        }
    }

    /// Lower `best` to the smallest size any primitive overlapping the box
    /// `[low, high]` reports, via `eval` (which itself lowers `best`).
    fn query(
        &self,
        low: &[Scalar; D],
        high: &[Scalar; D],
        best: &mut Scalar,
        eval: &mut impl FnMut(&T, &mut Scalar),
    ) {
        if *best <= self.floor || (0..D).any(|k| self.low[k] > high[k] || self.high[k] < low[k]) {
            return;
        }
        match &self.node {
            BvhNode::Leaf(payload) => eval(payload, best),
            BvhNode::Split(a, b) => {
                a.query(low, high, best, eval);
                b.query(low, high, best, eval);
            }
        }
    }
}

/// A BVH of axis-aligned boxes, each tagged with a target element size: the
/// smallest tag over every box that overlaps a query cell. Used for both the
/// proximity slabs (solid behind a thin wall) and the curvature bands (a shell
/// hugging a curved face).
struct BoxField {
    bvh: Bvh<Scalar>,
}

impl BoxField {
    /// The smallest box target whose box overlaps the cube `center ± half`,
    /// or `None` if no box reaches the cell.
    fn target(&self, center: &Coordinate<D>, half: Scalar) -> Option<Scalar> {
        let low: [Scalar; D] = from_fn(|k| center[k].value() - half);
        let high: [Scalar; D] = from_fn(|k| center[k].value() + half);
        let mut best = Scalar::INFINITY;
        self.bvh
            .query(&low, &high, &mut best, &mut |&target, best| {
                *best = best.min(target)
            });
        best.is_finite().then_some(best)
    }
}

/// One crease chord and the feature size it imposes at its own location.
struct CreaseSeg {
    a: [Scalar; D],
    b: [Scalar; D],
    source: Scalar,
}

/// A scalar sizing field driven by B-rep feature edges.
///
/// Only sharp edges ([`Brep::features`] creases) contribute — seams and other
/// parameterization artifacts are ignored. A curved edge is sampled into a
/// chord polyline so the whole rim drives refinement, not just its endpoints.
/// Each contributing chord carries a target size of the edge's arc length over
/// `segments_per_edge`. `gradation` is the rate that size may grow per unit
/// distance from the edge (the mesh-smoothness bound); `Some(0.0)` pins the
/// whole field to the finest feature size, `None` lets it grow as fast as it
/// likes — the feature size within one target-size of the edge, `maximum`
/// beyond, a single fine layer per feature.
///
/// `maximum` is `None` for no ceiling: cells far from every feature grow as
/// large as the octree root, so the mesh adapts away from the part instead of
/// filling the empty bounding box with `maximum`-sized cells.
///
/// [`with_proximity`](Self::with_proximity) adds a local-feature-size term for
/// thin geometry the crease term is blind to (a thin wall or a small cavity
/// need not be near any sharp edge). [`with_curvature`](Self::with_curvature)
/// adds a term that resolves a curved face by its own radius, not only at its
/// sharp rims. [`with_crease_proximity`](Self::with_crease_proximity) adds a
/// term for narrow features bounded by *two* sharp edges — a thin rib, a slot,
/// a stair-step — that the through-thickness term misses because the solid
/// either side of such a gap need not itself be thin. The field is the clamped
/// minimum of all contributions, so it is defined everywhere.
pub struct FeatureSizing {
    crease: Option<Bvh<CreaseSeg>>,
    minimum: Quantity<Length>,
    /// `INFINITY` when the caller passed `None` (no ceiling).
    maximum: Quantity<Length>,
    gradation: Option<Scalar>,
    proximity: Option<BoxField>,
    curvature: Option<BoxField>,
    separation: Option<BoxField>,
}

impl FeatureSizing {
    pub fn of(
        brep: &Brep,
        segments_per_edge: usize,
        minimum: Quantity<Length>,
        maximum: Option<Quantity<Length>>,
        gradation: Option<Scalar>,
    ) -> Self {
        let maximum = maximum.unwrap_or_else(|| Quantity::new(Scalar::INFINITY));
        let samples = segments_per_edge.max(1);
        let divisor = samples as Scalar;
        // A chord can never matter past the far side of the part; cap its
        // influence box there so an unbounded `maximum` (INFINITY reach) still
        // builds a finite BVH.
        let (mut cloud_low, mut cloud_high) = ([Scalar::INFINITY; D], [Scalar::NEG_INFINITY; D]);
        for vertex in &brep.vertices {
            for k in 0..D {
                cloud_low[k] = cloud_low[k].min(vertex[k].value());
                cloud_high[k] = cloud_high[k].max(vertex[k].value());
            }
        }
        let reach_cap = (0..D)
            .map(|k| cloud_high[k] - cloud_low[k])
            .fold(0.0_f64, Scalar::max)
            .max(minimum.value());
        let mut items: Vec<Item<CreaseSeg>> = Vec::new();
        for &edge in &brep.features().creases {
            let polyline = sample_edge(brep, &brep.edges[edge], samples);
            let length: Scalar = polyline
                .windows(2)
                .map(|pair| (&pair[1] - &pair[0]).norm().value())
                .sum();
            if length <= 0.0 {
                continue;
            }
            let source = (length / divisor).max(minimum.value()).min(maximum.value());
            // The chord stops mattering past the reach where its ramped size
            // hits `maximum`; inflate its box by that (capped at the part's own
            // size) so a point query only visits chords that can constrain it.
            let radius = match gradation {
                Some(rate) if rate > 1.0e-30 => (maximum.value() - source) / rate,
                Some(_) => reach_cap,
                None => source,
            }
            .min(reach_cap);
            for pair in polyline.windows(2) {
                let a: [Scalar; D] = from_fn(|k| pair[0][k].value());
                let b: [Scalar; D] = from_fn(|k| pair[1][k].value());
                let low = from_fn(|k| a[k].min(b[k]) - radius);
                let high = from_fn(|k| a[k].max(b[k]) + radius);
                items.push((low, high, source, CreaseSeg { a, b, source }));
            }
        }
        Self {
            crease: (!items.is_empty()).then(|| Bvh::build(items)),
            minimum,
            maximum,
            gradation,
            proximity: None,
            curvature: None,
            separation: None,
        }
    }

    /// Adds a surface-anchored local-feature-size term: every planar,
    /// cylindrical or conical face is tiled, each tile's local wall thickness
    /// is measured with one inward ray, and a box covering the slab of solid
    /// behind the tile is stored with a target of `thickness / cells_across`.
    /// Octree refinement then guarantees `cells_across` elements through any
    /// wall — including through its middle, where a sharp-edge term is blind and
    /// a bare face point-sample would miss, and all the way around a bore, where
    /// tiling only the neighbouring planar faces would leave an azimuthal gap.
    /// Spherical and toroidal thin walls are not yet anchored.
    pub fn with_proximity(
        mut self,
        brep: &Brep,
        cells_across: usize,
    ) -> Result<Self, &'static str> {
        let oracle = brep.oracle()?;
        let cells = cells_across.max(1) as Scalar;
        // Sample each face at roughly `maximum`, but no coarser than 1/16 of
        // the part (so an unbounded `maximum` still tiles).
        let (bounds_low, bounds_high) = oracle.bounds();
        let span = (0..D)
            .map(|k| bounds_high[k].value() - bounds_low[k].value())
            .fold(0.0_f64, Scalar::max);
        let tile = self
            .maximum
            .value()
            .min(span / 16.0)
            .max(self.minimum.value());
        let eps = tile * 1.0e-4;
        let mut slabs: Vec<Item<Scalar>> = Vec::new();
        for face in &brep.faces {
            let planar = match brep.planar_face(face) {
                Ok(planar) => planar,
                Err(_) => {
                    proximity_revolved(
                        brep,
                        face,
                        &oracle,
                        cells,
                        tile,
                        eps,
                        self.minimum.value(),
                        self.maximum.value(),
                        &mut slabs,
                    );
                    continue;
                }
            };
            let normal = from_fn::<Scalar, D, _>(|k| planar.normal[k].value());
            let (mut min_uv, mut max_uv) = ([Scalar::INFINITY; 2], [Scalar::NEG_INFINITY; 2]);
            for ring in &planar.rings {
                for &(uv, _) in ring {
                    for a in 0..2 {
                        min_uv[a] = min_uv[a].min(uv[a]);
                        max_uv[a] = max_uv[a].max(uv[a]);
                    }
                }
            }
            for &(centre, radius) in &planar.circles {
                for a in 0..2 {
                    min_uv[a] = min_uv[a].min(centre[a] - radius);
                    max_uv[a] = max_uv[a].max(centre[a] + radius);
                }
            }
            if (0..2).any(|a| max_uv[a] <= min_uv[a]) {
                continue;
            }
            let counts: [usize; 2] =
                from_fn(|a| (((max_uv[a] - min_uv[a]) / tile).ceil() as usize).clamp(1, 24));
            let step: [Scalar; 2] = from_fn(|a| (max_uv[a] - min_uv[a]) / counts[a] as Scalar);
            for iu in 0..counts[0] {
                for iv in 0..counts[1] {
                    let centre = [
                        min_uv[0] + (iu as Scalar + 0.5) * step[0],
                        min_uv[1] + (iv as Scalar + 0.5) * step[1],
                    ];
                    if !planar.contains(centre) {
                        continue;
                    }
                    let surface = planar.unproject(centre);
                    let inside = Coordinate::<D>::from(from_fn::<Scalar, D, _>(|k| {
                        surface[k].value() - eps * normal[k]
                    }));
                    let inward = from_fn::<Scalar, D, _>(|k| -normal[k]);
                    let Some(hit) = oracle.ray_distance(&inside, inward) else {
                        continue;
                    };
                    let thickness = hit + eps;
                    if !(thickness.is_finite() && thickness > 0.0) {
                        continue;
                    }
                    let target = (thickness / cells)
                        .max(self.minimum.value())
                        .min(self.maximum.value());
                    if target >= self.maximum.value() {
                        continue; // does not constrain anything
                    }
                    let (mut low, mut high) = ([Scalar::INFINITY; D], [Scalar::NEG_INFINITY; D]);
                    for du in [-0.5, 0.5] {
                        for dv in [-0.5, 0.5] {
                            let corner = [centre[0] + du * step[0], centre[1] + dv * step[1]];
                            let face_point = planar.unproject(corner);
                            for depth in [0.0, thickness] {
                                for k in 0..D {
                                    let c = face_point[k].value() - depth * normal[k];
                                    low[k] = low[k].min(c);
                                    high[k] = high[k].max(c);
                                }
                            }
                        }
                    }
                    slabs.push((
                        from_fn(|k| low[k] - eps),
                        from_fn(|k| high[k] + eps),
                        target,
                        target,
                    ));
                }
            }
        }
        // Crease-adjacent thickness: a sharp concave edge between two planar
        // faces can run close and parallel to a third, unrelated face (a
        // nearby fillet or bore) without either adjacent face itself being
        // thin anywhere the face-tiling above samples — the gap only shows up
        // right at the crease. Fire one ray per crease sample, inward along
        // the bisector of the two adjacent planes, and slab the same way.
        //
        // Proof-of-concept scope: only creases whose *both* incident faces
        // are planar get a bisector (matches the case this was built against,
        // a sharp corner between two planes running near a third surface);
        // a crease touching a curved face is left to the through-thickness
        // tiling of that curved face instead.
        //
        // TEMPORARY: gated behind STEP_DISABLE_CREASE_PROXIMITY so a baseline
        // (pre-this-session) mesh can still be produced for comparison while
        // this term is under evaluation — remove the gate once a verdict is
        // reached either way.
        const CREASE_SAMPLES: usize = 9;
        if std::env::var("STEP_DISABLE_CREASE_PROXIMITY").is_err() {
        for &edge in &brep.features().creases {
            let brep_edge = &brep.edges[edge];
            let mut normals: Vec<[Scalar; D]> = Vec::new();
            for face in &brep.faces {
                let touches = face
                    .bounds
                    .iter()
                    .any(|bound| bound.half_edges.iter().any(|he| he.edge == edge));
                if touches {
                    let Surface::Plane(plane) = &face.surface else {
                        normals.clear();
                        break;
                    };
                    let sign = if face.forward { 1.0 } else { -1.0 };
                    normals.push(from_fn(|k| sign * plane.normal[k].value()));
                }
            }
            let [n0, n1] = match normals.as_slice() {
                [a, b] => [*a, *b],
                _ => continue,
            };
            let bisector_raw: [Scalar; D] = from_fn(|k| -(n0[k] + n1[k]));
            let norm = dot(bisector_raw, bisector_raw).sqrt();
            if norm <= 1.0e-12 {
                continue; // faces back-to-back: not a real concave wedge
            }
            let bisector: [Scalar; D] = from_fn(|k| bisector_raw[k] / norm);
            // `sample_edge` special-cases `Curve::Line` to just the two
            // endpoints (a straight chord has no interior geometry to
            // resolve), which is exactly wrong here: we want densely-spaced
            // ray origins along the *whole* crease, straight or not, so a
            // gap that is uniform along a long straight edge still gets
            // sampled in its interior and not just at its two ends.
            let polyline = if let Curve::Line(_) = &brep_edge.curve {
                let [ia, ib] = brep_edge.vertices;
                let (a, b) = (point(&brep.vertices[ia]), point(&brep.vertices[ib]));
                (0..=CREASE_SAMPLES)
                    .map(|i| {
                        let t = i as Scalar / CREASE_SAMPLES as Scalar;
                        Coordinate::<D>::from(from_fn(|k| a[k] + t * (b[k] - a[k])))
                    })
                    .collect()
            } else {
                sample_edge(brep, brep_edge, CREASE_SAMPLES)
            };
            // A box built around a single ray (inflated only by `eps`) is a
            // near-zero-width sliver: with samples spaced along a >1mm edge,
            // consecutive slivers leave the whole interior of the crease
            // uncovered. Instead, deposit one box per *segment* of the
            // polyline, spanning both its ray origins and both hit tips, so
            // adjacent boxes butt up and the crease's whole length is
            // continuously covered end to end.
            for pair in polyline.windows(2) {
                let bases: [[Scalar; D]; 2] =
                    from_fn(|i| from_fn(|k| pair[i][k].value()));
                let mut thicknesses = [Scalar::NAN; 2];
                let mut ok = true;
                for (i, base) in bases.iter().enumerate() {
                    let origin = Coordinate::<D>::from(from_fn::<Scalar, D, _>(|k| {
                        base[k] + eps * bisector[k]
                    }));
                    match oracle.ray_distance(&origin, bisector) {
                        Some(hit) if (hit + eps).is_finite() && hit + eps > 0.0 => {
                            thicknesses[i] = hit + eps;
                        }
                        _ => ok = false,
                    }
                }
                if !ok {
                    continue;
                }
                let thickness = thicknesses[0].min(thicknesses[1]);
                // Unlike every other term here, don't clamp this one up to
                // `self.minimum`: the whole point is a *genuinely* narrow
                // gap that is finer than the user's chosen floor. Clamping
                // it up to the floor is what produced a uniform near-floor
                // band down the entire crease and tangled where that band
                // met the surrounding coarser field. Let the octree's own
                // `levels` cap be the real floor instead; only cap above at
                // `maximum` so the term never *coarsens* anything.
                let target = (thickness / cells).min(self.maximum.value());
                if target >= self.maximum.value() {
                    continue; // does not constrain anything
                }
                let (mut low, mut high) = ([Scalar::INFINITY; D], [Scalar::NEG_INFINITY; D]);
                for (base, thick) in bases.iter().zip(thicknesses) {
                    let tip: [Scalar; D] = from_fn(|k| base[k] + thick * bisector[k]);
                    for k in 0..D {
                        low[k] = low[k].min(base[k]).min(tip[k]) - eps;
                        high[k] = high[k].max(base[k]).max(tip[k]) + eps;
                    }
                }
                slabs.push((low, high, target, target));
            }
        }
        }
        self.proximity = (!slabs.is_empty()).then(|| BoxField {
            bvh: Bvh::build(slabs),
        });
        Ok(self)
    }

    /// Adds a curvature term: every cylindrical, conical, spherical or toroidal
    /// face is tiled in its own parametrization and a band of boxes hugging the
    /// patch is stored with a target of `TAU * R / sections` — `sections`
    /// elements around a full circle of the local curvature radius `R` (the
    /// tube radius for a torus, the perpendicular-to-rulings radius for a
    /// cone). Octree refinement then resolves a curved wall by its radius, so a
    /// long slender cylinder no longer coarsens to nothing between its rims.
    /// B-spline faces are not yet handled. The face's parametric footprint is
    /// recovered from its bounding edges; a gap under an eighth of a turn is
    /// read as a full revolution.
    pub fn with_curvature(mut self, brep: &Brep, sections: usize) -> Result<Self, &'static str> {
        let sections = sections.max(1) as Scalar;
        let (minimum, maximum) = (self.minimum.value(), self.maximum.value());
        let mut boxes: Vec<Item<Scalar>> = Vec::new();
        for face in &brep.faces {
            let mut points: Vec<[Scalar; D]> = Vec::new();
            for bound in &face.bounds {
                for half_edge in &bound.half_edges {
                    let Some(edge) = brep.edges.get(half_edge.edge) else {
                        continue;
                    };
                    for sample in sample_edge(brep, edge, 24) {
                        points.push(from_fn(|k| sample[k].value()));
                    }
                }
            }
            for &pole in &face.poles {
                points.push(from_fn(|k| brep.vertices[pole][k].value()));
            }
            if points.len() < 2 {
                continue;
            }
            match &face.surface {
                Surface::Cylinder(cylinder) => {
                    curvature_cylinder(cylinder, &points, sections, minimum, maximum, &mut boxes)
                }
                Surface::Cone(cone) => {
                    curvature_cone(cone, &points, sections, minimum, maximum, &mut boxes)
                }
                Surface::Sphere(sphere) => {
                    curvature_sphere(sphere, &points, sections, minimum, maximum, &mut boxes)
                }
                Surface::Torus(torus) => {
                    curvature_torus(torus, &points, sections, minimum, maximum, &mut boxes)
                }
                Surface::Plane(_) | Surface::BSpline(_) | Surface::Revolution(_) => {}
            }
        }
        self.curvature = (!boxes.is_empty()).then(|| BoxField {
            bvh: Bvh::build(boxes),
        });
        Ok(self)
    }

    /// Adds a crease-separation term: wherever two sharp edges run close
    /// together with surface between them the whole way — the two edges of a
    /// thin rib, the walls of a slot, the tread of a stair-step — a band of
    /// boxes spanning the gap is stored with a target of `gap / cells_across`,
    /// forcing that many elements across the narrow region.
    ///
    /// This is the local-feature-size case the through-thickness term is blind
    /// to: the solid on either side of such a gap need not itself be thin, so
    /// no inward ray finds a short chord. It reads the sharp edges straight off
    /// [`Brep::features`] — no tessellation, no angle inference.
    ///
    /// Two creases are compared only when they belong to *unrelated* chains: a
    /// chain is a maximal run of creases meeting end to end at non-corner
    /// vertices, and a crease never sees a chain within `hops` steps of its own
    /// (so a sharp edge does not read its own smoothly-continuing neighbours as
    /// a narrow feature). A pair counts only when surface runs between the two
    /// the whole way they face each other (`joined`), which rejects the open
    /// mouth of a slot — sharp on both lips but empty between them.
    pub fn with_crease_proximity(
        mut self,
        brep: &Brep,
        radius: Quantity<Length>,
        cells_across: usize,
        hops: usize,
    ) -> Result<Self, &'static str> {
        let features = brep.features();
        if features.creases.is_empty() {
            return Ok(self);
        }
        let oracle = brep.oracle()?;
        let cells = cells_across.max(1) as Scalar;
        let radius = radius.value();
        let (minimum, maximum) = (self.minimum.value(), self.maximum.value());

        // Each crease as its two endpoints (for segment-distance) plus the
        // chord polyline (to lay boxes down its length).
        let creases: Vec<([Scalar; D], [Scalar; D])> = features
            .creases
            .iter()
            .map(|&edge| {
                let [a, b] = brep.edges[edge].vertices;
                (point(&brep.vertices[a]), point(&brep.vertices[b]))
            })
            .collect();

        // Vertex -> incident creases, by inverting `edge.vertices` over the
        // sharp edges (mirrors `Brep::features`).
        let mut incident: Vec<Vec<usize>> = vec![Vec::new(); brep.vertices.len()];
        for (index, &edge) in features.creases.iter().enumerate() {
            let [a, b] = brep.edges[edge].vertices;
            incident[a].push(index);
            incident[b].push(index);
        }
        let corner: Vec<bool> = {
            let mut is_corner = vec![false; brep.vertices.len()];
            for &vertex in &features.corners {
                is_corner[vertex] = true;
            }
            is_corner
        };

        // Chains: union two creases meeting at a non-corner vertex with
        // exactly two creases through it.
        let mut parent: Vec<usize> = (0..creases.len()).collect();
        fn root(parent: &mut [usize], mut crease: usize) -> usize {
            while parent[crease] != crease {
                parent[crease] = parent[parent[crease]];
                crease = parent[crease];
            }
            crease
        }
        for vertex in 0..brep.vertices.len() {
            let through = &incident[vertex];
            if !corner[vertex] && through.len() == 2 {
                let (one, two) = (root(&mut parent, through[0]), root(&mut parent, through[1]));
                parent[one] = two;
            }
        }
        let chain: Vec<usize> = (0..creases.len())
            .map(|crease| root(&mut parent, crease))
            .collect();

        // Chain adjacency: two chains are adjacent when creases of each meet at
        // a shared vertex.
        let mut adjacent: FxHashMap<usize, FxHashSet<usize>> =
            FxHashMap::default();
        for through in &incident {
            for &one in through {
                for &two in through {
                    if chain[one] != chain[two] {
                        adjacent.entry(chain[one]).or_default().insert(chain[two]);
                    }
                }
            }
        }
        // Chains excluded from a chain: itself and everything within `hops`.
        let mut excluded: FxHashMap<usize, FxHashSet<usize>> =
            FxHashMap::default();
        for &start in &chain {
            excluded.entry(start).or_insert_with(|| {
                let mut reached = FxHashSet::default();
                reached.insert(start);
                let mut frontier = vec![start];
                for _ in 0..hops {
                    let mut next = Vec::new();
                    for near in &frontier {
                        if let Some(neighbors) = adjacent.get(near) {
                            for &other in neighbors {
                                if reached.insert(other) {
                                    next.push(other);
                                }
                            }
                        }
                    }
                    frontier = next;
                }
                reached
            });
        }

        // The crease of each other chain nearest each crease, recorded from
        // both ends so the two agree.
        let mut nearest: Vec<FxHashMap<usize, (usize, Scalar)>> =
            vec![FxHashMap::default(); creases.len()];
        let record = |nearest: &mut Vec<FxHashMap<usize, (usize, Scalar)>>,
                          this: usize,
                          other: usize,
                          distance: Scalar| {
            let entry = nearest[this]
                .entry(chain[other])
                .or_insert((other, distance));
            if (distance, other) < (entry.1, entry.0) {
                *entry = (other, distance);
            }
        };
        for this in 0..creases.len() {
            for other in (this + 1)..creases.len() {
                if excluded[&chain[this]].contains(&chain[other]) {
                    continue;
                }
                let distance = segment_distance(&creases[this], &creases[other]);
                if distance < radius && joined(&creases[this], &creases[other], &oracle) {
                    record(&mut nearest, this, other, distance);
                    record(&mut nearest, other, this, distance);
                }
            }
        }

        // Deposit boxes for every pair where each crease is its own chain's
        // nearest to the other.
        let eps = minimum * 1.0e-3;
        let mut boxes: Vec<Item<Scalar>> = Vec::new();
        for this in 0..creases.len() {
            for &(other, distance) in nearest[this].values() {
                // Mutual-nearest: `other` must report `this` back.
                let mutual = nearest[other]
                    .get(&chain[this])
                    .is_some_and(|&(back, _)| back == this);
                // Each pair once (this < other), and only when mutual.
                if !mutual || this >= other {
                    continue;
                }
                let target = (distance / cells).max(minimum).min(maximum);
                if target >= maximum {
                    continue;
                }
                // Walk `this` and drop to `other`, boxing each gap span.
                let (from, to) = (&creases[this], &creases[other]);
                for sample in 0..JOINED_SAMPLES {
                    let fraction = sample as Scalar / (JOINED_SAMPLES - 1) as Scalar;
                    let here: [Scalar; D] = from_fn(|k| from.0[k] + fraction * (from.1[k] - from.0[k]));
                    let there = closest_on(to, &here);
                    let gap = dot(sub(there, here), sub(there, here)).sqrt();
                    if gap > radius {
                        continue;
                    }
                    let half = 0.5 * gap + target;
                    let midpoint: [Scalar; D] = from_fn(|k| 0.5 * (here[k] + there[k]));
                    let low = from_fn(|k| midpoint[k] - half - eps);
                    let high = from_fn(|k| midpoint[k] + half + eps);
                    boxes.push((low, high, target, target));
                }
            }
        }
        self.separation = (!boxes.is_empty()).then(|| BoxField {
            bvh: Bvh::build(boxes),
        });
        Ok(self)
    }

    /// The crease term alone: the smallest feature size the sharp edges impose
    /// on a cube centred at `center` with half-edge `half`, unclamped. Distance
    /// is measured from the nearest point of the cube, not its centre (the
    /// `half * sqrt(3)` circumradius slack), so whether a near-rim cell refines
    /// no longer depends on octree grid phase and mirror-image rims refine
    /// evenly — matching how the proximity and curvature box fields inflate
    /// their query by `half`.
    fn crease(&self, center: &Coordinate<D>, half: Scalar) -> Quantity<Length> {
        let Some(bvh) = &self.crease else {
            return self.maximum;
        };
        let p: [Scalar; D] = from_fn(|k| center[k].value());
        let low: [Scalar; D] = from_fn(|k| p[k] - half);
        let high: [Scalar; D] = from_fn(|k| p[k] + half);
        let slack = half * (D as Scalar).sqrt();
        let maximum = self.maximum.value();
        let gradation = self.gradation;
        let mut size = maximum;
        bvh.query(&low, &high, &mut size, &mut |seg, size| {
            let reach = (point_segment_distance(&p, &seg.a, &seg.b) - slack).max(0.0);
            let candidate = match gradation {
                Some(rate) => seg.source + reach * rate,
                None if reach <= seg.source => seg.source,
                None => maximum,
            };
            if candidate < *size {
                *size = candidate;
            }
        });
        Quantity::<Length>::new(size)
    }

    /// The target element size at `point` (a degenerate zero-size cell).
    pub fn at(&self, point: &Coordinate<D>) -> Quantity<Length> {
        self.at_cell(point, 0.0)
    }

    /// The target element size for a cube centred at `center` with half-edge
    /// `half`: the crease term, further capped wherever a proximity slab
    /// reaches into the cell.
    ///
    /// `minimum` is intentionally *not* applied as a floor here anymore: a
    /// genuinely narrow real gap (crease-adjacent proximity, thin walls,
    /// tight curvature) can legitimately need elements finer than the
    /// user's chosen `--min-size`, and re-clamping the *combined* field up
    /// to that floor after every term has already picked its target just
    /// forces a hard, badly-graded size discontinuity wherever the floor
    /// bites — which is what was producing new inversions right where a
    /// correctly-fine region met the surrounding coarser field. The octree's
    /// own `--max-levels` remains the real backstop against runaway depth.
    ///
    /// TEMPORARY: gated behind STEP_DISABLE_CREASE_PROXIMITY (same flag as
    /// the crease-adjacent term) so a byte-for-byte pre-this-session
    /// baseline mesh can still be produced for comparison — remove the gate
    /// once a verdict is reached either way.
    pub fn at_cell(&self, center: &Coordinate<D>, half: Scalar) -> Quantity<Length> {
        let mut size = self.crease(center, half);
        for field in [&self.proximity, &self.curvature, &self.separation]
            .into_iter()
            .flatten()
        {
            if let Some(target) = field.target(center, half) {
                size = size.min(Quantity::<Length>::new(target));
            }
        }
        if std::env::var("STEP_DISABLE_CREASE_PROXIMITY").is_ok() {
            size.max(self.minimum).min(self.maximum)
        } else {
            size.min(self.maximum)
        }
    }
}

impl Sizing for FeatureSizing {
    fn at(&self, point: &Coordinate<D>) -> Quantity<Length> {
        FeatureSizing::at(self, point)
    }
    fn at_cell(&self, center: &Coordinate<D>, half: Scalar) -> Quantity<Length> {
        FeatureSizing::at_cell(self, center, half)
    }
}

/// A chord polyline tracing `edge` between its two vertices: the arc for a
/// circle or ellipse (the full loop when the endpoints coincide), otherwise the
/// straight segment.
fn sample_edge(brep: &Brep, edge: &Edge, samples: usize) -> Vec<Coordinate<D>> {
    let [ia, ib] = edge.vertices;
    let (a, b) = (&brep.vertices[ia], &brep.vertices[ib]);
    match &edge.curve {
        Curve::Circle(circle) => arc_polyline(
            point(&circle.center),
            axis(&circle.axis),
            axis(&circle.reference_direction),
            circle.radius,
            circle.radius,
            point(a),
            point(b),
            ia == ib,
            samples,
        ),
        Curve::Ellipse(ellipse) => arc_polyline(
            point(&ellipse.center),
            axis(&ellipse.axis),
            axis(&ellipse.reference_direction),
            ellipse.major_radius,
            ellipse.minor_radius,
            point(a),
            point(b),
            ia == ib,
            samples,
        ),
        Curve::BSpline(bspline) => bspline.segment(a, b, samples + 1),
        Curve::Line(_) => vec![a.clone(), b.clone()],
    }
}

#[expect(clippy::too_many_arguments)]
pub(in crate::geometry::cad) fn arc_polyline(
    centre: [Scalar; D],
    normal: [Scalar; D],
    reference: [Scalar; D],
    major: Scalar,
    minor: Scalar,
    start: [Scalar; D],
    end: [Scalar; D],
    closed: bool,
    samples: usize,
) -> Vec<Coordinate<D>> {
    let u = normalize(sub(reference, project(reference, normal)));
    let w = cross(normal, u);
    let angle = |p: [Scalar; D]| {
        let rel = sub(p, centre);
        (dot(rel, w) / minor.max(1.0e-12)).atan2(dot(rel, u) / major.max(1.0e-12))
    };
    let start_angle = angle(start);
    let mut sweep = angle(end) - start_angle;
    if closed || sweep.abs() < 1.0e-9 {
        sweep = TAU;
    } else if sweep < 0.0 {
        // The STEP reader flips a circle/ellipse axis so the edge always runs
        // CCW about it; take the positive turn, which may exceed half a turn.
        sweep += TAU;
    }
    (0..=samples)
        .map(|i| {
            let theta = start_angle + sweep * (i as Scalar) / (samples as Scalar);
            let (c, s) = (theta.cos(), theta.sin());
            Coordinate::from(from_fn(|k| centre[k] + major * c * u[k] + minor * s * w[k]))
        })
        .collect()
}

/// Euclidean distance from `p` to the segment `a`–`b`, in raw coordinates.
fn point_segment_distance(p: &[Scalar; D], a: &[Scalar; D], b: &[Scalar; D]) -> Scalar {
    let ab: [Scalar; D] = from_fn(|k| b[k] - a[k]);
    let ap: [Scalar; D] = from_fn(|k| p[k] - a[k]);
    let length = (0..D).map(|k| ab[k] * ab[k]).sum::<Scalar>();
    let t = if length > 0.0 {
        ((0..D).map(|k| ap[k] * ab[k]).sum::<Scalar>() / length).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (0..D)
        .map(|k| (ap[k] - t * ab[k]).powi(2))
        .sum::<Scalar>()
        .sqrt()
}

/// `(min, max)` of `values`, or `(+inf, -inf)` when empty.
fn min_max(values: &[Scalar]) -> (Scalar, Scalar) {
    values.iter().fold(
        (Scalar::INFINITY, Scalar::NEG_INFINITY),
        |(low, high), &value| (low.min(value), high.max(value)),
    )
}

/// `(start, span)` for the arc covering every angle in `angles`: the complement
/// of the widest gap between consecutive (cyclic) samples, so it begins where
/// that gap ends. A gap under an eighth of a turn is read as sampling noise and
/// the range closes to a full revolution.
fn angular_extent(mut angles: Vec<Scalar>) -> (Scalar, Scalar) {
    if angles.is_empty() {
        return (0.0, TAU);
    }
    angles.sort_by(Scalar::total_cmp);
    // Default: the widest gap is the wrap from the last sample back to the
    // first, leaving the arc `[first, last]`.
    let (mut gap, mut start) = (angles[0] + TAU - angles[angles.len() - 1], angles[0]);
    for pair in angles.windows(2) {
        if pair[1] - pair[0] > gap {
            (gap, start) = (pair[1] - pair[0], pair[1]);
        }
    }
    if gap < TAU / 8.0 {
        (0.0, TAU)
    } else {
        (start, TAU - gap)
    }
}

/// The in-plane basis `(u, w)` of an axis placement: `u` is `reference`
/// orthogonalized against `axis`, `w = axis x u`.
fn placement_basis(reference: [Scalar; D], axis: [Scalar; D]) -> ([Scalar; D], [Scalar; D]) {
    let u = normalize(sub(reference, project(reference, axis)));
    (u, cross(axis, u))
}

/// Half-thickness of a curvature band, in units of the target size: the shell
/// must be a few cells deep or octree grid phase decides whether a cell near
/// the surface lands in it, and mirror-image faces refine unevenly.
const BAND: Scalar = 2.5;

/// Lays a grid of `target`-sized tiles over the parametric rectangle
/// `[u0, u1] x [v0, v1]`, where a unit of `u`/`v` spans `u_world`/`v_world` in
/// space, and pushes one box per tile: the tile's four corners swept along the
/// surface normal by `+/- BAND * target`, so cells straddling the face overlap.
#[expect(clippy::too_many_arguments)]
fn tile_band(
    (u0, u1): (Scalar, Scalar),
    (v0, v1): (Scalar, Scalar),
    u_world: Scalar,
    v_world: Scalar,
    target: Scalar,
    point: impl Fn(Scalar, Scalar) -> [Scalar; D],
    normal: impl Fn(Scalar, Scalar) -> [Scalar; D],
    out: &mut Vec<Item<Scalar>>,
) {
    if !(target.is_finite() && target > 0.0 && u1 > u0 && v1 > v0) {
        return;
    }
    let count =
        |span: Scalar, scale: Scalar| ((span * scale / target).ceil() as usize).clamp(1, 64);
    let (nu, nv) = (count(u1 - u0, u_world), count(v1 - v0, v_world));
    let (du, dv) = ((u1 - u0) / nu as Scalar, (v1 - v0) / nv as Scalar);
    let eps = target * 1.0e-3;
    for iu in 0..nu {
        for iv in 0..nv {
            let uc = u0 + (iu as Scalar + 0.5) * du;
            let vc = v0 + (iv as Scalar + 0.5) * dv;
            let (mut low, mut high) = ([Scalar::INFINITY; D], [Scalar::NEG_INFINITY; D]);
            for su in [-0.5, 0.5] {
                for sv in [-0.5, 0.5] {
                    let (u, v) = (uc + su * du, vc + sv * dv);
                    let (p, n) = (point(u, v), normal(u, v));
                    for band in [-BAND * target, BAND * target] {
                        for k in 0..D {
                            let c = p[k] + band * n[k];
                            low[k] = low[k].min(c);
                            high[k] = high[k].max(c);
                        }
                    }
                }
            }
            out.push((
                from_fn(|k| low[k] - eps),
                from_fn(|k| high[k] + eps),
                target,
                target,
            ));
        }
    }
}

fn curvature_cylinder(
    cylinder: &Cylinder,
    points: &[[Scalar; D]],
    sections: Scalar,
    minimum: Scalar,
    maximum: Scalar,
    out: &mut Vec<Item<Scalar>>,
) {
    let target = (TAU * cylinder.radius / sections).clamp(minimum, maximum);
    if target >= maximum {
        return;
    }
    let origin = point(&cylinder.origin);
    let a = axis(&cylinder.axis);
    let (u, w) = placement_basis(axis(&cylinder.reference_direction), a);
    let (mut heights, mut angles) = (Vec::new(), Vec::new());
    for &p in points {
        let rel = sub(p, origin);
        heights.push(dot(rel, a));
        angles.push(dot(rel, w).atan2(dot(rel, u)));
    }
    let (z0, z1) = min_max(&heights);
    let (a0, span) = angular_extent(angles);
    let radius = cylinder.radius;
    tile_band(
        (a0, a0 + span),
        (z0, z1),
        radius,
        1.0,
        target,
        |t, z| {
            let (c, s) = (t.cos(), t.sin());
            from_fn(|k| origin[k] + z * a[k] + radius * (c * u[k] + s * w[k]))
        },
        |t, _| {
            let (c, s) = (t.cos(), t.sin());
            from_fn(|k| c * u[k] + s * w[k])
        },
        out,
    );
}

fn curvature_cone(
    cone: &Cone,
    points: &[[Scalar; D]],
    sections: Scalar,
    minimum: Scalar,
    maximum: Scalar,
    out: &mut Vec<Item<Scalar>>,
) {
    let origin = point(&cone.origin);
    let a = axis(&cone.axis);
    let (u, w) = placement_basis(axis(&cone.reference_direction), a);
    let (tan, sin, cos) = (
        cone.semi_angle.tan(),
        cone.semi_angle.sin(),
        cone.semi_angle.cos().abs().max(1.0e-9),
    );
    // Local cross-section radius at axial coordinate `z` (measured from
    // `origin`, where it equals `cone.radius`); the curvature radius normal to
    // the rulings is that over `cos(semi_angle)`.
    let cross_radius = |z: Scalar| (cone.radius + z * tan).max(0.0);
    let (mut heights, mut angles) = (Vec::new(), Vec::new());
    for &p in points {
        let rel = sub(p, origin);
        heights.push(dot(rel, a));
        angles.push(dot(rel, w).atan2(dot(rel, u)));
    }
    let (z0, z1) = min_max(&heights);
    let (a0, span) = angular_extent(angles);
    let curvature = cross_radius(z0).min(cross_radius(z1)) / cos;
    let target = (TAU * curvature / sections).clamp(minimum, maximum);
    if target >= maximum {
        return;
    }
    tile_band(
        (a0, a0 + span),
        (z0, z1),
        cross_radius(0.5 * (z0 + z1)).max(target),
        1.0,
        target,
        |t, z| {
            let (c, s) = (t.cos(), t.sin());
            let radius = cross_radius(z);
            from_fn(|k| origin[k] + z * a[k] + radius * (c * u[k] + s * w[k]))
        },
        |t, _| {
            let (c, s) = (t.cos(), t.sin());
            normalize(from_fn(|k| cos * (c * u[k] + s * w[k]) - sin * a[k]))
        },
        out,
    );
}

fn curvature_sphere(
    sphere: &Sphere,
    points: &[[Scalar; D]],
    sections: Scalar,
    minimum: Scalar,
    maximum: Scalar,
    out: &mut Vec<Item<Scalar>>,
) {
    let target = (TAU * sphere.radius / sections).clamp(minimum, maximum);
    if target >= maximum {
        return;
    }
    let origin = point(&sphere.origin);
    let a = axis(&sphere.axis);
    let (u, w) = placement_basis(axis(&sphere.reference_direction), a);
    let radius = sphere.radius;
    let (mut angles, mut polars) = (Vec::new(), Vec::new());
    for &p in points {
        let d = normalize(sub(p, origin));
        polars.push(dot(d, a).clamp(-1.0, 1.0).acos());
        angles.push(dot(d, w).atan2(dot(d, u)));
    }
    let (mut f0, mut f1) = min_max(&polars);
    (f0, f1) = (f0.max(0.0), f1.min(PI));
    let (a0, span) = angular_extent(angles);
    tile_band(
        (a0, a0 + span),
        (f0, f1),
        radius,
        radius,
        target,
        |t, f| {
            let (ct, st, cf, sf) = (t.cos(), t.sin(), f.cos(), f.sin());
            from_fn(|k| origin[k] + radius * (sf * (ct * u[k] + st * w[k]) + cf * a[k]))
        },
        |t, f| {
            let (ct, st, cf, sf) = (t.cos(), t.sin(), f.cos(), f.sin());
            from_fn(|k| sf * (ct * u[k] + st * w[k]) + cf * a[k])
        },
        out,
    );
}

fn curvature_torus(
    torus: &Torus,
    points: &[[Scalar; D]],
    sections: Scalar,
    minimum: Scalar,
    maximum: Scalar,
    out: &mut Vec<Item<Scalar>>,
) {
    let target = (TAU * torus.minor_radius / sections).clamp(minimum, maximum);
    if target >= maximum {
        return;
    }
    let origin = point(&torus.origin);
    let a = axis(&torus.axis);
    let (u, w) = placement_basis(axis(&torus.reference_direction), a);
    let (major, minor) = (torus.major_radius, torus.minor_radius);
    let radial = |theta: Scalar| {
        let (c, s) = (theta.cos(), theta.sin());
        from_fn::<Scalar, D, _>(|k| c * u[k] + s * w[k])
    };
    let (mut angles, mut tubes) = (Vec::new(), Vec::new());
    for &p in points {
        let rel = sub(p, origin);
        let theta = dot(rel, w).atan2(dot(rel, u));
        angles.push(theta);
        let dir = radial(theta);
        let from_centre = from_fn::<Scalar, D, _>(|k| p[k] - origin[k] - major * dir[k]);
        tubes.push(dot(from_centre, a).atan2(dot(from_centre, dir)));
    }
    let (a0, aspan) = angular_extent(angles);
    let (f0, fspan) = angular_extent(tubes);
    tile_band(
        (a0, a0 + aspan),
        (f0, f0 + fspan),
        major + minor,
        minor,
        target,
        |theta, phi| {
            let dir = radial(theta);
            let (cf, sf) = (phi.cos(), phi.sin());
            from_fn(|k| origin[k] + major * dir[k] + minor * (cf * dir[k] + sf * a[k]))
        },
        |theta, phi| {
            let dir = radial(theta);
            let (cf, sf) = (phi.cos(), phi.sin());
            from_fn(|k| cf * dir[k] + sf * a[k])
        },
        out,
    );
}

/// Proximity tiling for a cylindrical or conical face: the through-wall
/// analogue of the planar path, but parametrized by `(angle, axial)`. Each tile
/// centre asks the oracle for its local through-dimension (the shortest
/// surface-to-surface chord, a through-hole's lumen counting as infinite), and
/// a box spanning +/- that around the patch is stored with a target of
/// `thickness / cells`. This closes the azimuthal gap the planar tiling leaves
/// around a bore, without depending on which side of the face is solid.
#[expect(clippy::too_many_arguments)]
fn proximity_revolved(
    brep: &Brep,
    face: &Face,
    oracle: &BrepOracle,
    cells: Scalar,
    tile: Scalar,
    eps: Scalar,
    minimum: Scalar,
    maximum: Scalar,
    out: &mut Vec<Item<Scalar>>,
) {
    match &face.surface {
        Surface::Cylinder(c) => {
            let a = axis(&c.axis);
            let (u, w) = placement_basis(axis(&c.reference_direction), a);
            let radius = c.radius;
            proximity_ruled(
                brep,
                face,
                oracle,
                cells,
                tile,
                eps,
                minimum,
                maximum,
                point(&c.origin),
                a,
                u,
                w,
                |_z| radius,
                out,
            );
        }
        Surface::Cone(c) => {
            let a = axis(&c.axis);
            let (u, w) = placement_basis(axis(&c.reference_direction), a);
            let (tan, base) = (c.semi_angle.tan(), c.radius);
            proximity_ruled(
                brep,
                face,
                oracle,
                cells,
                tile,
                eps,
                minimum,
                maximum,
                point(&c.origin),
                a,
                u,
                w,
                |z| (base + z * tan).max(0.0),
                out,
            );
        }
        _ => {}
    }
}

/// The `(angle, axial)` tiler shared by the cylinder and cone proximity paths;
/// `radius_at(z)` is the cross-section radius at axial coordinate `z`.
#[expect(clippy::too_many_arguments)]
fn proximity_ruled(
    brep: &Brep,
    face: &Face,
    oracle: &BrepOracle,
    cells: Scalar,
    tile: Scalar,
    eps: Scalar,
    minimum: Scalar,
    maximum: Scalar,
    origin: [Scalar; D],
    a: [Scalar; D],
    u: [Scalar; D],
    w: [Scalar; D],
    radius_at: impl Fn(Scalar) -> Scalar,
    out: &mut Vec<Item<Scalar>>,
) {
    let (mut heights, mut angles) = (Vec::new(), Vec::new());
    for bound in &face.bounds {
        for half_edge in &bound.half_edges {
            let Some(edge) = brep.edges.get(half_edge.edge) else {
                continue;
            };
            for sample in sample_edge(brep, edge, 24) {
                let rel = sub(from_fn(|k| sample[k].value()), origin);
                heights.push(dot(rel, a));
                angles.push(dot(rel, w).atan2(dot(rel, u)));
            }
        }
    }
    if heights.len() < 2 {
        return;
    }
    let (z0, z1) = min_max(&heights);
    let (a0, span) = angular_extent(angles);
    if !(z1 > z0 && span > 0.0) {
        return;
    }
    let n_ang =
        (((span * radius_at(0.5 * (z0 + z1)).max(tile)) / tile).ceil() as usize).clamp(1, 48);
    let n_z = (((z1 - z0) / tile).ceil() as usize).clamp(1, 48);
    let (d_ang, d_z) = (span / n_ang as Scalar, (z1 - z0) / n_z as Scalar);
    let surface = |t: Scalar, z: Scalar| -> [Scalar; D] {
        let (co, si) = (t.cos(), t.sin());
        let r = radius_at(z);
        from_fn(|k| origin[k] + z * a[k] + r * (co * u[k] + si * w[k]))
    };
    for iu in 0..n_ang {
        for iz in 0..n_z {
            let tc = a0 + (iu as Scalar + 0.5) * d_ang;
            let zc = z0 + (iz as Scalar + 0.5) * d_z;
            let here = surface(tc, zc);
            let (co, si) = (tc.cos(), tc.sin());
            let radial: [Scalar; D] = from_fn(|k| co * u[k] + si * w[k]);
            // Querying exactly on the face grazes the rim edge and the ray hit
            // test is unreliable there, so probe just off it to each side and
            // take the shorter chord: outward finds a bore's wall, inward a
            // solid boss's diameter, and the empty side returns infinity.
            let off = (0.1 * radius_at(zc)).max(4.0 * eps);
            let thickness = [-off, off]
                .into_iter()
                .map(|d| {
                    oracle.local_diameter(&Coordinate::from(from_fn::<Scalar, D, _>(|k| {
                        here[k] + d * radial[k]
                    })))
                })
                .fold(Scalar::INFINITY, Scalar::min);
            if !(thickness.is_finite() && thickness > 0.0) {
                continue;
            }
            let target = (thickness / cells).max(minimum).min(maximum);
            if target >= maximum {
                continue;
            }
            // A shell +/- one thickness around the patch, so a cell anywhere in
            // the wall overlaps it whichever side of the face is solid.
            let (mut low, mut high) = ([Scalar::INFINITY; D], [Scalar::NEG_INFINITY; D]);
            for du in [-0.5, 0.5] {
                for dv in [-0.5, 0.5] {
                    let corner = surface(tc + du * d_ang, zc + dv * d_z);
                    for depth in [-thickness, thickness] {
                        for k in 0..D {
                            let c = corner[k] + depth * radial[k];
                            low[k] = low[k].min(c);
                            high[k] = high[k].max(c);
                        }
                    }
                }
            }
            out.push((
                from_fn(|k| low[k] - eps),
                from_fn(|k| high[k] + eps),
                target,
                target,
            ));
        }
    }
}

fn point(coordinate: &Coordinate<D>) -> [Scalar; D] {
    from_fn(|k| coordinate[k].value())
}

/// Whether surface runs between two creases the whole way along the stretch
/// they face each other over, so that the two of them bound one narrow ribbon
/// of surface rather than straddling open space. Each sample walks a point
/// along one crease, drops to the nearest point of the other, and asks whether
/// the surface passes through the middle: a flat ribbon passes through it
/// exactly and a curved one misses by only its sagitta, whereas across an open
/// gap the nearest surface is one of the creases' own, half a gap away. Both
/// creases are walked, since one may face only part of the other.
fn joined(
    one: &([Scalar; D], [Scalar; D]),
    two: &([Scalar; D], [Scalar; D]),
    oracle: &BrepOracle,
) -> bool {
    [(one, two), (two, one)].into_iter().all(|(from, to)| {
        (0..JOINED_SAMPLES).all(|sample| {
            let fraction = sample as Scalar / (JOINED_SAMPLES - 1) as Scalar;
            let here: [Scalar; D] = from_fn(|k| from.0[k] + fraction * (from.1[k] - from.0[k]));
            let there = closest_on(to, &here);
            let gap = dot(sub(there, here), sub(there, here)).sqrt();
            if gap <= 0.0 {
                return true;
            }
            let middle =
                Coordinate::<D>::from(from_fn::<Scalar, D, _>(|k| 0.5 * (here[k] + there[k])));
            oracle.distance(&middle) <= gap * JOINED_FRACTION
        })
    })
}

/// The point of segment `.0`–`.1` closest to `p`, in raw coordinates.
fn closest_on(segment: &([Scalar; D], [Scalar; D]), p: &[Scalar; D]) -> [Scalar; D] {
    let along = sub(segment.1, segment.0);
    let length = dot(along, along);
    if length <= 0.0 {
        return segment.0;
    }
    let t = (dot(sub(*p, segment.0), along) / length).clamp(0.0, 1.0);
    from_fn(|k| segment.0[k] + t * along[k])
}

/// Closest-point distance between two segments (Ericson, *Real-Time Collision
/// Detection*, section 5.1.9).
fn segment_distance(one: &([Scalar; D], [Scalar; D]), two: &([Scalar; D], [Scalar; D])) -> Scalar {
    let (p1, q1, p2, q2) = (one.0, one.1, two.0, two.1);
    let d1 = sub(q1, p1);
    let d2 = sub(q2, p2);
    let r = sub(p1, p2);
    let a = dot(d1, d1);
    let e = dot(d2, d2);
    let f = dot(d2, r);
    const EPSILON: Scalar = 1.0e-12;
    let (s, t) = if a <= EPSILON && e <= EPSILON {
        (0.0, 0.0)
    } else if a <= EPSILON {
        (0.0, (f / e).clamp(0.0, 1.0))
    } else {
        let c = dot(d1, r);
        if e <= EPSILON {
            ((-c / a).clamp(0.0, 1.0), 0.0)
        } else {
            let b = dot(d1, d2);
            let denominator = a * e - b * b;
            let s = if denominator.abs() > EPSILON {
                ((b * f - c * e) / denominator).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let t = (b * s + f) / e;
            if t < 0.0 {
                ((-c / a).clamp(0.0, 1.0), 0.0)
            } else if t > 1.0 {
                (((b - c) / a).clamp(0.0, 1.0), 1.0)
            } else {
                (s, t)
            }
        }
    };
    let closest_one: [Scalar; D] = from_fn(|k| p1[k] + d1[k] * s);
    let closest_two: [Scalar; D] = from_fn(|k| p2[k] + d2[k] * t);
    dot(
        sub(closest_one, closest_two),
        sub(closest_one, closest_two),
    )
    .sqrt()
}

fn axis(direction: &crate::geometry::Direction<D>) -> [Scalar; D] {
    from_fn(|k| direction[k].value())
}

fn sub(a: [Scalar; D], b: [Scalar; D]) -> [Scalar; D] {
    from_fn(|k| a[k] - b[k])
}

fn dot(a: [Scalar; D], b: [Scalar; D]) -> Scalar {
    (0..D).map(|k| a[k] * b[k]).sum()
}

fn cross(a: [Scalar; D], b: [Scalar; D]) -> [Scalar; D] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn project(v: [Scalar; D], onto: [Scalar; D]) -> [Scalar; D] {
    let scale = dot(v, onto) / dot(onto, onto).max(1.0e-30);
    from_fn(|k| scale * onto[k])
}

fn normalize(v: [Scalar; D]) -> [Scalar; D] {
    let norm = dot(v, v).sqrt();
    if norm > 1.0e-30 {
        v.map(|x| x / norm)
    } else {
        [1.0, 0.0, 0.0]
    }
}
