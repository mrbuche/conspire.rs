#[cfg(test)]
mod test;

mod patch;

use super::{
    Brep, D, Face, Loop,
    curve::{self, Curve},
    surface::{self, Surface},
};
use crate::{
    geometry::{Coordinate, Direction, solid::SolidOracle},
    math::Scalar,
};
use patch::{Curved, FacePatch};
use std::array::from_fn;

/// [`SolidOracle`] backed by the analytic B-rep: closest-point projection onto
/// each face's exact surface. Planar faces are trimmed to their loops (polygon
/// or exact disk/annulus, possibly mixed with circular-arc edges). Cylindrical
/// and conical faces are trimmed exactly from their bounding edges — an axial
/// ruling or a circular arc ⊥ the axis collapses to a straight segment in the
/// surface's own (angle, axial distance) chart, so a genuine partial sweep
/// (fillet, chamfer remnant) is meshable; on a cylinder, a tilted elliptical
/// edge (an oblique planar cut) is trimmed exactly too, via the sinusoid it
/// traces in that chart. A tilted edge on a cone, or a free-form edge on
/// either, still errs. Spherical and toroidal faces are taken whole. A
/// B-spline face errs.
///
/// [`signed_distance`](Self::signed_distance)'s sign is a ray-parity test
/// against these trimmed faces (OCCT's `BRepClass3d_SolidClassifier`
/// approach); the magnitude is the nearest trimmed face's distance.
pub struct BrepOracle {
    patches: Vec<FacePatch>,
    /// One axis-aligned box per patch, the ray/point broad-phase: a ray or
    /// query that misses a patch's box skips its narrow-phase entirely.
    boxes: Vec<([Scalar; D], [Scalar; D])>,
}

impl Brep {
    /// An analytic [`SolidOracle`] projecting onto this solid's surface, for
    /// fitting a background mesh.
    pub fn oracle(&self) -> Result<BrepOracle, &'static str> {
        if self.faces.is_empty() {
            return Err("brep has no faces");
        }
        let patches = self
            .faces
            .iter()
            .map(|face| self.face_patch(face))
            .collect::<Result<Vec<_>, _>>()?;
        let boxes = patches
            .iter()
            .map(|patch| {
                let (mut low, mut high) = patch.bounds();
                // A hair of slack so a ray grazing a box face is not rejected.
                let pad = (0..D)
                    .map(|k| high[k] - low[k])
                    .fold(0.0, Scalar::max)
                    .max(1.0)
                    * 1.0e-9;
                low = from_fn(|k| low[k] - pad);
                high = from_fn(|k| high[k] + pad);
                (low, high)
            })
            .collect();
        Ok(BrepOracle { patches, boxes })
    }

    fn face_patch(&self, face: &Face) -> Result<FacePatch, &'static str> {
        match &face.surface {
            Surface::Plane(_) => Ok(FacePatch::Planar(self.planar_face(face)?)),
            Surface::Cylinder(surface) => self.cylinder_patch(surface, face),
            Surface::Cone(surface) => self.cone_patch(surface, face),
            Surface::Sphere(surface) => Ok(sphere_patch(surface, face.forward)),
            Surface::Torus(surface) => Ok(torus_patch(surface, face.forward)),
            Surface::BSpline(_) => Err("B-spline faces are not yet meshable"),
        }
    }

    /// World-space corners of every vertex on `face`'s loops, poles included.
    fn face_vertices(&self, face: &Face) -> Result<Vec<[Scalar; D]>, &'static str> {
        let mut points = Vec::new();
        for bound in &face.bounds {
            for vertex in bound.vertices(&self.edges)? {
                points.push(from_fn(|k| self.vertices[vertex][k].value()));
            }
        }
        for &pole in &face.poles {
            points.push(from_fn(|k| self.vertices[pole][k].value()));
        }
        Ok(points)
    }

    /// The `(u, v)` boundary of one bound in `origin`/`axis`'s chart —
    /// `u` the angle around `axis`, unwrapped continuously past a single turn
    /// as the loop is walked, `v` the signed axial distance — or `None` if the
    /// bound has no angular restriction (a coincident-endpoint seam circle,
    /// the whole-turn case). Each ring point carries the shape of the edge
    /// leading to the next one: `None` for straight, `Some` for the sinusoid
    /// an oblique planar cut traces on a cylinder (`cylinder_radius: Some`) —
    /// unsupported on a cone (`None`).
    fn uv_ring(
        &self,
        bound: &Loop,
        origin: [Scalar; D],
        axis: [Scalar; D],
        cylinder_radius: Option<Scalar>,
    ) -> Result<Option<Ring>, &'static str> {
        let mut ring: Ring = Vec::new();
        let mut cursor: Option<[Scalar; 2]> = None;
        for half_edge in &bound.half_edges {
            let edge = self
                .edges
                .get(half_edge.edge)
                .ok_or("half-edge references a missing edge")?;
            let (start, end) = if half_edge.forward {
                (edge.vertices[0], edge.vertices[1])
            } else {
                (edge.vertices[1], edge.vertices[0])
            };
            let current = match cursor {
                Some(point) => point,
                None => {
                    let start_point: [Scalar; D] = from_fn(|k| self.vertices[start][k].value());
                    to_uv(origin, axis, start_point)
                }
            };
            let end_point: [Scalar; D] = from_fn(|k| self.vertices[end][k].value());
            let (kind, next) = match &edge.curve {
                Curve::Line(_) => {
                    let [u_end, v_end] = to_uv(origin, axis, end_point);
                    if wrap(u_end - current[0]).abs() > 1.0e-6 {
                        return Err("non-axial straight edge on a curved face");
                    }
                    (None, [current[0], v_end])
                }
                Curve::Circle(circle) => {
                    let caxis: [Scalar; D] = from_fn(|k| circle.axis[k].value());
                    let alignment = dot(caxis, axis);
                    if alignment.abs() < 1.0 - 1.0e-6 {
                        return Err("tilted circular edge on a curved face");
                    }
                    if start == end {
                        return Ok(None);
                    }
                    let sign = if half_edge.forward { alignment } else { -alignment };
                    let [u_end, v_end] = to_uv(origin, axis, end_point);
                    let mut delta = wrap(u_end - current[0]);
                    if sign > 0.0 && delta < 0.0 {
                        delta += std::f64::consts::TAU;
                    } else if sign < 0.0 && delta > 0.0 {
                        delta -= std::f64::consts::TAU;
                    }
                    (None, [current[0] + delta, v_end])
                }
                Curve::Ellipse(ellipse) => {
                    let Some(radius) = cylinder_radius else {
                        return Err("tilted elliptical edge on a conical face is not yet supported");
                    };
                    let sinusoid = ellipse_sinusoid(ellipse, origin, axis, radius)?;
                    let [u_end, v_end] = to_uv(origin, axis, end_point);
                    let u_end = current[0] + wrap(u_end - current[0]);
                    (Some(sinusoid), [u_end, v_end])
                }
                _ => return Err("unsupported edge on a curved face trim"),
            };
            ring.push((current, kind));
            cursor = Some(next);
        }
        Ok(Some(ring))
    }

    /// Every bound of `face` as a `(u, v)` polygon, or `None` if any bound
    /// sweeps the whole turn unrestricted.
    fn trim_rings(
        &self,
        face: &Face,
        origin: [Scalar; D],
        axis: [Scalar; D],
        cylinder_radius: Option<Scalar>,
    ) -> Result<Option<Vec<Ring>>, &'static str> {
        let mut rings = Vec::new();
        for bound in &face.bounds {
            match self.uv_ring(bound, origin, axis, cylinder_radius)? {
                Some(ring) => rings.push(ring),
                None => return Ok(None),
            }
        }
        Ok(Some(rings))
    }

    fn cylinder_patch(
        &self,
        surface: &surface::Cylinder,
        face: &Face,
    ) -> Result<FacePatch, &'static str> {
        let axis: [Scalar; D] = from_fn(|k| surface.axis[k].value());
        let origin: [Scalar; D] = from_fn(|k| surface.origin[k].value());
        let vertices = self.face_vertices(face)?;
        let (low, high) = axial_span(&vertices, origin, axis)?;
        let radius = surface.radius;
        let rings = self.trim_rings(face, origin, axis, Some(radius))?;
        let curved = Curved::Cylinder {
            origin,
            axis,
            radius,
            low,
            high,
            rings,
            sign: orientation(face.forward),
        };
        let base: [Scalar; D] = from_fn(|k| origin[k] + low * axis[k]);
        let (bl, bh) = frustum_bounds(base, axis, radius, radius, high - low);
        Ok(FacePatch::Curved { curved, low: bl, high: bh })
    }

    fn cone_patch(&self, surface: &surface::Cone, face: &Face) -> Result<FacePatch, &'static str> {
        let axis: [Scalar; D] = from_fn(|k| surface.axis[k].value());
        let origin: [Scalar; D] = from_fn(|k| surface.origin[k].value());
        let vertices = self.face_vertices(face)?;
        let (low, high) = axial_span(&vertices, origin, axis)?;
        let slope = surface.semi_angle.tan();
        let base_radius = (surface.radius + low * slope).max(0.0);
        let tip_radius = (surface.radius + high * slope).max(0.0);
        let rings = self.trim_rings(face, origin, axis, None)?;
        let curved = Curved::Cone {
            origin,
            axis,
            radius: surface.radius,
            slope,
            low,
            high,
            rings,
            sign: orientation(face.forward),
        };
        let base: [Scalar; D] = from_fn(|k| origin[k] + low * axis[k]);
        let (bl, bh) = frustum_bounds(base, axis, base_radius, tip_radius, high - low);
        Ok(FacePatch::Curved { curved, low: bl, high: bh })
    }
}

fn sphere_patch(surface: &surface::Sphere, forward: bool) -> FacePatch {
    let centre: [Scalar; D] = from_fn(|k| surface.origin[k].value());
    let radius = surface.radius;
    FacePatch::Curved {
        curved: Curved::Sphere {
            centre,
            radius,
            sign: orientation(forward),
        },
        low: from_fn(|k| centre[k] - radius),
        high: from_fn(|k| centre[k] + radius),
    }
}

fn torus_patch(surface: &surface::Torus, forward: bool) -> FacePatch {
    let centre: [Scalar; D] = from_fn(|k| surface.origin[k].value());
    let axis: [Scalar; D] = from_fn(|k| surface.axis[k].value());
    let (major, minor) = (surface.major_radius, surface.minor_radius);
    let reach: [Scalar; D] =
        from_fn(|k| (major + minor) * (1.0 - axis[k] * axis[k]).max(0.0).sqrt() + minor * axis[k].abs());
    FacePatch::Curved {
        curved: Curved::Torus {
            centre,
            axis,
            major,
            minor,
            sign: orientation(forward),
        },
        low: from_fn(|k| centre[k] - reach[k]),
        high: from_fn(|k| centre[k] + reach[k]),
    }
}

impl BrepOracle {
    /// `(low, high)` world corners enclosing every face.
    pub fn bounds(&self) -> (Coordinate<D>, Coordinate<D>) {
        let mut low = [Scalar::INFINITY; D];
        let mut high = [Scalar::NEG_INFINITY; D];
        for patch in &self.patches {
            let (patch_low, patch_high) = patch.bounds();
            for k in 0..D {
                low[k] = low[k].min(patch_low[k]);
                high[k] = high[k].max(patch_high[k]);
            }
        }
        (low.into(), high.into())
    }

    fn nearest(&self, query: &Coordinate<D>) -> Option<(Coordinate<D>, Direction<D>, Scalar)> {
        let point: [Scalar; D] = from_fn(|k| query[k].value());
        let mut best: Option<(Coordinate<D>, Direction<D>, Scalar)> = None;
        for (patch, boxed) in self.patches.iter().zip(&self.boxes) {
            // Skip a patch whose box is already farther than the best hit.
            if best
                .as_ref()
                .is_some_and(|(_, _, d)| point_box_distance(point, boxed) >= *d)
            {
                continue;
            }
            let candidate = patch.closest(query);
            if best.as_ref().is_none_or(|(_, _, d)| candidate.2 < *d) {
                best = Some(candidate);
            }
        }
        best
    }

    /// The patches whose box the ray `origin + t·direction` (`t ≥ 0`) meets.
    fn ray_candidates(
        &self,
        origin: [Scalar; D],
        direction: [Scalar; D],
    ) -> impl Iterator<Item = &FacePatch> {
        self.patches
            .iter()
            .zip(&self.boxes)
            .filter(move |(_, boxed)| ray_hits_box(origin, direction, boxed))
            .map(|(patch, _)| patch)
    }

    /// Unsigned distance from `query` to the nearest trimmed face.
    pub fn distance(&self, query: &Coordinate<D>) -> Scalar {
        self.nearest(query)
            .map_or(Scalar::INFINITY, |(_, _, distance)| distance)
    }

    /// The local through-dimension at `query`: the shortest surface-to-surface
    /// chord through it, searched along the coordinate axes and the nearest
    /// face's normal. This is the thickness of a wall or the width of a cavity
    /// the query sits in — the quantity proximity sizing wants, unlike
    /// [`distance`](Self::distance), which is small next to *any* surface and
    /// would over-refine the skin of a thick body. A chord that escapes one
    /// side without a hit (an open recess, a through hole along its axis)
    /// counts as infinite, so an open direction never drives refinement.
    pub fn local_diameter(&self, query: &Coordinate<D>) -> Scalar {
        let origin: [Scalar; D] = from_fn(|k| query[k].value());
        let graze = self.distance(query).max(1.0e-9) * 1.0e-3;
        let nearest_along = |direction: [Scalar; D]| {
            self.ray_candidates(origin, direction)
                .flat_map(|patch| patch.ray_hits(origin, direction))
                .filter(|&t| t > graze)
                .fold(Scalar::INFINITY, Scalar::min)
        };
        let mut directions: Vec<[Scalar; D]> = (0..D)
            .map(|axis| from_fn(|k| if k == axis { 1.0 } else { 0.0 }))
            .collect();
        if let Some((_, normal, _)) = self.nearest(query) {
            directions.push(from_fn(|k| normal[k].value()));
        }
        directions
            .into_iter()
            .map(|direction| {
                nearest_along(direction) + nearest_along(from_fn(|k| -direction[k]))
            })
            .fold(Scalar::INFINITY, Scalar::min)
    }

    /// Every patch's `(surface type, distance, closest point, outward normal)`
    /// for `query`, nearest first — a probe for why a query picks the face it
    /// does.
    #[cfg(test)]
    pub(crate) fn patch_report(
        &self,
        query: &Coordinate<D>,
    ) -> Vec<(&'static str, Scalar, [Scalar; D], [Scalar; D])> {
        let mut rows: Vec<_> = self
            .patches
            .iter()
            .map(|patch| {
                let kind = match patch {
                    FacePatch::Planar(_) => "plane",
                    FacePatch::Curved { curved, .. } => match curved {
                        Curved::Cylinder { .. } => "cyl",
                        Curved::Cone { .. } => "cone",
                        Curved::Sphere { .. } => "sphere",
                        Curved::Torus { .. } => "torus",
                    },
                };
                let (point, normal, distance) = patch.closest(query);
                (
                    kind,
                    distance,
                    from_fn(|k| point[k].value()),
                    from_fn(|k| normal[k].value()),
                )
            })
            .collect();
        rows.sort_by(|a, b| a.1.total_cmp(&b.1));
        rows
    }
}

/// Three fixed ray directions with pairwise-irrational-ish components, so a
/// ray is unlikely to graze an edge or lie in a face for all three at once.
const RAY_DIRECTIONS: [[Scalar; D]; 3] = [
    [0.862_667, 0.411_988, 0.291_536],
    [0.301_511, 0.904_534, 0.301_511],
    [0.334_412, 0.243_975, 0.910_367],
];

impl BrepOracle {
    /// Whether `query` is inside the solid, by ray parity against the exact
    /// trimmed faces (OCCT's `BRepClass3d_SolidClassifier` approach): count the
    /// crossings of a ray from `query`; odd is inside. A ray that grazes an
    /// edge — two hits at the same parameter — is discarded and the next
    /// direction tried.
    fn encloses(&self, query: &Coordinate<D>) -> bool {
        let origin: [Scalar; D] = from_fn(|k| query[k].value());
        let (low, high) = self.bounds();
        let graze = (0..D)
            .map(|k| high[k].value() - low[k].value())
            .fold(0.0, Scalar::max)
            * 1.0e-7;
        let mut parity = false;
        for direction in RAY_DIRECTIONS {
            let mut hits: Vec<Scalar> = self
                .ray_candidates(origin, direction)
                .flat_map(|patch| patch.ray_hits(origin, direction))
                .collect();
            hits.sort_by(Scalar::total_cmp);
            if hits.first().is_some_and(|&t| t < graze) {
                return true; // on the surface
            }
            let mut crossings = 0usize;
            let mut ambiguous = false;
            let mut previous = Scalar::NEG_INFINITY;
            for &t in &hits {
                if t - previous < graze {
                    ambiguous = true; // grazed a shared edge or vertex
                } else {
                    crossings += 1;
                    previous = t;
                }
            }
            parity = crossings % 2 == 1;
            if !ambiguous {
                return parity;
            }
        }
        parity
    }
}

impl SolidOracle for BrepOracle {
    fn project(&self, query: &Coordinate<D>) -> Option<(Coordinate<D>, Direction<D>)> {
        self.nearest(query).map(|(point, normal, _)| (point, normal))
    }

    /// Magnitude is the distance to the nearest trimmed face; the sign is the
    /// ray parity of the exact trimmed boundary (positive inside), which stays
    /// right at a void's medial axis where a nearest-face normal cannot.
    fn signed_distance(&self, query: &Coordinate<D>) -> Scalar {
        let Some((_, _, distance)) = self.nearest(query) else {
            return Scalar::NEG_INFINITY;
        };
        if self.encloses(query) {
            distance
        } else {
            -distance
        }
    }
}

fn orientation(forward: bool) -> Scalar {
    if forward { 1.0 } else { -1.0 }
}

/// Whether the ray `origin + t·direction`, `t ≥ 0`, meets the box `[low, high]`
/// (a slab test; `direction` need not be unit).
fn ray_hits_box(
    origin: [Scalar; D],
    direction: [Scalar; D],
    (low, high): &([Scalar; D], [Scalar; D]),
) -> bool {
    let mut enter = Scalar::NEG_INFINITY;
    let mut exit = Scalar::INFINITY;
    for k in 0..D {
        if direction[k].abs() < 1.0e-30 {
            if origin[k] < low[k] || origin[k] > high[k] {
                return false;
            }
        } else {
            let inverse = 1.0 / direction[k];
            let mut near = (low[k] - origin[k]) * inverse;
            let mut far = (high[k] - origin[k]) * inverse;
            if near > far {
                std::mem::swap(&mut near, &mut far);
            }
            enter = enter.max(near);
            exit = exit.min(far);
            if enter > exit {
                return false;
            }
        }
    }
    exit >= 0.0
}

/// Distance from `point` to the box `[low, high]` (zero when inside).
fn point_box_distance(point: [Scalar; D], (low, high): &([Scalar; D], [Scalar; D])) -> Scalar {
    (0..D)
        .map(|k| (low[k] - point[k]).max(point[k] - high[k]).max(0.0).powi(2))
        .sum::<Scalar>()
        .sqrt()
}

/// The `[low, high]` span of `points` projected onto `axis` from `origin`.
/// Errs rather than inventing a span when the face has no usable extent — a
/// degenerate span has no honest analytic patch.
fn axial_span(
    points: &[[Scalar; D]],
    origin: [Scalar; D],
    axis: [Scalar; D],
) -> Result<(Scalar, Scalar), &'static str> {
    let mut low = Scalar::INFINITY;
    let mut high = Scalar::NEG_INFINITY;
    for point in points {
        let along = (0..D).map(|k| (point[k] - origin[k]) * axis[k]).sum::<Scalar>();
        low = low.min(along);
        high = high.max(along);
    }
    if !(low.is_finite() && high.is_finite()) {
        return Err("cylindrical/conical face has no vertices to bound its axial extent");
    }
    if high - low < 1.0e-9 {
        return Err("cylindrical/conical face has a degenerate (zero-height) axial extent");
    }
    Ok((low, high))
}

/// AABB of a frustum: the union of its two end circles, each an exact
/// axis-aligned box of a circle with the given radius, centre and `axis`.
fn frustum_bounds(
    base: [Scalar; D],
    axis: [Scalar; D],
    base_radius: Scalar,
    tip_radius: Scalar,
    height: Scalar,
) -> ([Scalar; D], [Scalar; D]) {
    let tip: [Scalar; D] = from_fn(|k| base[k] + height * axis[k]);
    let mut low = [Scalar::INFINITY; D];
    let mut high = [Scalar::NEG_INFINITY; D];
    for (centre, radius) in [(base, base_radius), (tip, tip_radius)] {
        for k in 0..D {
            let extent = radius * (1.0 - axis[k] * axis[k]).max(0.0).sqrt();
            low[k] = low[k].min(centre[k] - extent);
            high[k] = high[k].max(centre[k] + extent);
        }
    }
    (low, high)
}

fn dot(a: [Scalar; D], b: [Scalar; D]) -> Scalar {
    (0..D).map(|k| a[k] * b[k]).sum()
}

/// An orthonormal pair spanning the plane perpendicular to `axis`.
fn basis(axis: [Scalar; D]) -> ([Scalar; D], [Scalar; D]) {
    let seed = if axis[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let d = dot(seed, axis);
    let mut u: [Scalar; D] = from_fn(|k| seed[k] - d * axis[k]);
    let norm = dot(u, u).sqrt().max(1.0e-30);
    u = u.map(|x| x / norm);
    let v = [
        axis[1] * u[2] - axis[2] * u[1],
        axis[2] * u[0] - axis[0] * u[2],
        axis[0] * u[1] - axis[1] * u[0],
    ];
    (u, v)
}

/// `(u, v)` of `point` in `origin`/`axis`'s own chart: `u` the angle around
/// `axis` from an arbitrary but fixed in-plane reference, `v` the signed axial
/// distance from `origin`.
fn to_uv(origin: [Scalar; D], axis: [Scalar; D], point: [Scalar; D]) -> [Scalar; 2] {
    let (u_hat, v_hat) = basis(axis);
    let rel: [Scalar; D] = from_fn(|k| point[k] - origin[k]);
    let v = dot(rel, axis);
    let radial: [Scalar; D] = from_fn(|k| rel[k] - v * axis[k]);
    [dot(radial, v_hat).atan2(dot(radial, u_hat)), v]
}

/// The unit radial direction at angle `u` around `axis`.
fn uv_direction(axis: [Scalar; D], u: Scalar) -> [Scalar; D] {
    let (u_hat, v_hat) = basis(axis);
    from_fn(|k| u.cos() * u_hat[k] + u.sin() * v_hat[k])
}

/// `delta` reduced into `(-pi, pi]`.
fn wrap(delta: Scalar) -> Scalar {
    let mut d = delta % std::f64::consts::TAU;
    if d > std::f64::consts::PI {
        d -= std::f64::consts::TAU;
    } else if d <= -std::f64::consts::PI {
        d += std::f64::consts::TAU;
    }
    d
}

/// A `(u, v)` trim-ring point paired with the shape of the edge leading to
/// the next one: `None` for straight, `Some` for a sinusoid cut.
pub(super) type Ring = Vec<([Scalar; 2], Option<Sinusoid>)>;

/// The exact trace `v(u) = k + a*cos(u - phi)` an oblique plane leaves on a
/// cylinder's `(u, v)` chart.
#[derive(Clone, Copy)]
pub(super) struct Sinusoid {
    k: Scalar,
    a: Scalar,
    phi: Scalar,
}

impl Sinusoid {
    fn v(&self, u: Scalar) -> Scalar {
        self.k + self.a * (u - self.phi).cos()
    }
}

/// The plane of `ellipse` meets the cylinder (`origin`/`axis`/`radius`) along
/// `v(u) = k + a*cos(u - phi)`: substitute the cylinder's own parametrization
/// into the plane equation `n.(p - centre) = 0` and solve for `v`, linear in
/// `cos(u)` and `sin(u)`.
fn ellipse_sinusoid(
    ellipse: &curve::Ellipse,
    origin: [Scalar; D],
    axis: [Scalar; D],
    radius: Scalar,
) -> Result<Sinusoid, &'static str> {
    let (u_hat, v_hat) = basis(axis);
    let normal: [Scalar; D] = from_fn(|k| ellipse.axis[k].value());
    let centre: [Scalar; D] = from_fn(|k| ellipse.center[k].value());
    let n_axis = dot(normal, axis);
    if n_axis.abs() < 1.0e-9 {
        return Err("elliptical edge's plane is parallel to the cylinder axis");
    }
    let (nu, nv) = (dot(normal, u_hat), dot(normal, v_hat));
    let k = dot(normal, from_fn(|i| centre[i] - origin[i])) / n_axis;
    let (au, av) = (-radius * nu / n_axis, -radius * nv / n_axis);
    Ok(Sinusoid { k, a: au.hypot(av), phi: av.atan2(au) })
}

/// Whether `uv` lies inside `rings`, trying both neighbouring turns since `u`
/// is periodic and the rings may be unwrapped past a single turn.
fn periodic_contains(uv: [Scalar; 2], rings: &[Ring]) -> bool {
    [0.0, std::f64::consts::TAU, -std::f64::consts::TAU]
        .into_iter()
        .any(|shift| ring_contains([uv[0] + shift, uv[1]], rings))
}

/// Even-odd ray-crossing test against `rings` (line segments as usual; a
/// sinusoid edge solves `v(u) = py` for up to two candidate `u`s, `cos` being
/// two-to-one, each checked against the edge's own span) — a sinusoid can
/// cross the ray twice even with both endpoints on one side.
fn ring_contains([px, py]: [Scalar; 2], rings: &[Ring]) -> bool {
    let mut inside = false;
    for ring in rings {
        let count = ring.len();
        for i in 0..count {
            let (a, kind) = ring[i];
            let (b, _) = ring[(i + 1) % count];
            match kind {
                None => {
                    let [ax, ay] = a;
                    let [bx, by] = b;
                    if (ay > py) != (by > py) {
                        let crossing = ax + (py - ay) / (by - ay) * (bx - ax);
                        if px < crossing {
                            inside = !inside;
                        }
                    }
                }
                Some(sinusoid) => {
                    let target = (py - sinusoid.k) / sinusoid.a;
                    if target.abs() > 1.0 {
                        continue;
                    }
                    let (lo, hi) = (a[0].min(b[0]), a[0].max(b[0]));
                    let offset = target.acos();
                    for candidate in [sinusoid.phi + offset, sinusoid.phi - offset] {
                        let mid = (lo + hi) / 2.0;
                        let u = candidate + ((mid - candidate) / std::f64::consts::TAU).round() * std::f64::consts::TAU;
                        if u >= lo - 1.0e-9 && u <= hi + 1.0e-9 && px < u {
                            inside = !inside;
                        }
                    }
                }
            }
        }
    }
    inside
}

/// The point of `rings` nearest `uv`, in an arc-length metric (`weight(v)` is
/// the local radius) so the snap distance is physical, not raw angle-vs-length.
/// Periodic in `u` like [`periodic_contains`].
fn periodic_nearest(
    uv: [Scalar; 2],
    rings: &[Ring],
    weight: impl Fn(Scalar) -> Scalar,
) -> [Scalar; 2] {
    let mut best = uv;
    let mut best_distance = Scalar::INFINITY;
    for shift in [0.0, std::f64::consts::TAU, -std::f64::consts::TAU] {
        let query = [uv[0] + shift, uv[1]];
        for ring in rings {
            let count = ring.len();
            for i in 0..count {
                let (a, kind) = ring[i];
                let (b, _) = ring[(i + 1) % count];
                let candidate = match kind {
                    None => {
                        let wq = [weight(query[1]) * query[0], query[1]];
                        let (wa, wb) = ([weight(a[1]) * a[0], a[1]], [weight(b[1]) * b[0], b[1]]);
                        let (ex, ey) = (wb[0] - wa[0], wb[1] - wa[1]);
                        let span = ex * ex + ey * ey;
                        let t = if span > 0.0 {
                            (((wq[0] - wa[0]) * ex + (wq[1] - wa[1]) * ey) / span).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        let w = [wa[0] + t * ex, wa[1] + t * ey];
                        [w[0] / weight(w[1]).max(1.0e-12), w[1]]
                    }
                    Some(sinusoid) => nearest_on_sinusoid(query, a[0], b[0], &sinusoid),
                };
                let distance = (candidate[0] - query[0]).powi(2) + (candidate[1] - query[1]).powi(2);
                if distance < best_distance {
                    best_distance = distance;
                    best = [candidate[0] - shift, candidate[1]];
                }
            }
        }
    }
    best
}

/// The point on `v = sinusoid(u)`, `u` between `u_a` and `u_b`, nearest `uv`:
/// bisects for an interior critical point of the squared distance and takes
/// the best of that and the two endpoints — no closed form exists for the
/// nearest point on a general sinusoid, so this is the same bisection-to-an-
/// exact-root approach as [`crate::geometry::csg::Ellipsoid`]'s oracle, not a
/// sampled approximation of the curve itself.
fn nearest_on_sinusoid(uv: [Scalar; 2], u_a: Scalar, u_b: Scalar, sinusoid: &Sinusoid) -> [Scalar; 2] {
    let (lo, hi) = (u_a.min(u_b), u_a.max(u_b));
    let derivative = |u: Scalar| (u - uv[0]) - sinusoid.a * (u - sinusoid.phi).sin() * (sinusoid.v(u) - uv[1]);
    let distance = |u: Scalar| (u - uv[0]).powi(2) + (sinusoid.v(u) - uv[1]).powi(2);
    let mut candidates = vec![lo, hi];
    let (flo, fhi) = (derivative(lo), derivative(hi));
    if flo * fhi <= 0.0 {
        let (mut a, mut fa, mut b) = (lo, flo, hi);
        for _ in 0..60 {
            let mid = 0.5 * (a + b);
            let fmid = derivative(mid);
            if fmid == 0.0 {
                a = mid;
                b = mid;
                break;
            }
            if (fmid > 0.0) == (fa > 0.0) {
                a = mid;
                fa = fmid;
            } else {
                b = mid;
            }
        }
        candidates.push(0.5 * (a + b));
    }
    let best_u = candidates
        .into_iter()
        .min_by(|&x, &y| distance(x).total_cmp(&distance(y)))
        .unwrap();
    [best_u, sinusoid.v(best_u)]
}
