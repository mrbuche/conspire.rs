#[cfg(test)]
mod test;

use super::{Brep, D, Face, curve::Curve, surface::Surface};
use crate::{
    geometry::{Coordinate, CoordinatesRef, Direction, bbox::BoundingBox},
    math::{CrossProduct, Scalar, Tensor},
};
use std::array::from_fn;

const EPSILON: Scalar = 1e-10;

/// A circular arc from one ring vertex to the next, swept from the first
/// vertex's angle around `centre`, `ccw` (increasing angle) or not, by
/// whatever the true angular separation to the next vertex is.
#[derive(Clone, Copy)]
pub struct Arc2 {
    pub centre: [Scalar; 2],
    pub radius: Scalar,
    pub ccw: bool,
}

/// A planar B-rep face lifted into its own 2D frame: an orthonormal in-plane
/// basis `(u, v)` with `v = normal x u`, the outward unit normal, and every
/// bounding loop projected into that frame. A ring is a sequence of `(point,
/// arc to the next point)` pairs — `None` for a straight edge — so a loop may
/// mix straight rulings and circular fillet arcs; a loop that is itself one
/// whole circle (or two matching half-circles) is stored separately as an
/// exact `(centre, radius)` disk/annulus.
pub struct PlanarFace {
    pub origin: Coordinate<D>,
    pub normal: Direction<D>,
    pub u: Direction<D>,
    pub v: Direction<D>,
    pub rings: Vec<Vec<([Scalar; 2], Option<Arc2>)>>,
    pub circles: Vec<([Scalar; 2], Scalar)>,
    /// The outer polygon loop in world space, for box-overlap tests.
    pub outline: Vec<[Scalar; D]>,
    pub aabb: BoundingBox<D>,
}

impl PlanarFace {
    /// `point` dropped into the `(u, v)` frame, measured from `origin`.
    pub fn project(&self, point: &Coordinate<D>) -> [Scalar; 2] {
        let delta = point - &self.origin;
        [(&delta * &self.u).value(), (&delta * &self.v).value()]
    }
    /// The `(u, v)` pair lifted back onto the plane.
    pub fn unproject(&self, [s, t]: [Scalar; 2]) -> Coordinate<D> {
        let coordinates: [Scalar; D] =
            from_fn(|k| self.origin[k].value() + s * self.u[k].value() + t * self.v[k].value());
        Coordinate::from(coordinates)
    }
    /// Signed distance from `point` to the plane, positive on the outward side.
    pub fn plane_distance(&self, point: &Coordinate<D>) -> Scalar {
        ((point - &self.origin) * &self.normal).value()
    }

    /// Whether `uv` lies in the trimmed region: even-odd parity over every
    /// polygon/arc ring and circular loop.
    pub fn contains(&self, uv: [Scalar; 2]) -> bool {
        let mut inside = super::inside::mixed_point_in_polygon(uv, &self.rings);
        for &(centre, radius) in &self.circles {
            let d = (uv[0] - centre[0]).powi(2) + (uv[1] - centre[1]).powi(2);
            inside ^= d < radius * radius;
        }
        inside
    }

    /// The point of the trimming boundary nearest `uv`.
    pub fn nearest_boundary(&self, uv: [Scalar; 2]) -> [Scalar; 2] {
        let mut best = uv;
        let mut best_distance = Scalar::INFINITY;
        let mut consider = |candidate: [Scalar; 2]| {
            let d = (candidate[0] - uv[0]).powi(2) + (candidate[1] - uv[1]).powi(2);
            if d < best_distance {
                best_distance = d;
                best = candidate;
            }
        };
        for ring in &self.rings {
            let count = ring.len();
            for i in 0..count {
                let (a, arc) = ring[i];
                let (b, _) = ring[(i + 1) % count];
                match arc {
                    None => {
                        let (ex, ey) = (b[0] - a[0], b[1] - a[1]);
                        let span = ex * ex + ey * ey;
                        let t = if span > 0.0 {
                            (((uv[0] - a[0]) * ex + (uv[1] - a[1]) * ey) / span).clamp(0.0, 1.0)
                        } else {
                            0.0
                        };
                        consider([a[0] + t * ex, a[1] + t * ey]);
                    }
                    Some(arc) => consider(nearest_on_arc(uv, a, b, &arc)),
                }
            }
        }
        for &(centre, radius) in &self.circles {
            let (dx, dy) = (uv[0] - centre[0], uv[1] - centre[1]);
            let norm = (dx * dx + dy * dy).sqrt();
            let dir = if norm > EPSILON {
                [dx / norm, dy / norm]
            } else {
                [1.0, 0.0]
            };
            consider([centre[0] + radius * dir[0], centre[1] + radius * dir[1]]);
        }
        best
    }
}

/// `delta` normalized to `arc.ccw`'s sense: `[0, tau)` if increasing,
/// `(-tau, 0]` if decreasing. Unlike a symmetric wrap into `(-pi, pi]`, this
/// stays valid for a sweep past half a turn.
pub(super) fn offset_in_sense(delta: Scalar, ccw: bool) -> Scalar {
    let mut d = delta % std::f64::consts::TAU;
    if ccw {
        if d < 0.0 {
            d += std::f64::consts::TAU;
        }
    } else if d > 0.0 {
        d -= std::f64::consts::TAU;
    }
    d
}

/// The angular sweep from `a` to `b` around `arc`'s centre, in `arc.ccw`'s
/// sense: `(start angle, signed sweep)` with `start + sweep` landing exactly
/// on `b`'s angle.
pub(super) fn arc_sweep(a: [Scalar; 2], b: [Scalar; 2], arc: &Arc2) -> (Scalar, Scalar) {
    let angle = |p: [Scalar; 2]| (p[1] - arc.centre[1]).atan2(p[0] - arc.centre[0]);
    let start = angle(a);
    let sweep = offset_in_sense(angle(b) - start, arc.ccw);
    (start, sweep)
}

/// The point on the bounded arc from `a` to `b` nearest `uv`: the radial
/// projection if its angle falls within the swept range, else whichever
/// endpoint is nearer.
fn nearest_on_arc(uv: [Scalar; 2], a: [Scalar; 2], b: [Scalar; 2], arc: &Arc2) -> [Scalar; 2] {
    let (start, sweep) = arc_sweep(a, b, arc);
    let (dx, dy) = (uv[0] - arc.centre[0], uv[1] - arc.centre[1]);
    let norm = (dx * dx + dy * dy).sqrt();
    if norm > EPSILON {
        let offset = offset_in_sense(dy.atan2(dx) - start, arc.ccw);
        if (0.0..=1.0).contains(&(offset / sweep)) {
            return [
                arc.centre[0] + arc.radius * dx / norm,
                arc.centre[1] + arc.radius * dy / norm,
            ];
        }
    }
    let d = |p: [Scalar; 2]| (p[0] - uv[0]).powi(2) + (p[1] - uv[1]).powi(2);
    if d(a) <= d(b) { a } else { b }
}

impl Brep {
    pub fn planar_face(&self, face: &Face) -> Result<PlanarFace, &'static str> {
        let Surface::Plane(plane) = &face.surface else {
            return Err("planar_face called on a non-planar face");
        };
        let sign = if face.forward { 1.0 } else { -1.0 };
        let normal = (&plane.normal * sign).normalized();
        let reference = &plane.reference_direction;
        let projection = reference - &(&normal * (reference * &normal));
        if projection.norm().value() <= EPSILON {
            return Err("degenerate plane axes");
        }
        let u = projection.normalized();
        let v = normal.cross(&u);

        let first = face
            .bounds
            .iter()
            .filter_map(|bound| bound.vertices(&self.edges).ok())
            .flatten()
            .next()
            .ok_or("face has no bounding vertices")?;
        let origin = self.vertices[first].clone();
        let uv = |point: &Coordinate<D>| {
            let delta = point - &origin;
            [(&delta * &u).value(), (&delta * &v).value()]
        };

        let mut refs: Vec<&Coordinate<D>> = Vec::new();
        let mut outline = Vec::new();
        let mut rings = Vec::new();
        let mut circles = Vec::new();

        for bound in &face.bounds {
            let ring = bound.vertices(&self.edges)?;
            for &vertex in &ring {
                refs.push(&self.vertices[vertex]);
            }
            let curves: Vec<&Curve> = bound
                .half_edges
                .iter()
                .map(|half_edge| &self.edges[half_edge.edge].curve)
                .collect();

            if curves.iter().all(|curve| matches!(curve, Curve::Line(_))) {
                if ring.len() < 3 {
                    return Err("planar face has a straight loop of fewer than three vertices");
                }
                if outline.is_empty() {
                    outline.extend(ring.iter().map(|&v| from_fn(|k| self.vertices[v][k].value())));
                }
                rings.push(ring.iter().map(|&v| (uv(&self.vertices[v]), None)).collect());
            } else if let [Curve::Circle(circle)] = curves.as_slice() {
                if !parallel(&circle.axis, &normal) {
                    return Err("planar face's circular edge is not in the face plane");
                }
                circles.push((uv(&circle.center), circle.radius));
            } else if let [Curve::Circle(a), Curve::Circle(b)] = curves.as_slice() {
                if !parallel(&a.axis, &normal)
                    || (a.radius - b.radius).abs() > EPSILON
                    || (0..D).any(|k| (a.center[k].value() - b.center[k].value()).abs() > EPSILON)
                {
                    return Err("planar face's split circular loop is inconsistent");
                }
                circles.push((uv(&a.center), a.radius));
            } else if curves.iter().all(|curve| matches!(curve, Curve::Line(_) | Curve::Circle(_))) {
                // A genuine mix of straight rulings and fillet arcs (e.g. a
                // rounded-rectangle hole): keep every edge, in order, as an
                // exact straight or arc segment — no chord approximation.
                let mut mixed = Vec::with_capacity(ring.len());
                for (i, half_edge) in bound.half_edges.iter().enumerate() {
                    let point = uv(&self.vertices[ring[i]]);
                    let arc = match &self.edges[half_edge.edge].curve {
                        Curve::Line(_) => None,
                        Curve::Circle(circle) => {
                            if !parallel(&circle.axis, &normal) {
                                return Err("planar face's circular edge is not in the face plane");
                            }
                            let alignment = (0..D)
                                .map(|k| circle.axis[k].value() * normal[k].value())
                                .sum::<Scalar>();
                            let ccw = if half_edge.forward { alignment > 0.0 } else { alignment < 0.0 };
                            Some(Arc2 { centre: uv(&circle.center), radius: circle.radius, ccw })
                        }
                        _ => unreachable!("checked all-Line-or-Circle above"),
                    };
                    mixed.push((point, arc));
                }
                rings.push(mixed);
            } else {
                return Err("planar face has a mixed or partial trimming loop");
            }
        }

        let aabb = BoundingBox::from(refs.into_iter().collect::<CoordinatesRef<'_, D>>());
        Ok(PlanarFace {
            origin,
            normal,
            u,
            v,
            rings,
            circles,
            outline,
            aabb,
        })
    }
}

fn parallel(a: &Direction<D>, b: &Direction<D>) -> bool {
    (0..D).map(|k| a[k].value() * b[k].value()).sum::<Scalar>().abs() > 1.0 - 1.0e-6
}
