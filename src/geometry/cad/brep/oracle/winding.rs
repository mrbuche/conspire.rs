//! Generalized winding number of a point against the solid's exact trimmed
//! boundary — the inside/outside test that replaces reading a sign off the
//! single nearest face.
//!
//! Each face bound is walked into a closed 3D polyline (straight edges verbatim,
//! circular and elliptical edges sampled along their true arc). The winding
//! number is `1/4π` times the sum, over every fan triangle of every loop, of
//! the signed solid angle that triangle subtends at the query
//! (Van Oosterom–Strackee). For a shell whose loops are consistently wound —
//! which [`Brep::orient`](super::super::Brep::orient) has already checked —
//! this is `≈ 1` inside the solid, `≈ 0` outside, and smooth across the
//! surface, so it is robust where a nearest-face normal is not (creases,
//! medial axes, feature-dense pockets, small trim gaps).

use super::super::{Brep, D, curve::Curve};
use crate::math::Scalar;
use std::{array::from_fn, f64::consts::TAU};

/// The largest arc a single polyline segment spans when a curved edge is
/// sampled — a compromise between winding-number accuracy and cost.
const ARC_STEP: Scalar = std::f64::consts::FRAC_PI_8;

impl Brep {
    /// Every *planar* face bound as a closed 3D polyline, curved edges sampled,
    /// re-wound to the STEP "material on the left" convention (outer loop with
    /// the outward normal, holes against it) so the winding number sees holes
    /// subtract — many exporters leave inner loops wound like the outer and
    /// lean on the containment test to sort holes out. Curved faces contribute
    /// through [`FacePatch::winding_triangles`] instead.
    pub(super) fn winding_loops(&self) -> Result<Vec<Vec<[Scalar; D]>>, &'static str> {
        let mut loops = Vec::new();
        for face in &self.faces {
            let Some(outward) = face.normal() else {
                continue;
            };
            for (index, bound) in face.bounds.iter().enumerate() {
                let mut ring: Vec<[Scalar; D]> = Vec::new();
                for half_edge in &bound.half_edges {
                    let edge = self
                        .edges
                        .get(half_edge.edge)
                        .ok_or("half-edge references a missing edge")?;
                    let [start, end] = if half_edge.forward {
                        [edge.vertices[0], edge.vertices[1]]
                    } else {
                        [edge.vertices[1], edge.vertices[0]]
                    };
                    let a: [Scalar; D] = from_fn(|k| self.vertices[start][k].value());
                    let b: [Scalar; D] = from_fn(|k| self.vertices[end][k].value());
                    // Append this edge's points but not its endpoint — the next
                    // half-edge starts there, so the ring closes without a dup.
                    match &edge.curve {
                        Curve::Line(_) | Curve::BSpline(_) => ring.push(a),
                        Curve::Circle(circle) => ring.extend(arc_samples(
                            from_fn(|k| circle.center[k].value()),
                            from_fn(|k| circle.axis[k].value()),
                            from_fn(|k| circle.reference_direction[k].value()),
                            circle.radius,
                            circle.radius,
                            a,
                            b,
                            half_edge.forward,
                            start == end,
                        )),
                        Curve::Ellipse(ellipse) => ring.extend(arc_samples(
                            from_fn(|k| ellipse.center[k].value()),
                            from_fn(|k| ellipse.axis[k].value()),
                            from_fn(|k| ellipse.reference_direction[k].value()),
                            ellipse.major_radius,
                            ellipse.minor_radius,
                            a,
                            b,
                            half_edge.forward,
                            start == end,
                        )),
                    }
                }
                if ring.len() < 3 {
                    continue;
                }
                // Decide the winding sense in the face's own orthonormal plane
                // frame — its magnitude is scale-free, so the sign stays
                // reliable for a tiny hole rim where a raw 3D Newell normal,
                // fed sampled-arc points with a little out-of-plane error, is
                // just noise. Outer loops go counter-clockwise about the
                // outward normal, holes the other way.
                let outer = index == 0;
                if (signed_area(&ring, outward) > 0.0) != outer {
                    ring.reverse();
                }
                loops.push(ring);
            }
        }
        Ok(loops)
    }
}

/// Points along one circular or elliptical edge from `a` toward `b`, `a`
/// included and `b` excluded, following the edge's own rotational sense
/// (`forward` about `axis`, reversed otherwise; a whole turn when the
/// endpoints coincide).
#[expect(clippy::too_many_arguments)]
fn arc_samples(
    centre: [Scalar; D],
    axis: [Scalar; D],
    reference: [Scalar; D],
    major: Scalar,
    minor: Scalar,
    a: [Scalar; D],
    b: [Scalar; D],
    forward: bool,
    closed: bool,
) -> Vec<[Scalar; D]> {
    let e1 = unit(reject(reference, axis)).unwrap_or([1.0, 0.0, 0.0]);
    let e2 = cross(axis, e1);
    let angle = |p: [Scalar; D]| {
        let rel: [Scalar; D] = from_fn(|k| p[k] - centre[k]);
        dot(rel, e2).atan2(dot(rel, e1))
    };
    let start = angle(a);
    let sweep = if closed {
        if forward { TAU } else { -TAU }
    } else {
        let raw = angle(b) - start;
        if forward {
            raw.rem_euclid(TAU)
        } else {
            raw.rem_euclid(TAU) - TAU
        }
    };
    let steps = ((sweep.abs() / ARC_STEP).ceil() as usize).max(1);
    (0..steps)
        .map(|i| {
            let theta = start + sweep * (i as Scalar) / (steps as Scalar);
            from_fn(|k| centre[k] + major * theta.cos() * e1[k] + minor * theta.sin() * e2[k])
        })
        .collect()
}

/// `1/4π` times the total signed solid angle the trimmed boundary subtends at
/// `query`: `≈ 1` inside the solid, `≈ 0` outside. Planar faces contribute
/// through `loops` (fan-summed, exact); curved faces through `triangles`.
pub(super) fn generalized_winding_number(
    query: [Scalar; D],
    loops: &[Vec<[Scalar; D]>],
    triangles: &[[[Scalar; D]; 3]],
) -> Scalar {
    let mut total = 0.0;
    for ring in loops {
        let apex = ring[0];
        for i in 1..ring.len() - 1 {
            total += solid_angle(query, apex, ring[i], ring[i + 1]);
        }
    }
    for &[a, b, c] in triangles {
        total += solid_angle(query, a, b, c);
    }
    total / (2.0 * TAU)
}

/// Signed solid angle of triangle `(a, b, c)` seen from `q`
/// (Van Oosterom–Strackee).
fn solid_angle(q: [Scalar; D], a: [Scalar; D], b: [Scalar; D], c: [Scalar; D]) -> Scalar {
    let va: [Scalar; D] = from_fn(|k| a[k] - q[k]);
    let vb: [Scalar; D] = from_fn(|k| b[k] - q[k]);
    let vc: [Scalar; D] = from_fn(|k| c[k] - q[k]);
    let (la, lb, lc) = (norm(va), norm(vb), norm(vc));
    let numerator = dot(va, cross(vb, vc));
    let denominator =
        la * lb * lc + dot(va, vb) * lc + dot(vb, vc) * la + dot(vc, va) * lb;
    2.0 * numerator.atan2(denominator)
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

fn norm(a: [Scalar; D]) -> Scalar {
    dot(a, a).sqrt()
}

fn unit(a: [Scalar; D]) -> Option<[Scalar; D]> {
    let n = norm(a);
    (n > 1.0e-12).then(|| a.map(|x| x / n))
}

/// `v` with its `axis` component removed.
fn reject(v: [Scalar; D], axis: [Scalar; D]) -> [Scalar; D] {
    let d = dot(v, axis);
    from_fn(|k| v[k] - d * axis[k])
}

/// Twice the signed area of `ring` projected into the orthonormal plane whose
/// `+z` is `normal` — positive when the ring runs counter-clockwise about
/// `normal`.
fn signed_area(ring: &[[Scalar; D]], normal: [Scalar; D]) -> Scalar {
    let seed = if normal[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let u = unit(reject(seed, normal)).unwrap_or([1.0, 0.0, 0.0]);
    let v = cross(normal, u);
    let mut area = 0.0;
    for i in 0..ring.len() {
        let a = ring[i];
        let b = ring[(i + 1) % ring.len()];
        let (ax, ay) = (dot(a, u), dot(a, v));
        let (bx, by) = (dot(b, u), dot(b, v));
        area += ax * by - bx * ay;
    }
    area
}
