//! Consistent outward orientation for a B-rep shell.
//!
//! A STEP file carries orientation twice: each `ADVANCED_FACE`'s `same_sense`
//! flag, and the winding of each `FACE_OUTER_BOUND`. Exporters keep the
//! windings coherent (ISO 10303-42 requires the material to lie left of an
//! oriented loop) but are sloppy with `same_sense`, so [`Brep::orient`] rebuilds
//! `Face::forward` from the windings: anchor each connected component to a
//! polygonal planar face — whose loop normal, by the right-hand rule, is the
//! true outward normal — then carry that across shared edges (adjacent faces
//! traverse a shared edge oppositely). A component with no polygonal planar
//! face falls back to the sign of its normal at its extreme vertex.

#[cfg(test)]
mod test;

use super::{Brep, D, Face, surface::Surface};
use std::array::from_fn;

impl Brep {
    /// Rewrites every face's `forward` flag so the shell's normals point
    /// consistently out of the solid. Best-effort: faces an incomplete shell
    /// cannot reach keep their as-read flag.
    pub fn orient(&mut self) {
        let faces = self.faces.len();
        if faces == 0 {
            return;
        }

        // Every use of an edge, as (face, traversal direction).
        let mut uses: Vec<Vec<(usize, bool)>> = vec![Vec::new(); self.edges.len()];
        for (index, face) in self.faces.iter().enumerate() {
            for bound in &face.bounds {
                for half_edge in &bound.half_edges {
                    if let Some(slot) = uses.get_mut(half_edge.edge) {
                        slot.push((index, half_edge.forward));
                    }
                }
            }
        }

        // Propagate a relative orientation from each seed: `flip[f]` is whether
        // face `f` is reversed relative to its component's seed.
        let mut flip = vec![false; faces];
        let mut seen = vec![false; faces];
        let mut component = vec![usize::MAX; faces];
        let mut components = 0;
        for seed in 0..faces {
            if seen[seed] {
                continue;
            }
            seen[seed] = true;
            component[seed] = components;
            let mut stack = vec![seed];
            while let Some(a) = stack.pop() {
                for bound in &self.faces[a].bounds {
                    for half_edge in &bound.half_edges {
                        let partners = match uses.get(half_edge.edge) {
                            // Skip open (1), seam-on-self, non-manifold (>2).
                            Some(partners) if partners.len() == 2 => partners,
                            _ => continue,
                        };
                        for &(b, forward_b) in partners {
                            if b == a {
                                continue;
                            }
                            let target = flip[a] ^ (half_edge.forward == forward_b);
                            if !seen[b] {
                                seen[b] = true;
                                flip[b] = target;
                                component[b] = components;
                                stack.push(b);
                            }
                        }
                    }
                }
            }
            components += 1;
        }

        for target in 0..components {
            let member = |index: usize| component[index] == target;
            // The forward flag every face in the component should have,
            // deduced from the component seed's winding where possible.
            let base = (0..faces)
                .filter(|&index| member(index))
                .find_map(|index| Some(winding_forward(self, index)? ^ flip[index]));

            match base {
                Some(base) => {
                    for index in (0..faces).filter(|&index| member(index)) {
                        self.faces[index].forward = base ^ flip[index];
                    }
                }
                None => {
                    // No polygonal planar face: keep the relative orientation,
                    // fix the global sense from the extreme-vertex normal.
                    for index in (0..faces).filter(|&index| member(index)) {
                        self.faces[index].forward ^= flip[index];
                    }
                    if inward_at_extremes(self, &component, target) {
                        for index in (0..faces).filter(|&index| member(index)) {
                            self.faces[index].forward = !self.faces[index].forward;
                        }
                    }
                }
            }
        }
    }
}

/// Whether a planar face's `plane.normal` already points the way its outer
/// loop winds (right-hand rule) — i.e. the correct `forward`. `None` unless
/// the face is planar with an outer loop of at least three usable vertices.
fn winding_forward(brep: &Brep, face: usize) -> Option<bool> {
    let Surface::Plane(plane) = &brep.faces[face].surface else {
        return None;
    };
    let ring = brep.faces[face]
        .bounds
        .first()?
        .vertices(&brep.edges)
        .ok()?;
    if ring.len() < 3 {
        return None;
    }
    let point: Vec<[f64; D]> = ring
        .iter()
        .map(|&vertex| from_fn(|k| brep.vertices[vertex][k].value()))
        .collect();
    // Newell's method: robust to non-planarity and vertex count.
    let mut normal = [0.0; D];
    for i in 0..point.len() {
        let a = point[i];
        let b = point[(i + 1) % point.len()];
        normal[0] += (a[1] - b[1]) * (a[2] + b[2]);
        normal[1] += (a[2] - b[2]) * (a[0] + b[0]);
        normal[2] += (a[0] - b[0]) * (a[1] + b[1]);
    }
    let alignment: f64 = (0..D).map(|k| normal[k] * plane.normal[k].value()).sum();
    (alignment.abs() > 1.0e-18).then_some(alignment > 0.0)
}

/// Whether the component's normal points inward at its extreme vertex along a
/// majority of the axes — the all-curved fallback for the global sense.
fn inward_at_extremes(brep: &Brep, component: &[usize], target: usize) -> bool {
    let mut violations = 0i32;
    for axis in 0..D {
        for sign in [1.0, -1.0] {
            let mut at: Option<(usize, [f64; D])> = None;
            for (index, face) in brep.faces.iter().enumerate() {
                if component[index] != target {
                    continue;
                }
                for point in face_points(brep, face) {
                    if at.is_none_or(|(_, best)| sign * point[axis] > sign * best[axis]) {
                        at = Some((index, point));
                    }
                }
            }
            if let Some((index, point)) = at
                && let Some(normal) = outward_normal(&brep.faces[index], point)
            {
                if sign * normal[axis] < -1.0e-9 {
                    violations += 1;
                } else if sign * normal[axis] > 1.0e-9 {
                    violations -= 1;
                }
            }
        }
    }
    violations > 0
}

/// Every loop vertex and pole of `face`, in world space.
fn face_points(brep: &Brep, face: &Face) -> Vec<[f64; D]> {
    let mut points: Vec<[f64; D]> = face
        .poles
        .iter()
        .map(|&vertex| from_fn(|k| brep.vertices[vertex][k].value()))
        .collect();
    for bound in &face.bounds {
        for half_edge in &bound.half_edges {
            let vertex = brep.edges[half_edge.edge].vertices[0];
            points.push(from_fn(|k| brep.vertices[vertex][k].value()));
        }
    }
    points
}

/// The surface's unit outward normal at `at`, with `face.forward` applied;
/// `None` for a B-spline face or a degenerate (on-axis / at-centre) query.
fn outward_normal(face: &Face, at: [f64; D]) -> Option<[f64; D]> {
    let sign = if face.forward { 1.0 } else { -1.0 };
    let scale = |v: [f64; D]| from_fn(|k| sign * v[k]);
    match &face.surface {
        Surface::Plane(plane) => Some(scale(from_fn(|k| plane.normal[k].value()))),
        Surface::Sphere(sphere) => {
            let centre: [f64; D] = from_fn(|k| sphere.origin[k].value());
            unit(from_fn(|k| at[k] - centre[k])).map(scale)
        }
        Surface::Cylinder(cylinder) => {
            let axis: [f64; D] = from_fn(|k| cylinder.axis[k].value());
            let origin: [f64; D] = from_fn(|k| cylinder.origin[k].value());
            unit(radial(from_fn(|k| at[k] - origin[k]), axis)).map(scale)
        }
        Surface::Cone(cone) => {
            let axis: [f64; D] = from_fn(|k| cone.axis[k].value());
            let origin: [f64; D] = from_fn(|k| cone.origin[k].value());
            let radial = unit(radial(from_fn(|k| at[k] - origin[k]), axis))?;
            let slope = cone.semi_angle.tan();
            let norm = (1.0 + slope * slope).sqrt();
            Some(scale(from_fn(|k| (radial[k] - slope * axis[k]) / norm)))
        }
        Surface::Torus(torus) => {
            let axis: [f64; D] = from_fn(|k| torus.axis[k].value());
            let centre: [f64; D] = from_fn(|k| torus.origin[k].value());
            let inplane = unit(radial(from_fn(|k| at[k] - centre[k]), axis))?;
            let ring: [f64; D] = from_fn(|k| centre[k] + torus.major_radius * inplane[k]);
            unit(from_fn(|k| at[k] - ring[k])).map(scale)
        }
        Surface::BSpline(_) => None,
    }
}

fn radial(rel: [f64; D], axis: [f64; D]) -> [f64; D] {
    let along: f64 = (0..D).map(|k| rel[k] * axis[k]).sum();
    from_fn(|k| rel[k] - along * axis[k])
}

fn unit(v: [f64; D]) -> Option<[f64; D]> {
    let norm = (0..D).map(|k| v[k] * v[k]).sum::<f64>().sqrt();
    (norm > 1.0e-12).then(|| v.map(|x| x / norm))
}
