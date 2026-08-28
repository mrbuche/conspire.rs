#[cfg(test)]
mod test;

use super::{Brep, D, Face, surface::Surface};
use crate::{
    geometry::{
        Coordinates,
        mesh::{Connectivity, Mesh, Tessellation},
    },
    math::{TensorRank1, TensorVec},
};
use std::array::from_fn;

const EPSILON: f64 = 1e-10;

impl Brep {
    /// Triangulates every face against its plane and welds the result into a
    /// [`Tessellation`]. Only planar faces without holes are supported.
    pub fn tessellate(&self) -> Result<Tessellation, &'static str> {
        let mut connectivity = Vec::new();
        for face in &self.faces {
            self.tessellate_face(face, &mut connectivity)?;
        }
        let mut coordinates = Coordinates::with_capacity(self.vertices.len());
        self.vertices
            .iter()
            .for_each(|vertex| coordinates.push(vertex.clone()));
        let mesh: Mesh<D> = (
            vec![Connectivity::Triangular(connectivity.into())],
            coordinates,
        )
            .into();
        Ok(Tessellation::from(mesh))
    }

    fn tessellate_face(
        &self,
        face: &Face,
        connectivity: &mut Vec<[usize; 3]>,
    ) -> Result<(), &'static str> {
        if face.bounds.len() > 1 {
            return Err("faces with holes are not yet supported");
        }
        let Surface::Plane(plane) = &face.surface else {
            return Err("only planar faces are supported");
        };
        let outer = face.bounds.first().ok_or("face has no outer loop")?;
        let ring = outer.vertices(&self.edges)?;
        if ring.len() < 3 {
            return Err("face outer loop has fewer than three vertices");
        }
        let normal = array(&plane.normal);
        let outward = if face.forward { normal } else { negate(normal) };
        let reference = array(&plane.reference_direction);
        let u = normalize(reject(reference, outward)).ok_or("degenerate plane axes")?;
        let v = cross(outward, u);
        let polygon: Vec<[f64; 2]> = ring
            .iter()
            .map(|&vertex| {
                let point = array(&self.vertices[vertex]);
                [dot(point, u), dot(point, v)]
            })
            .collect();
        ear_clip(&polygon).into_iter().for_each(|[a, b, c]| {
            connectivity.push([ring[a], ring[b], ring[c]]);
        });
        Ok(())
    }
}

fn array<I, U>(tensor: &TensorRank1<D, I, U>) -> [f64; D] {
    from_fn(|axis| tensor[axis].value())
}

fn negate(a: [f64; D]) -> [f64; D] {
    a.map(|entry| -entry)
}

fn dot(a: [f64; D], b: [f64; D]) -> f64 {
    (0..D).map(|axis| a[axis] * b[axis]).sum()
}

fn cross(a: [f64; D], b: [f64; D]) -> [f64; D] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// The component of `a` orthogonal to the unit vector `onto`.
fn reject(a: [f64; D], onto: [f64; D]) -> [f64; D] {
    let scale = dot(a, onto);
    from_fn(|axis| a[axis] - scale * onto[axis])
}

fn normalize(a: [f64; D]) -> Option<[f64; D]> {
    let norm = dot(a, a).sqrt();
    (norm > EPSILON).then(|| a.map(|entry| entry / norm))
}

fn cross_2d(origin: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])
}

fn inside(point: [f64; 2], a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> bool {
    cross_2d(a, b, point) >= -EPSILON
        && cross_2d(b, c, point) >= -EPSILON
        && cross_2d(c, a, point) >= -EPSILON
}

/// Fan-free ear clipping for a simple planar polygon, returning triangles as
/// indices into `polygon`. Assumes the polygon is counter-clockwise after an
/// orientation check, so emitted triangles wind counter-clockwise too.
fn ear_clip(polygon: &[[f64; 2]]) -> Vec<[usize; 3]> {
    let count = polygon.len();
    if count < 3 {
        return Vec::new();
    }
    let mut indices: Vec<usize> = (0..count).collect();
    let area: f64 = (0..count)
        .map(|i| {
            let a = polygon[i];
            let b = polygon[(i + 1) % count];
            a[0] * b[1] - b[0] * a[1]
        })
        .sum();
    if area < 0.0 {
        indices.reverse();
    }
    let mut triangles = Vec::with_capacity(count - 2);
    while indices.len() > 3 {
        let n = indices.len();
        let ear = (0..n).find(|&i| {
            let prev = indices[(i + n - 1) % n];
            let curr = indices[i];
            let next = indices[(i + 1) % n];
            let (a, b, c) = (polygon[prev], polygon[curr], polygon[next]);
            cross_2d(a, b, c) > EPSILON
                && !indices
                    .iter()
                    .any(|&k| k != prev && k != curr && k != next && inside(polygon[k], a, b, c))
        });
        let Some(i) = ear else { break };
        triangles.push([indices[(i + n - 1) % n], indices[i], indices[(i + 1) % n]]);
        indices.remove(i);
    }
    if indices.len() == 3 {
        triangles.push([indices[0], indices[1], indices[2]]);
    }
    triangles
}

impl TryFrom<&Brep> for Tessellation {
    type Error = &'static str;
    fn try_from(brep: &Brep) -> Result<Self, Self::Error> {
        brep.tessellate()
    }
}
