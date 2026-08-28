#[cfg(test)]
mod test;

use super::{Brep, D};
use std::array::from_fn;

/// Cosine of the deviation past which a dihedral or turn counts as sharp.
const SHARP: f64 = 0.866_025_403_784_438_6;

/// The edges and vertices a mesh must reproduce exactly, read straight off the
/// B-rep topology rather than recovered from a tessellation by angle.
pub struct Features {
    /// Indices into [`Brep::edges`] of the sharp edges.
    pub creases: Vec<usize>,
    /// Indices into [`Brep::vertices`] of the hard points.
    pub corners: Vec<usize>,
}

impl Brep {
    pub fn features(&self) -> Features {
        let mut incident = vec![Vec::new(); self.edges.len()];
        for (index, face) in self.faces.iter().enumerate() {
            for bound in &face.bounds {
                for half_edge in &bound.half_edges {
                    if let Some(faces) = incident.get_mut(half_edge.edge) {
                        faces.push(index);
                    }
                }
            }
        }
        let creases: Vec<usize> = (0..self.edges.len())
            .filter(|&edge| match incident[edge].as_slice() {
                [a, b] => dot(self.faces[*a].normal(), self.faces[*b].normal()) < SHARP,
                _ => true,
            })
            .collect();

        let mut through: Vec<Vec<usize>> = vec![Vec::new(); self.vertices.len()];
        for &edge in &creases {
            let [a, b] = self.edges[edge].vertices;
            through[a].push(b);
            through[b].push(a);
        }
        let corners: Vec<usize> = (0..self.vertices.len())
            .filter(|&vertex| match through[vertex].as_slice() {
                [] => false,
                [one, two] => dot(self.tangent(vertex, *one), self.tangent(vertex, *two)) > -SHARP,
                _ => true,
            })
            .collect();

        Features { creases, corners }
    }

    /// Unit vector from `vertex` toward `neighbor`.
    fn tangent(&self, vertex: usize, neighbor: usize) -> [f64; D] {
        let from = &self.vertices[vertex];
        let to = &self.vertices[neighbor];
        let delta: [f64; D] = from_fn(|axis| to[axis].value() - from[axis].value());
        let norm = dot(delta, delta).sqrt();
        if norm > f64::EPSILON {
            delta.map(|entry| entry / norm)
        } else {
            delta
        }
    }
}

fn dot(a: [f64; D], b: [f64; D]) -> f64 {
    (0..D).map(|axis| a[axis] * b[axis]).sum()
}
