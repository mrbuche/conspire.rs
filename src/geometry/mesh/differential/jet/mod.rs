#[cfg(test)]
mod test;

use crate::{
    geometry::{Coordinate, Coordinates},
    math::{CrossProduct, FxHashSet, Scalar, SquareMatrix, Tensor, Vector},
};

const D: usize = 3;

pub struct Jet {
    pub normal: Coordinate<D>,
    pub principal_curvatures: [Scalar; 2],
}

impl Jet {
    pub fn max_abs_curvature(&self) -> Scalar {
        self.principal_curvatures[0]
            .abs()
            .max(self.principal_curvatures[1].abs())
    }
}

pub fn fit_jet(
    center: &Coordinate<D>,
    neighbors: &Coordinates<D>,
    normal_guess: &Coordinate<D>,
) -> Option<Jet> {
    if neighbors.len() < 5 {
        return None;
    }
    let w = normal_guess.clone().normalized();
    let reference = if w[0].abs() < 0.9 {
        Coordinate::const_from([1.0, 0.0, 0.0])
    } else {
        Coordinate::const_from([0.0, 1.0, 0.0])
    };
    let u = w.cross(reference).normalized();
    let v = w.cross(&u);
    let mut normal_equations = SquareMatrix::zero(5);
    let mut rhs = Vector::zero(5);
    for neighbor in neighbors {
        let delta = neighbor - center;
        let (a, b, height) = (&delta * &u, &delta * &v, &delta * &w);
        let basis = [a, b, a * a, a * b, b * b];
        for i in 0..5 {
            for j in 0..5 {
                normal_equations[i][j] += basis[i] * basis[j];
            }
            rhs[i] += basis[i] * height;
        }
    }
    let c = normal_equations.solve_lu(&rhs).ok()?;
    let (h_u, h_v) = (c[0], c[1]);
    let (h_uu, h_uv, h_vv) = (2.0 * c[2], c[3], 2.0 * c[4]);
    let area = 1.0 + h_u * h_u + h_v * h_v;
    let root = area.sqrt();
    let (big_e, big_f, big_g) = (1.0 + h_u * h_u, h_u * h_v, 1.0 + h_v * h_v);
    let (big_l, big_m, big_n) = (h_uu / root, h_uv / root, h_vv / root);
    let mean = (big_e * big_n - 2.0 * big_f * big_m + big_g * big_l) / (2.0 * area);
    let gauss = (big_l * big_n - big_m * big_m) / area;
    let spread = (mean * mean - gauss).max(0.0).sqrt();
    let normal = (&(&w - &(&u * h_u)) - &(&v * h_v)).normalized();
    Some(Jet {
        normal,
        principal_curvatures: [mean + spread, mean - spread],
    })
}

pub fn vertex_jets(connectivity: &[[usize; 3]], coordinates: &Coordinates<D>) -> Vec<Option<Jet>> {
    let count = coordinates.len();
    let mut neighbors = vec![FxHashSet::default(); count];
    let mut normals = Coordinates::zero(count);
    for &[a, b, c] in connectivity {
        for (i, j) in [(a, b), (b, c), (c, a)] {
            neighbors[i].insert(j);
            neighbors[j].insert(i);
        }
        let (e1, e2) = (
            &coordinates[b] - &coordinates[a],
            &coordinates[c] - &coordinates[a],
        );
        let face = Coordinate::const_from([
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ]);
        for vertex in [a, b, c] {
            normals[vertex] += &face;
        }
    }
    (0..count)
        .map(|vertex| {
            let mut ring = neighbors[vertex].clone();
            neighbors[vertex]
                .iter()
                .for_each(|&w| ring.extend(&neighbors[w]));
            ring.remove(&vertex);
            let points = ring.iter().map(|&w| coordinates[w].clone()).collect();
            fit_jet(&coordinates[vertex], &points, &normals[vertex])
        })
        .collect()
}
