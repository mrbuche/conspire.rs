#[cfg(test)]
mod test;

use crate::{
    geometry::{Coordinate, Coordinates},
    math::{CrossProduct, FxHashSet, Scalar, SquareMatrix, Tensor, Vector},
};

pub struct Jet {
    pub normal: Coordinate<3>,
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
    center: &Coordinate<3>,
    neighbors: &[Coordinate<3>],
    normal_guess: &Coordinate<3>,
) -> Option<Jet> {
    if neighbors.len() < 5 {
        return None;
    }
    let w = normal_guess.clone().normalized();
    let reference: Coordinate<3> = if w[0].abs() < 0.9 {
        [1.0, 0.0, 0.0].into()
    } else {
        [0.0, 1.0, 0.0].into()
    };
    let u = w.clone().cross(reference).normalized();
    let v = w.clone().cross(u.clone());
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

pub fn vertex_jets(connectivity: &[[usize; 3]], coordinates: &Coordinates<3>) -> Vec<Option<Jet>> {
    let count = coordinates.len();
    let mut neighbors: Vec<FxHashSet<usize>> = vec![FxHashSet::default(); count];
    let mut normals = vec![[0.0; 3]; count];
    for &[a, b, c] in connectivity {
        for (i, j) in [(a, b), (b, c), (c, a)] {
            neighbors[i].insert(j);
            neighbors[j].insert(i);
        }
        let (e1, e2) = (
            &coordinates[b] - &coordinates[a],
            &coordinates[c] - &coordinates[a],
        );
        let face = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];
        for vertex in [a, b, c] {
            (0..3).for_each(|k| normals[vertex][k] += face[k]);
        }
    }
    (0..count)
        .map(|vertex| {
            let mut ring = neighbors[vertex].clone();
            neighbors[vertex]
                .iter()
                .for_each(|&w| ring.extend(&neighbors[w]));
            ring.remove(&vertex);
            let points: Vec<Coordinate<3>> = ring.iter().map(|&w| coordinates[w].clone()).collect();
            fit_jet(&coordinates[vertex], &points, &normals[vertex].into())
        })
        .collect()
}
