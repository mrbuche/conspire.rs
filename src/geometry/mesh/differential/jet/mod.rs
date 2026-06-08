#[cfg(test)]
mod test;

use crate::{
    geometry::Coordinate,
    math::{CrossProduct, Scalar, SquareMatrix, Tensor, Vector},
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
