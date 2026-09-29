use super::DirichletLocal;
use crate::domain::feti::dual_primal::condense::Condensed;
use crate::math::{SquareMatrix, Vector};

#[test]
fn splits_interior_and_boundary_and_computes_the_schur_complement() {
    let stiffness: SquareMatrix = [[4.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 4.0]]
        .into_iter()
        .map(|row| row.into_iter().collect())
        .collect();
    let dual_dofs = vec![1, 2];
    let interface_dofs = vec![1];
    let local = DirichletLocal::build(&stiffness, &dual_dofs, &interface_dofs);
    assert_eq!(local.boundary_dofs(), &[1]);
    let x: Vector = [1.0].into_iter().collect();
    assert!((local.apply(&x)[0] - 3.75).abs() < 1e-12);
}

#[test]
fn implicit_application_matches_the_explicit_schur_complement() {
    let stiffness: SquareMatrix = (0..6)
        .map(|i| {
            (0..6)
                .map(|j| {
                    if i == j {
                        5.0
                    } else {
                        1.0 / (1.0 + (i as f64 - j as f64).abs())
                    }
                })
                .collect()
        })
        .collect();
    let dual_dofs: Vec<usize> = (0..6).collect();
    let local = DirichletLocal::build(&stiffness, &dual_dofs, &[1, 3, 4]);
    assert_eq!(local.boundary_dofs(), &[1, 3, 4]);
    let explicit = Condensed::try_condense(&stiffness, &Vector::zero(6), &[1, 3, 4], &[0, 2, 5])
        .unwrap()
        .schur;
    let x: Vector = [1.0, -2.0, 0.5].into_iter().collect();
    let implicit = local.apply(&x);
    (0..3).for_each(|row| {
        let reference: f64 = (0..3).map(|column| explicit[row][column] * x[column]).sum();
        assert!((implicit[row] - reference).abs() < 1e-12);
    });
}
