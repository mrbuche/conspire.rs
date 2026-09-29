use super::{DirichletLocal, Subdomain};
use crate::domain::feti::{
    dual_primal::{
        CornerSelection,
        condense::Condensed,
        rigid::{kernel, kernel_pins},
    },
    interface::build_interfaces,
};
use crate::geometry::mesh::Partition;
use crate::math::{Matrix, SquareMatrix, Vector};

#[test]
fn splits_interior_and_boundary_and_computes_the_schur_complement() {
    let stiffness: SquareMatrix = [[4.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 4.0]]
        .into_iter()
        .map(|row| row.into_iter().collect())
        .collect();
    let dual_dofs = vec![1, 2];
    let interface_dofs = vec![1];
    let local = DirichletLocal::try_build(&stiffness, &dual_dofs, &interface_dofs).unwrap();
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
    let local = DirichletLocal::try_build(&stiffness, &dual_dofs, &[1, 3, 4]).unwrap();
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

#[test]
fn a_singular_interior_block_is_refused_though_the_dual_block_is_not() {
    let stiffness: SquareMatrix = [[0.0, 1.0], [1.0, 0.0]]
        .into_iter()
        .map(|row| row.into_iter().collect())
        .collect();
    assert!(stiffness.factorize_lu().is_ok());
    assert!(DirichletLocal::try_build(&stiffness, &[0, 1], &[1]).is_none());
}

#[test]
fn a_floating_subdomain_solves_a_consistent_system_through_its_kernel() {
    let partition = Partition::from_parts_nodes(vec![vec![0, 1], vec![1, 2]]);
    let (interfaces, _) = build_interfaces(&partition, &CornerSelection::new(vec![]), 1);
    let interface = interfaces.into_iter().next().unwrap();
    let stiffness: SquareMatrix = [[1.0, -1.0], [-1.0, 1.0]]
        .into_iter()
        .map(|row| row.into_iter().collect())
        .collect();
    let dual_dofs = vec![0, 1];
    let modes = kernel(&[[0.0], [1.0]], &[]);
    assert_eq!(modes.len(), 1);
    let pins = kernel_pins(&modes, &dual_dofs);
    let keep: Vec<usize> = (0..2).filter(|position| !pins.contains(position)).collect();
    let reduced: SquareMatrix = keep
        .iter()
        .map(|&row| keep.iter().map(|&col| stiffness[row][col]).collect())
        .collect();
    let dirichlet = DirichletLocal::try_build(&stiffness, &dual_dofs, interface.dofs()).unwrap();
    let subdomain = Subdomain::new(
        (),
        interface,
        stiffness.clone(),
        reduced.factorize_lu().unwrap(),
        dual_dofs,
        2,
        Matrix::zero(2, 0),
        Matrix::zero(0, 2),
        Vec::new(),
        Vec::new(),
        dirichlet,
    )
    .with_kernel(modes, keep);
    assert_eq!(subdomain.kernel().len(), 1);
    let force: Vector = [2.0, -2.0].into_iter().collect();
    let solved = subdomain.local_solve(&force);
    let applied = &stiffness * &solved;
    (0..2).for_each(|dof| assert!((applied[dof] - force[dof]).abs() < 1e-12));
}
