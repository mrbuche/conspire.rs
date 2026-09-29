use crate::{
    constitutive::solid::{
        elastic::test::{BULK_MODULUS, SHEAR_MODULUS},
        hyperelastic::NeoHookean,
    },
    domain::feti::{Feti, Formulation, SolveError, dual_primal::BoundaryConditions},
    fem::{
        NodalCoordinates, NodalReferenceCoordinates,
        block::{Block, element::linear::Tetrahedron},
    },
    geometry::mesh::Partition,
    math::{Tensor, Vector},
};

fn coordinates() -> Vec<[f64; 3]> {
    vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 2.0],
    ]
}

fn block() -> Block<NeoHookean, Tetrahedron, 1, 3, 4, 4> {
    let reference_coordinates = NodalReferenceCoordinates::from(coordinates());
    Block::from((
        NeoHookean {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        },
        vec![[0, 1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 5]],
        &reference_coordinates,
    ))
}

fn partition() -> Partition {
    Partition::from_parts_elements(
        vec![vec![0], vec![1], vec![2]],
        &[vec![0, 1, 2, 3], vec![1, 2, 3, 4], vec![1, 2, 3, 5]],
    )
}

fn solve(
    block: &Block<NeoHookean, Tetrahedron, 1, 3, 4, 4>,
    nodal_coordinates: &NodalCoordinates<3>,
    partition: &Partition,
    boundary_conditions: &BoundaryConditions,
) -> Result<Vector, SolveError> {
    Feti {
        partition: partition.clone(),
        ..Default::default()
    }
    .solve(block, nodal_coordinates, boundary_conditions)
}

#[test]
fn a_free_floating_assembly_has_no_boundary_condition_to_pin_it() {
    let block = block();
    let nodal_coordinates = NodalCoordinates::from(coordinates());
    let result = solve(
        &block,
        &nodal_coordinates,
        &partition(),
        &BoundaryConditions::none(),
    );
    assert!(matches!(result, Err(SolveError::SingularCoarseProblem)));
}

#[test]
fn a_supported_assembly_at_zero_deformation_solves_to_zero_displacement() {
    let block = block();
    let nodal_coordinates = NodalCoordinates::from(coordinates());
    let boundary_conditions = BoundaryConditions::new(vec![
        (0, 0),
        (0, 1),
        (0, 2),
        (4, 0),
        (4, 1),
        (4, 2),
        (5, 0),
        (5, 1),
        (5, 2),
    ]);
    let solution = solve(
        &block,
        &nodal_coordinates,
        &partition(),
        &boundary_conditions,
    )
    .unwrap_or_else(|_| panic!("solve failed"));
    assert_eq!(solution.len(), 6 * 3);
    solution.iter().for_each(|&entry| {
        assert!(entry.is_finite());
        assert!(entry.abs() < 1e-8);
    });
}

#[test]
fn a_boundary_condition_on_a_corner_node_solves_correctly() {
    let block = block();
    let nodal_coordinates = NodalCoordinates::from(coordinates());
    let boundary_conditions =
        BoundaryConditions::new(vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 2)]);
    let solution = solve(
        &block,
        &nodal_coordinates,
        &partition(),
        &boundary_conditions,
    )
    .unwrap_or_else(|_| panic!("solve failed"));
    assert_eq!(solution.len(), 6 * 3);
    solution.iter().for_each(|&entry| {
        assert!(entry.is_finite());
        assert!(entry.abs() < 1e-8);
    });
}

#[test]
fn element_systems_are_stress_free_and_symmetric_at_zero_deformation() {
    use crate::domain::feti::DecomposableElements;
    let block = block();
    let systems = block
        .element_systems(&NodalCoordinates::from(coordinates()))
        .unwrap();
    assert_eq!(systems.elements.len(), 3);
    systems.elements.iter().for_each(|element| {
        element
            .force
            .iter()
            .for_each(|&entry| assert!(entry.abs() < 1e-10));
        (0..12).for_each(|row| {
            (0..12).for_each(|column| {
                assert!(
                    (element.stiffness[row][column] - element.stiffness[column][row]).abs() < 1e-8
                )
            })
        })
    });
}

fn solve_classical(boundary_conditions: &BoundaryConditions) -> Result<Vector, SolveError> {
    Feti {
        partition: partition(),
        formulation: Formulation::Classical,
        ..Default::default()
    }
    .solve(
        &block(),
        &NodalCoordinates::from(coordinates()),
        boundary_conditions,
    )
}

#[test]
fn classical_refuses_a_subdomain_that_no_boundary_condition_pins() {
    let result = solve_classical(&BoundaryConditions::new(vec![(0, 0), (0, 1), (0, 2)]));
    assert!(matches!(
        result,
        Err(SolveError::FloatingSubdomain { removed: 3, .. })
    ));
}

#[test]
fn classical_matches_dual_primal_when_every_subdomain_is_pinned() {
    let boundary_conditions = BoundaryConditions::new(
        (1..4)
            .flat_map(|node| (0..3).map(move |component| (node, component)))
            .collect(),
    );
    let classical =
        solve_classical(&boundary_conditions).unwrap_or_else(|_| panic!("classical solve failed"));
    let dual_primal = solve(
        &block(),
        &NodalCoordinates::from(coordinates()),
        &partition(),
        &boundary_conditions,
    )
    .unwrap_or_else(|_| panic!("dual-primal solve failed"));
    classical
        .iter()
        .zip(dual_primal.iter())
        .for_each(|(a, b)| assert!((a - b).abs() < 1e-10));
}
