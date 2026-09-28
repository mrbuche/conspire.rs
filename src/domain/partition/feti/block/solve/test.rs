use super::super::super::dual_primal::BoundaryConditions;
use super::{SolveError, solve};
use crate::{
    constitutive::solid::{
        elastic::test::{BULK_MODULUS, SHEAR_MODULUS},
        hyperelastic::NeoHookean,
    },
    fem::{
        NodalCoordinates, NodalReferenceCoordinates,
        block::{Block, element::linear::Tetrahedron},
    },
    geometry::mesh::Partition,
    math::Tensor,
};

/// Three tetrahedra sharing one face {1,2,3} — each subdomain is one
/// element, and since all three meet at nodes 1, 2 and 3, the standard
/// heuristic (shared by >= 3 subdomains) makes all three corners, pinning
/// out each SUBDOMAIN's local rigid-body modes. Coordinates: nodes 0-3
/// are the codebase's standard reference tetrahedron; nodes 4 and 5 are
/// chosen so elements [1,2,3,4] and [1,2,3,5] both have positive volume.
///
/// With no boundary conditions applied, this assembly is completely
/// free-floating in space and so still has 6 GLOBAL rigid-body modes.
/// Corner condensation only removes a subdomain's own local floating
/// modes; it can't remove a mode that moves the whole structure
/// together, since a Schur complement can't have lower rank than the
/// directions of the original system's null space that survive
/// projection onto the corner DOFs — see the two tests below for both
/// the unsupported (singular, correctly refused) and supported
/// (well-posed) cases.
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
    Partition::from_parts_nodes(vec![vec![0, 1, 2, 3], vec![1, 2, 3, 4], vec![1, 2, 3, 5]])
}

/// A free-floating assembly with no Dirichlet boundary condition has
/// global rigid-body modes that corner condensation alone can't remove
/// (see `coordinates`' doc comment) — `solve()` correctly refuses this
/// rather than silently returning garbage. Runs the whole pipeline
/// (real-Block extraction through condensation, coarse assembly, dual
/// PCG wiring, and primal recovery) up to that point without panicking
/// anywhere else, which is what this test actually exercises.
#[test]
fn a_free_floating_assembly_has_no_boundary_condition_to_pin_it() {
    let block = block();
    let nodal_coordinates = NodalCoordinates::from(coordinates());
    let result = solve(
        &block,
        &nodal_coordinates,
        &partition(),
        &BoundaryConditions::none(),
        3,
    );
    assert!(matches!(result, Err(SolveError::SingularCoarseProblem)));
}

/// Fixes nodes 0, 4 and 5 — the three exclusive apex nodes, one per
/// subdomain, none of them shared/corner nodes — completely. Three
/// non-collinear fully-fixed points remove all 6 global rigid-body
/// modes (more than sufficient; redundant on the translations).
///
/// With zero applied force and every prescribed value equal to the
/// reference position, the unique equilibrium is zero displacement
/// everywhere.
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
        3,
    )
    .unwrap_or_else(|_| panic!("solve failed"));
    assert_eq!(solution.len(), 6 * 3);
    solution.iter().for_each(|&entry| {
        assert!(entry.is_finite());
        assert!(entry.abs() < 1e-8);
    });
}

/// Fixes node 0 (exclusive) fully, plus node 1 and node 2 — both
/// CORNER nodes, shared by all three subdomains — fully and z-only
/// respectively: node0(xyz) removes translations, node1(xyz) removes 2
/// rotations (about axes through both node0 and node1), node2(z)
/// removes the third. This is the same scheme an earlier version of
/// this test used when `coarse::assemble` still sized the coarse
/// problem from raw corner-node count rather than the DOFs that
/// actually survive boundary conditions — it hit a singular coarse
/// problem then purely from that sizing bug, now fixed
/// (`CornerDofs`), not from anything wrong with this boundary
/// condition placement. This test is the direct regression check for
/// that fix: boundary conditions on corner-node components now work.
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
        3,
    )
    .unwrap_or_else(|_| panic!("solve failed"));
    assert_eq!(solution.len(), 6 * 3);
    solution.iter().for_each(|&entry| {
        assert!(entry.is_finite());
        assert!(entry.abs() < 1e-8);
    });
}
