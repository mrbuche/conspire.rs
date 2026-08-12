use crate::math::assert::Assert;
use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{Connectivity, Mesh, differential::laplace::Weighting},
    },
    math::{Scalar, Tensor, assert::AssertionError},
};

fn tri() -> Mesh<3> {
    (
        vec![Connectivity::Triangular(vec![[0_usize, 1, 2]].into())],
        Coordinates::from([
            Coordinate::const_from([0.0, 0.0, 0.0]),
            Coordinate::const_from([2.0, 0.0, 0.0]),
            Coordinate::const_from([0.0, 2.0, 0.0]),
        ]),
    )
        .into()
}

fn centroid(mesh: &Mesh<3>) -> Coordinate<3> {
    mesh.coordinates().iter().sum::<Coordinate<3>>() / 3.0
}

fn spread(mesh: &Mesh<3>) -> Scalar {
    let coordinates = mesh.coordinates();
    let center = centroid(mesh);
    (0..3)
        .map(|node| {
            (0..3)
                .map(|i| (coordinates[node][i] - center[i]).powi(2))
                .sum::<Scalar>()
        })
        .sum()
}

#[test]
fn full_step_moves_each_vertex_to_neighbor_centroid() -> Result<(), AssertionError> {
    let mut mesh = tri();
    mesh.laplace_smooth(1, 1.0, Weighting::Uniform, false, false)
        .unwrap();
    let coordinates = mesh.coordinates();
    Assert::default().eq_within_tols(&coordinates[0], &[1.0, 1.0, 0.0].into())?;
    Assert::default().eq_within_tols(&coordinates[1], &[0.0, 1.0, 0.0].into())?;
    Assert::default().eq_within_tols(&coordinates[2], &[1.0, 0.0, 0.0].into())
}

#[test]
fn zero_scale_is_identity() -> Result<(), AssertionError> {
    let mut mesh = tri();
    mesh.laplace_smooth(5, 0.0, Weighting::Uniform, false, false)
        .unwrap();
    let coordinates = mesh.coordinates();
    Assert::default().eq_within_tols(&coordinates[0], &[0.0, 0.0, 0.0].into())?;
    Assert::default().eq_within_tols(&coordinates[1], &[2.0, 0.0, 0.0].into())?;
    Assert::default().eq_within_tols(&coordinates[2], &[0.0, 2.0, 0.0].into())
}

#[test]
fn preserves_centroid() -> Result<(), AssertionError> {
    let before = centroid(&tri());
    let mut mesh = tri();
    mesh.laplace_smooth(4, 0.5, Weighting::Uniform, false, false)
        .unwrap();
    Assert::default().eq_within_tols(&before, &centroid(&mesh))
}

#[test]
fn shrinks_toward_centroid() {
    let before = spread(&tri());
    let mut mesh = tri();
    mesh.laplace_smooth(4, 0.5, Weighting::Uniform, false, false)
        .unwrap();
    assert!(spread(&mesh) < before);
}

fn square_about(center: Coordinate<3>) -> Mesh<3> {
    Mesh::from((
        vec![Connectivity::Triangular(
            vec![[0_usize, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]].into(),
        )],
        Coordinates::from([
            Coordinate::const_from([0.0, 0.0, 0.0]),
            Coordinate::const_from([2.0, 0.0, 0.0]),
            Coordinate::const_from([2.0, 2.0, 0.0]),
            Coordinate::const_from([0.0, 2.0, 0.0]),
            center,
        ]),
    ))
}

#[test]
fn preserve_boundary_ignores_interior_neighbors() -> Result<(), AssertionError> {
    let mut preserved_a = square_about([1.0, 1.0, 0.0].into());
    let mut preserved_b = square_about([1.5, 0.5, 0.0].into());
    preserved_a
        .laplace_smooth(1, 1.0, Weighting::Uniform, true, false)
        .unwrap();
    preserved_b
        .laplace_smooth(1, 1.0, Weighting::Uniform, true, false)
        .unwrap();
    Assert::default().eq_within_tols(&preserved_a.coordinates()[0], &[1.0, 1.0, 0.0].into())?;
    Assert::default()
        .eq_within_tols(&preserved_a.coordinates()[0], &preserved_b.coordinates()[0])?;
    let mut free = square_about([1.5, 0.5, 0.0].into());
    free.laplace_smooth(1, 1.0, Weighting::Uniform, false, false)
        .unwrap();
    assert!(
        (free.coordinates()[0][0] - preserved_b.coordinates()[0][0]).abs() > 1e-6
            || (free.coordinates()[0][1] - preserved_b.coordinates()[0][1]).abs() > 1e-6
    );
    Ok(())
}

fn two_block_strip(corner: Coordinate<3>) -> Mesh<3> {
    Mesh::from((
        vec![
            Connectivity::Triangular(vec![[0_usize, 1, 4], [0, 4, 3]].into()),
            Connectivity::Triangular(vec![[1_usize, 2, 5], [1, 5, 4]].into()),
        ],
        Coordinates::from([
            corner,
            Coordinate::const_from([1.0, 0.0, 0.0]),
            Coordinate::const_from([2.0, 0.0, 0.0]),
            Coordinate::const_from([0.0, 1.0, 0.0]),
            Coordinate::const_from([1.0, 1.0, 0.0]),
            Coordinate::const_from([2.0, 1.0, 0.0]),
        ]),
    ))
}

#[test]
fn preserve_interfaces_ignores_off_interface_neighbors() -> Result<(), AssertionError> {
    let mut interface_a = two_block_strip([0.0, 0.0, 0.0].into());
    let mut interface_b = two_block_strip([-1.0, -1.0, 0.0].into());
    interface_a
        .laplace_smooth(1, 1.0, Weighting::Uniform, false, true)
        .unwrap();
    interface_b
        .laplace_smooth(1, 1.0, Weighting::Uniform, false, true)
        .unwrap();
    Assert::default().eq_within_tols(&interface_a.coordinates()[1], &[1.0, 1.0, 0.0].into())?;
    Assert::default()
        .eq_within_tols(&interface_a.coordinates()[1], &interface_b.coordinates()[1])?;
    let mut free = two_block_strip([-1.0, -1.0, 0.0].into());
    free.laplace_smooth(1, 1.0, Weighting::Uniform, false, false)
        .unwrap();
    assert!(
        (free.coordinates()[1][0] - interface_b.coordinates()[1][0]).abs() > 1e-6
            || (free.coordinates()[1][1] - interface_b.coordinates()[1][1]).abs() > 1e-6
    );
    Ok(())
}

#[test]
fn cotangent_full_step() -> Result<(), AssertionError> {
    let mut mesh = tri();
    mesh.laplace_smooth(1, 1.0, Weighting::Cotangent, false, false)
        .unwrap();
    let coordinates = mesh.coordinates();
    Assert::default().eq_within_tols(&coordinates[0], &[1.0, 1.0, 0.0].into())?;
    Assert::default().eq_within_tols(&coordinates[1], &[0.0, 0.0, 0.0].into())?;
    Assert::default().eq_within_tols(&coordinates[2], &[0.0, 0.0, 0.0].into())
}

fn polygon() -> Mesh<3> {
    (
        vec![Connectivity::Polygonal(
            (
                vec![vec![0_usize, 1, 2, 3]],
                vec![vec![0_usize, 1], vec![1, 2], vec![2, 3], vec![3, 0]],
            )
                .into(),
        )],
        Coordinates::from([
            Coordinate::const_from([0.0, 0.0, 0.0]),
            Coordinate::const_from([2.0, 0.0, 0.0]),
            Coordinate::const_from([2.0, 2.0, 0.0]),
            Coordinate::const_from([0.0, 2.0, 0.0]),
        ]),
    )
        .into()
}

fn polyhedron() -> Mesh<3> {
    (
        vec![Connectivity::Polyhedral(
            (
                vec![vec![0_usize, 1, 2, 3, 4, 5]],
                vec![
                    vec![0_usize, 1, 2, 3],
                    vec![4, 5, 6, 7],
                    vec![0, 1, 5, 4],
                    vec![1, 2, 6, 5],
                    vec![2, 3, 7, 6],
                    vec![3, 0, 4, 7],
                ],
            )
                .into(),
        )],
        Coordinates::from([
            Coordinate::const_from([0.0, 0.0, 0.0]),
            Coordinate::const_from([1.0, 0.0, 0.0]),
            Coordinate::const_from([1.0, 1.0, 0.0]),
            Coordinate::const_from([0.0, 1.0, 0.0]),
            Coordinate::const_from([0.0, 0.0, 1.0]),
            Coordinate::const_from([1.0, 0.0, 1.0]),
            Coordinate::const_from([1.0, 1.0, 1.0]),
            Coordinate::const_from([0.0, 1.0, 1.0]),
        ]),
    )
        .into()
}

#[test]
fn polygonal_edge_adjacency() {
    assert_eq!(
        polygon().node_node_connectivity(),
        [vec![1, 3], vec![0, 2], vec![1, 3], vec![0, 2]]
    );
}

#[test]
fn polyhedral_edge_adjacency() {
    assert_eq!(
        polyhedron().node_node_connectivity(),
        [
            vec![1, 3, 4],
            vec![0, 2, 5],
            vec![1, 3, 6],
            vec![0, 2, 7],
            vec![0, 5, 7],
            vec![1, 4, 6],
            vec![2, 5, 7],
            vec![3, 4, 6]
        ]
    );
}

#[test]
fn polygonal_full_step() -> Result<(), AssertionError> {
    let mut mesh = polygon();
    mesh.laplace_smooth(1, 1.0, Weighting::Uniform, false, false)
        .unwrap();
    let coordinates = mesh.coordinates();
    Assert::default().eq_within_tols(&coordinates[0], &[1.0, 1.0, 0.0].into())?;
    Assert::default().eq_within_tols(&coordinates[2], &[1.0, 1.0, 0.0].into())
}

#[test]
fn polyhedral_full_step() -> Result<(), AssertionError> {
    let mut mesh = polyhedron();
    mesh.laplace_smooth(1, 1.0, Weighting::Uniform, false, false)
        .unwrap();
    Assert::default().eq_within_tols(
        &mesh.coordinates()[0],
        &[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0].into(),
    )
}

#[test]
fn polyhedral_preserve_boundary_retains_all_neighbors() -> Result<(), AssertionError> {
    let mut free = polyhedron();
    let mut mesh = polyhedron();
    free.laplace_smooth(4, 0.5, Weighting::Uniform, false, false)
        .unwrap();
    mesh.laplace_smooth(4, 0.5, Weighting::Uniform, true, false)
        .unwrap();
    free.coordinates()
        .iter()
        .zip(mesh.coordinates())
        .try_for_each(|(a, b)| Assert::default().eq_within_tols(a, b))
}

fn two_cube_coordinates() -> Coordinates<3> {
    Coordinates::from([
        Coordinate::const_from([0.0, 0.0, 0.0]),
        Coordinate::const_from([1.0, 0.0, 0.0]),
        Coordinate::const_from([1.0, 1.0, 0.0]),
        Coordinate::const_from([0.0, 1.0, 0.0]),
        Coordinate::const_from([0.0, 0.0, 1.0]),
        Coordinate::const_from([1.0, 0.0, 1.0]),
        Coordinate::const_from([1.0, 1.0, 1.0]),
        Coordinate::const_from([0.0, 1.0, 1.0]),
        Coordinate::const_from([2.0, 0.0, 0.0]),
        Coordinate::const_from([2.0, 1.0, 0.0]),
        Coordinate::const_from([2.0, 0.0, 1.0]),
        Coordinate::const_from([2.0, 1.0, 1.0]),
    ])
}

fn right_cube() -> Connectivity {
    Connectivity::Polyhedral(
        (
            vec![vec![0_usize, 1, 2, 3, 4, 5]],
            vec![
                vec![1_usize, 8, 9, 2],
                vec![5, 10, 11, 6],
                vec![1, 8, 10, 5],
                vec![8, 9, 11, 10],
                vec![9, 2, 6, 11],
                vec![2, 1, 5, 6],
            ],
        )
            .into(),
    )
}

fn two_polyhedra() -> Mesh<3> {
    (
        vec![
            Connectivity::Polyhedral(
                (
                    vec![vec![0_usize, 1, 2, 3, 4, 5]],
                    vec![
                        vec![0_usize, 1, 2, 3],
                        vec![4, 5, 6, 7],
                        vec![0, 1, 5, 4],
                        vec![1, 2, 6, 5],
                        vec![2, 3, 7, 6],
                        vec![3, 0, 4, 7],
                    ],
                )
                    .into(),
            ),
            right_cube(),
        ],
        two_cube_coordinates(),
    )
        .into()
}

fn hexahedron_and_polyhedron() -> Mesh<3> {
    (
        vec![
            Connectivity::Hexahedral(vec![[0_usize, 1, 2, 3, 4, 5, 6, 7]].into()),
            right_cube(),
        ],
        two_cube_coordinates(),
    )
        .into()
}

#[test]
fn polyhedral_node_element_connectivity() {
    assert_eq!(
        two_polyhedra().node_element_connectivity(),
        [
            vec![0],
            vec![0, 1],
            vec![0, 1],
            vec![0],
            vec![0],
            vec![0, 1],
            vec![0, 1],
            vec![0],
            vec![1],
            vec![1],
            vec![1],
            vec![1]
        ]
    );
}

#[test]
fn polyhedral_preserve_interfaces_holds_the_interface_plane() -> Result<(), AssertionError> {
    let mut mesh = two_polyhedra();
    mesh.laplace_smooth(1, 1.0, Weighting::Uniform, false, true)
        .unwrap();
    Assert::default().eq_within_tols(&mesh.coordinates()[1], &[1.0, 0.5, 0.5].into())?;
    Assert::default().eq_within_tols(&mesh.coordinates()[6], &[1.0, 0.5, 0.5].into())
}

#[test]
fn mixed_preserve_interfaces_holds_the_interface_plane() -> Result<(), AssertionError> {
    let mut mesh = hexahedron_and_polyhedron();
    mesh.laplace_smooth(1, 1.0, Weighting::Uniform, false, true)
        .unwrap();
    Assert::default().eq_within_tols(&mesh.coordinates()[1], &[1.0, 0.5, 0.5].into())?;
    Assert::default().eq_within_tols(&mesh.coordinates()[6], &[1.0, 0.5, 0.5].into())
}

#[test]
fn mixed_free_smoothing_matches_all_polyhedral() -> Result<(), AssertionError> {
    let mut polyhedral = two_polyhedra();
    let mut mixed = hexahedron_and_polyhedron();
    polyhedral
        .laplace_smooth(2, 0.5, Weighting::Uniform, false, false)
        .unwrap();
    mixed
        .laplace_smooth(2, 0.5, Weighting::Uniform, false, false)
        .unwrap();
    polyhedral
        .coordinates()
        .iter()
        .zip(mixed.coordinates())
        .try_for_each(|(a, b)| Assert::default().eq_within_tols(a, b))
}

#[test]
fn cotangent_rejects_meshes_that_are_not_all_triangular() {
    let mut polyhedral = polyhedron();
    let mut mixed = hexahedron_and_polyhedron();
    let mut hexahedral: Mesh<3> = (
        vec![Connectivity::Hexahedral(
            vec![[0_usize, 1, 2, 3, 4, 5, 6, 7]].into(),
        )],
        two_cube_coordinates(),
    )
        .into();
    let rejected = Err("cotangent weighting requires an all-triangular mesh");
    assert_eq!(
        polyhedral.laplace_smooth(1, 0.5, Weighting::Cotangent, false, false),
        rejected
    );
    assert_eq!(
        mixed.laplace_smooth(1, 0.5, Weighting::Cotangent, false, false),
        rejected
    );
    assert_eq!(
        hexahedral.taubin_smooth(1, 0.1, 0.5, Weighting::Cotangent, false, false),
        rejected
    );
}
