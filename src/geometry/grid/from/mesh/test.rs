use crate::{
    geometry::{
        Coordinate, Coordinates,
        grid::Voxels,
        mesh::{Connectivities, Connectivity, Mesh},
    },
    math::{Set, TensorVec},
};

#[test]
fn tetrahedron() {
    let mut coordinates = Coordinates::new();
    [
        [0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [0.0, 4.0, 0.0],
        [0.0, 0.0, 4.0],
    ]
    .into_iter()
    .for_each(|point| coordinates.push(Coordinate::const_from(point)));
    let connectivity = Connectivity::Tetrahedral(vec![[0, 1, 2, 3]].into());
    let mesh = Mesh::from((
        Connectivities::from((vec![connectivity], vec![5])),
        Set::from(coordinates),
    ));
    let voxels = Voxels::from_finite_elements(&mesh, 1.0);
    assert_eq!(*voxels.nel(), [4, 4, 4]);
    assert_eq!(voxels.data()[0], 5);
    assert_eq!(voxels.data()[3 + 12 + 48], 0);
}

#[test]
fn round_trips_hex_mesh() {
    let data: Vec<usize> = (1..=8).collect();
    let mesh = Mesh::from_voxels(Voxels::new(data.clone(), [2, 2, 2]), None);
    let voxels = Voxels::from_finite_elements(&mesh, 1.0);
    assert_eq!(*voxels.nel(), [2, 2, 2]);
    assert_eq!(voxels.data(), data);
}

#[test]
fn unfilled_voxels_are_void() {
    let data: Vec<usize> = (1..=8).collect();
    let mesh = Mesh::from_voxels(Voxels::new(data, [2, 2, 2]), Some(&[8]));
    let voxels = Voxels::from_finite_elements(&mesh, 1.0);
    assert_eq!(*voxels.nel(), [2, 2, 2]);
    assert_eq!(voxels.data()[1 + 2 + 4], 0);
    assert_eq!(voxels.data()[0], 1);
}
