use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{Connectivity, Mesh},
    },
    math::assert::AssertionError,
};

pub const CONNECTIVITY: [[usize; 3]; 12] = [
    [0, 2, 1],
    [0, 3, 2],
    [4, 5, 6],
    [4, 6, 7],
    [0, 1, 5],
    [0, 5, 4],
    [3, 6, 2],
    [3, 7, 6],
    [0, 4, 7],
    [0, 7, 3],
    [1, 2, 6],
    [1, 6, 5],
];

pub const COORDINATES: [Coordinate<3>; 8] = [
    Coordinate::const_from([0.0, 0.0, 0.0]),
    Coordinate::const_from([1.0, 0.0, 0.0]),
    Coordinate::const_from([1.0, 1.0, 0.0]),
    Coordinate::const_from([0.0, 1.0, 0.0]),
    Coordinate::const_from([0.0, 0.0, 1.0]),
    Coordinate::const_from([1.0, 0.0, 1.0]),
    Coordinate::const_from([1.0, 1.0, 1.0]),
    Coordinate::const_from([0.0, 1.0, 1.0]),
];

pub fn mesh() -> Mesh<3> {
    let connectivities = vec![Connectivity::Triangular(CONNECTIVITY.to_vec().into())];
    let coordinates = Coordinates::from(COORDINATES);
    Mesh::from((connectivities, coordinates))
}

pub fn mesh_with_node_sets() -> Mesh<3> {
    let mut mesh = mesh();
    mesh.set_node_sets(vec![vec![0, 1], vec![2, 3]].into());
    mesh
}

#[test]
fn connectivity_coordinates() -> Result<(), AssertionError> {
    let _ = mesh();
    Ok(())
}

// #[test]
// fn connectivity_coordinates_ref() -> Result<(), AssertionError> {
//     let connectivity = CONNECTIVITY.to_vec();
//     let coordinates = Coordinates::from(COORDINATES);
//     let _ = TriangularMesh::from((connectivity, &coordinates));
//     Ok(())
// }

// #[test]
// fn connectivity_ref_coordinates() -> Result<(), AssertionError> {
//     let connectivity = CONNECTIVITY.to_vec();
//     let coordinates = Coordinates::from(COORDINATES);
//     let _ = TriangularMesh::from((&connectivity, coordinates));
//     Ok(())
// }

// #[test]
// fn connectivity_ref_coordinates_ref() -> Result<(), AssertionError> {
//     let connectivity = CONNECTIVITY.to_vec();
//     let coordinates = Coordinates::from(COORDINATES);
//     let _ = TriangularMesh::from((&connectivity, &coordinates));
//     Ok(())
// }
