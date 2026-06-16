use crate::{
    geometry::{
        Coordinates,
        mesh::{
            Connectivity, Mesh,
            tessellation::from::test::{NORMALS, tessellation},
            test::{CONNECTIVITY, COORDINATES},
        },
    },
    math::Tensor,
};

#[test]
fn triangular_mesh() {
    let mesh = Mesh::from(tessellation());
    match &mesh.connectivities()[0] {
        Connectivity::Triangular(triangles) => {
            assert!(triangles.iter().eq(CONNECTIVITY.iter()))
        }
        _ => panic!("expected Triangular block"),
    }
    let expected_coords = Coordinates::from(COORDINATES);
    assert_eq!(mesh.coordinates(), &expected_coords)
}

#[test]
fn connectivities_and_coordinates_and_normals() {
    let (connectivities, coordinates, normals) = tessellation().into();
    match &connectivities.members()[0] {
        Connectivity::Triangular(triangles) => {
            assert!(triangles.iter().eq(CONNECTIVITY.iter()))
        }
        _ => panic!("expected Triangular block"),
    }
    let expected_coords = Coordinates::from(COORDINATES);
    assert_eq!(coordinates, expected_coords);
    normals[0]
        .iter()
        .zip(NORMALS.iter())
        .for_each(|(a, b)| assert_eq!(a, b))
}
