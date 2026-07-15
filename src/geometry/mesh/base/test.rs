use crate::{
    geometry::{
        Coordinates,
        mesh::{
            Connectivity, Mesh,
            test::{CONNECTIVITY, COORDINATES, mesh},
        },
    },
    math::assert::{AssertionError, assert_eq},
};

#[test]
fn bounding_boxes_and_centroids() {
    let mesh: Mesh<3> = mesh();
    mesh.bounding_boxes_and_centroids()
        .zip(mesh.bounding_boxes())
        .zip(mesh.centroids())
        .for_each(|(((bounding_box, centroid), bbox), cntrd)| {
            assert_eq!(bounding_box, bbox);
            assert_eq!(centroid, cntrd)
        })
}

#[test]
fn connectivities() {
    let mesh = mesh();
    match &mesh.connectivities()[0] {
        Connectivity::Triangular(triangles) => {
            assert!(triangles.iter().eq(CONNECTIVITY.iter()))
        }
        _ => panic!("expected Triangular block"),
    }
}

#[test]
fn coordinates() -> Result<(), AssertionError> {
    let mesh = mesh();
    let coordinates = Coordinates::from(COORDINATES);
    assert_eq(mesh.coordinates(), &coordinates)
}

#[test]
fn number_of_nodes() -> Result<(), AssertionError> {
    let mesh = mesh();
    assert_eq(&mesh.number_of_nodes(), &COORDINATES.len())
}
