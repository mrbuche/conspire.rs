use crate::{
    geometry::{
        Coordinates,
        mesh::{
            Connectivity, Mesh, PrimitiveConnectivity,
            from::test::{CONNECTIVITY, COORDINATES, mesh},
        },
    },
    math::test::{TestError, assert_eq},
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
        Connectivity::Triangular(PrimitiveConnectivity(t)) => {
            assert_eq!(t, &CONNECTIVITY.to_vec())
        }
        _ => panic!("expected Triangular block"),
    }
}

#[test]
fn coordinates() -> Result<(), TestError> {
    let mesh = mesh();
    let coordinates = Coordinates::from(COORDINATES);
    assert_eq(mesh.coordinates(), &coordinates)
}

#[test]
fn number_of_nodes() -> Result<(), TestError> {
    let mesh = mesh();
    assert_eq(&mesh.number_of_nodes(), &COORDINATES.len())
}
