use crate::{
    geometry::{
        Coordinates,
        mesh::{
            Mesh,
            from::test::{CONNECTIVITY, COORDINATES},
        },
    },
    math::test::{TestError, assert_eq},
};

#[test]
fn bounding_boxes_and_centroids() {
    let connectivity = CONNECTIVITY.to_vec();
    let coordinates = Coordinates::from(COORDINATES);
    let mesh = Mesh::from((&connectivity, coordinates));
    mesh.bounding_boxes_and_centroids()
        .zip(mesh.bounding_boxes())
        .zip(mesh.centroids())
        .for_each(|(((bounding_box, centroid), bbox), cntrd)| {
            assert_eq!(bounding_box, bbox);
            assert_eq!(centroid, cntrd)
        })
}

#[test]
fn connectivity() {
    let connectivity = CONNECTIVITY.to_vec();
    let coordinates = Coordinates::from(COORDINATES);
    let mesh = Mesh::from((&connectivity, coordinates));
    assert_eq!(mesh.connectivity(), &connectivity)
}

#[test]
fn coordinates() -> Result<(), TestError> {
    let connectivity = CONNECTIVITY.to_vec();
    let coordinates = Coordinates::from(COORDINATES);
    let mesh = Mesh::from((connectivity, &coordinates));
    assert_eq(mesh.coordinates(), &coordinates)
}

#[test]
fn number_of_nodes() -> Result<(), TestError> {
    let connectivity = CONNECTIVITY.to_vec();
    let coordinates = Coordinates::from(COORDINATES);
    let mesh = Mesh::from((connectivity, coordinates));
    assert_eq(&mesh.number_of_nodes(), &COORDINATES.len())
}
