use crate::math::assert::Assert;
use crate::{
    geometry::{
        Coordinates,
        mesh::{
            Connectivity, Mesh,
            test::{CONNECTIVITY, COORDINATES, mesh},
        },
    },
    math::assert::AssertionError,
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
    Assert::eq(mesh.coordinates(), &coordinates)
}

fn blocks() -> Mesh<3> {
    (
        vec![
            Connectivity::Tetrahedral(vec![[0, 1, 2, 3]].into()),
            Connectivity::Tetrahedral(vec![[0, 2, 1, 4], [1, 2, 3, 4]].into()),
            Connectivity::Hexahedral(vec![[0, 1, 2, 3, 4, 5, 6, 7]].into()),
        ],
        vec![[0.0, 0.0, 0.0]; 8].into(),
    )
        .into()
}

#[test]
fn elements_blocks() {
    let mesh = blocks();
    assert_eq!(mesh.elements_blocks(), [0, 1, 1, 2]);
    assert_eq!(mesh.elements_blocks().len(), mesh.number_of_elements());
}

#[test]
fn elements_blocks_of_one_block() {
    let mesh: Mesh<3> = Mesh::from((
        vec![Connectivity::Polyhedral(
            (
                vec![vec![0, 1], vec![1, 2]],
                vec![vec![0, 1, 2], vec![1, 2, 3], vec![2, 3, 4]],
            )
                .into(),
        )],
        vec![[0.0, 0.0, 0.0]; 5].into(),
    ));
    assert_eq!(mesh.elements_blocks(), [0, 0]);
}

#[test]
fn elements_blocks_of_no_blocks() {
    let mesh: Mesh<3> = Mesh::from((Vec::<Connectivity>::new(), Vec::<[f64; 3]>::new().into()));
    assert!(mesh.elements_blocks().is_empty());
}

#[test]
fn number_of_nodes() {
    let mesh = mesh();
    assert_eq!(mesh.number_of_nodes(), COORDINATES.len())
}
