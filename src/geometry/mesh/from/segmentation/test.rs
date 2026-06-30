use crate::geometry::{Coordinate, grid::Voxels, mesh::Mesh, segmentation::Segmentation};

#[test]
fn from_segmentation_applies_embedding() {
    let segmentation = Segmentation::new(
        Voxels::new(vec![1u8], [1, 1, 1]),
        Coordinate::from([2.0, 3.0, 4.0]),
        Coordinate::from([10.0, 20.0, 30.0]),
    );
    let mesh = Mesh::from_segmentation(segmentation, None);
    let coordinates = mesh.coordinates();
    assert_eq!(coordinates[0], Coordinate::from([10.0, 20.0, 30.0]));
    assert_eq!(coordinates[7], Coordinate::from([12.0, 23.0, 34.0]));
}
