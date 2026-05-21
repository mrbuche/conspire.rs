use crate::{
    geometry::mesh::{TriangularMesh, tessellation::Tessellation},
    math::Tensor,
};

#[test]
fn from_triangluar_mesh() {
    let connectivity = vec![[0, 1, 2], [0, 3, 1]];
    let coordinates = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
    .into();
    let mesh = TriangularMesh::<1, _>::from((connectivity, coordinates));
    let tessellation = Tessellation::from(mesh);
    tessellation
        .normals
        .iter()
        .for_each(|normal| assert_eq!(normal, &[0.0, 0.0, 1.0].into()));
}
