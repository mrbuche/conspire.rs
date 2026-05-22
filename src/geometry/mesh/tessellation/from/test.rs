use crate::{
    geometry::{
        Coordinates,
        mesh::{TriangularMesh, tessellation::Tessellation},
    },
    math::{
        Tensor,
        test::{TestError, assert_eq},
    },
};

#[test]
fn from_triangluar_mesh() -> Result<(), TestError> {
    let connectivity: Vec<[usize; _]> = vec![[0, 1, 2], [0, 3, 1]];
    let coordinates = Coordinates::<_, 0>::from(vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
    ]);
    let mesh = TriangularMesh::from((connectivity, coordinates));
    let tessellation = Tessellation::from(mesh);
    tessellation
        .normals
        .iter()
        .try_for_each(|normal| assert_eq(normal, &[0.0, 0.0, 1.0].into()))
}
