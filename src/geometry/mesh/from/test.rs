use crate::{geometry::mesh::test::mesh, math::test::TestError};

#[test]
fn connectivity_coordinates() -> Result<(), TestError> {
    let _ = mesh();
    Ok(())
}

// #[test]
// fn connectivity_coordinates_ref() -> Result<(), TestError> {
//     let connectivity = CONNECTIVITY.to_vec();
//     let coordinates = Coordinates::from(COORDINATES);
//     let _ = TriangularMesh::from((connectivity, &coordinates));
//     Ok(())
// }

// #[test]
// fn connectivity_ref_coordinates() -> Result<(), TestError> {
//     let connectivity = CONNECTIVITY.to_vec();
//     let coordinates = Coordinates::from(COORDINATES);
//     let _ = TriangularMesh::from((&connectivity, coordinates));
//     Ok(())
// }

// #[test]
// fn connectivity_ref_coordinates_ref() -> Result<(), TestError> {
//     let connectivity = CONNECTIVITY.to_vec();
//     let coordinates = Coordinates::from(COORDINATES);
//     let _ = TriangularMesh::from((&connectivity, &coordinates));
//     Ok(())
// }
