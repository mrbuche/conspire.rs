use crate::{
    geometry::{
        Coordinates,
        mesh::{Connectivity, Mesh},
    },
    math::assert::AssertionError,
};

#[test]
fn restrict_breaks_vertex_pinch() -> Result<(), AssertionError> {
    let coordinates = Coordinates::from(vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [2.0, 1.0, 1.0],
        [2.0, 2.0, 1.0],
        [1.0, 2.0, 1.0],
        [1.0, 1.0, 2.0],
        [2.0, 1.0, 2.0],
        [2.0, 2.0, 2.0],
        [1.0, 2.0, 2.0],
    ]);
    let connectivities = vec![Connectivity::Hexahedral(
        vec![[0, 1, 2, 3, 4, 5, 6, 7], [6, 8, 9, 10, 11, 12, 13, 14]].into(),
    )];
    let mut mesh = Mesh::from((connectivities, coordinates));
    mesh.restrict().unwrap();
    assert_eq!(mesh.number_of_elements(), 1);
    Ok(())
}

#[test]
fn restrict_leaves_manifold_mesh_untouched() -> Result<(), AssertionError> {
    let coordinates = Coordinates::from(vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [2.0, 0.0, 1.0],
        [2.0, 1.0, 1.0],
    ]);
    let connectivities = vec![Connectivity::Hexahedral(
        vec![[0, 1, 2, 3, 4, 5, 6, 7], [1, 8, 9, 2, 5, 10, 11, 6]].into(),
    )];
    let mut mesh = Mesh::from((connectivities, coordinates));
    mesh.restrict().unwrap();
    assert_eq!(mesh.number_of_elements(), 2);
    Ok(())
}
