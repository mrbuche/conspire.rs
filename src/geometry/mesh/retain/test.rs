use crate::geometry::{
    Coordinates,
    mesh::{
        Connectivity, Mesh,
        test::{COORDINATES, mesh},
    },
};

#[test]
fn retains_elements_and_compacts_nodes() {
    let mut mesh: Mesh<3> = mesh();
    let mut seen = Vec::new();
    mesh.retain_elements(|index, _, _| {
        seen.push(index);
        index < 2
    });
    assert_eq!(seen, (0..12).collect::<Vec<_>>());
    match &mesh.connectivities()[0] {
        Connectivity::Triangular(triangles) => {
            assert_eq!(
                triangles.iter().copied().collect::<Vec<_>>(),
                [[0, 1, 2], [0, 3, 1]]
            )
        }
        _ => panic!("expected Triangular block"),
    }
    assert_eq!(mesh.number_of_nodes(), 4);
    assert_eq!(
        mesh.coordinates(),
        &Coordinates::from([
            COORDINATES[0].clone(),
            COORDINATES[2].clone(),
            COORDINATES[1].clone(),
            COORDINATES[3].clone(),
        ])
    )
}

#[test]
fn retaining_nothing_leaves_no_nodes() {
    let mut mesh: Mesh<3> = mesh();
    mesh.retain_elements(|_, _, _| false);
    assert_eq!(mesh.number_of_nodes(), 0);
    match &mesh.connectivities()[0] {
        Connectivity::Triangular(triangles) => assert!(triangles.iter().next().is_none()),
        _ => panic!("expected Triangular block"),
    }
}
