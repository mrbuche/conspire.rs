use crate::geometry::{
    Coordinates,
    mesh::{
        Connectivities, Connectivity, NodeSet, NodeSets, SideSet, SideSets,
        test::{CONNECTIVITY, COORDINATES, mesh},
    },
};

#[test]
fn connectivities_and_coordinates() {
    let (connectivities, coordinates) = mesh().into();
    let expected = Coordinates::from(COORDINATES);
    assert_eq!(coordinates, expected);
    match connectivities.into_blocks().into_iter().next().unwrap() {
        Connectivity::Triangular(triangles) => {
            assert!(triangles.into_iter().eq(CONNECTIVITY))
        }
        _ => panic!("expected Triangular block"),
    }
}

#[test]
fn connectivities_coordinates_and_sets() {
    let mut mesh = mesh();
    mesh.set_node_sets(vec![NodeSet::new(vec![0, 1]).with_name("bottom")].into());
    mesh.set_side_sets(vec![SideSet::new(vec![(0, 1)]).with_label(4)].into());
    let (connectivities, coordinates, node_sets, side_sets): (
        Connectivities,
        Coordinates<3>,
        NodeSets,
        SideSets,
    ) = mesh.into();
    assert_eq!(connectivities.blocks().len(), 1);
    assert_eq!(coordinates, Coordinates::from(COORDINATES));
    assert_eq!(node_sets.by_name("bottom").unwrap().nodes(), [0, 1]);
    assert_eq!(side_sets.by_label(4).unwrap().sides(), [(0, 1)]);
}
