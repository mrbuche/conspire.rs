use crate::geometry::{
    mesh::tessellation::from::test::tessellation,
    ntree::{Balancing, CurvatureSizing},
};

#[test]
fn projected_network_covers_the_core_boundary() {
    let tessellation = tessellation();
    let network = tessellation
        .projected_network(Balancing::Strong, 4.0, CurvatureSizing::default(), 8)
        .unwrap();
    assert!(network.core.number_of_elements() > 0);
    assert!(network.surface.number_of_elements() > 0);
    assert!(!network.edges.is_empty());
    assert_eq!(network.curves.len(), network.edges.len());
    let quads = network.core.exterior_faces();
    let mut edges: Vec<[usize; 2]> = quads
        .iter()
        .flat_map(|quad| {
            (0..4).map(|i| {
                let mut edge = [quad[i], quad[(i + 1) % 4]];
                edge.sort_unstable();
                edge
            })
        })
        .collect();
    edges.sort_unstable();
    edges.dedup();
    assert_eq!(network.edges, edges);
    network
        .curves
        .iter()
        .for_each(|curve| assert!(curve.len() >= 2));
}
