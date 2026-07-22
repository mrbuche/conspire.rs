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
    let mut edges: Vec<[usize; 2]> = network
        .quads
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
    assert!(
        network
            .edges
            .iter()
            .all(|edge| edges.binary_search(edge).is_ok())
    );
    let mut covered = vec![false; network.quads.len()];
    network.faces.iter().flatten().for_each(|&quad| {
        assert!(!covered[quad], "quad {quad} in two faces");
        covered[quad] = true;
    });
    assert!(covered.into_iter().all(|seen| seen));
    network
        .curves
        .iter()
        .for_each(|curve| assert!(curve.len() >= 2));
}
