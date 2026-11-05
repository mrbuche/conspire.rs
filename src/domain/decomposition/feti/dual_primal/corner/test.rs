use super::CornerSelection;
use crate::geometry::mesh::Decomposition;

#[test]
fn three_subdomains_share_one_corner() {
    let decomposition =
        Decomposition::from_parts_nodes(vec![vec![0, 1, 2], vec![2, 3, 4], vec![2, 5, 6]]);
    let corners = CornerSelection::from_decomposition(&decomposition);
    assert_eq!(corners.nodes(), &[2]);
}

#[test]
fn two_subdomains_share_no_corner() {
    let decomposition = Decomposition::from_parts_nodes(vec![vec![0, 1, 2], vec![2, 3, 4]]);
    let corners = CornerSelection::from_decomposition(&decomposition);
    assert!(corners.nodes().is_empty());
}
