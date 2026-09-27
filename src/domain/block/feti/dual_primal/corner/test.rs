use super::CornerSelection;
use crate::geometry::mesh::Partition;

#[test]
fn three_subdomains_share_one_corner() {
    let partition = Partition::from_parts_nodes(vec![vec![0, 1, 2], vec![2, 3, 4], vec![2, 5, 6]]);
    let corners = CornerSelection::from_partition(&partition);
    assert_eq!(corners.nodes(), &[2]);
}

#[test]
fn two_subdomains_share_no_corner() {
    let partition = Partition::from_parts_nodes(vec![vec![0, 1, 2], vec![2, 3, 4]]);
    let corners = CornerSelection::from_partition(&partition);
    assert!(corners.nodes().is_empty());
}
