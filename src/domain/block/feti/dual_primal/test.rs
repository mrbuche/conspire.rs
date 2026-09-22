use super::{CornerSelection, build_splits};
use crate::domain::block::feti::interface::Partition;

#[test]
fn three_subdomains_share_one_corner() {
    let partition = Partition::new(vec![vec![0, 1, 2], vec![2, 3, 4], vec![2, 5, 6]]);
    let corners = CornerSelection::from_partition(&partition);
    assert_eq!(corners.nodes(), &[2]);
}

#[test]
fn two_subdomains_share_no_corner() {
    let partition = Partition::new(vec![vec![0, 1, 2], vec![2, 3, 4]]);
    let corners = CornerSelection::from_partition(&partition);
    assert!(corners.nodes().is_empty());
}

#[test]
fn split_separates_primal_and_dual_dofs() {
    let partition = Partition::new(vec![vec![0, 1, 2], vec![2, 3, 4], vec![2, 5, 6]]);
    let corners = CornerSelection::from_partition(&partition);
    let splits = build_splits(&partition, &corners, 3);
    assert_eq!(splits[0].primal(), &[6, 7, 8]);
    assert_eq!(splits[0].dual(), &[0, 1, 2, 3, 4, 5]);
    assert_eq!(splits[1].primal(), &[0, 1, 2]);
    assert_eq!(splits[1].dual(), &[3, 4, 5, 6, 7, 8]);
    assert_eq!(splits[2].primal(), &[0, 1, 2]);
    assert_eq!(splits[2].dual(), &[3, 4, 5, 6, 7, 8]);
}
