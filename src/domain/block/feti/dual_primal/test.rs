use super::{BoundaryConditions, CornerSelection, build_splits};
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
    let (splits, corner_dofs) = build_splits(&partition, &corners, &BoundaryConditions::none(), 3);
    assert_eq!(corner_dofs.count(), 3);
    assert_eq!(splits[0].primal(), &[6, 7, 8]);
    assert_eq!(splits[0].dual(), &[0, 1, 2, 3, 4, 5]);
    assert_eq!(splits[1].primal(), &[0, 1, 2]);
    assert_eq!(splits[1].dual(), &[3, 4, 5, 6, 7, 8]);
    assert_eq!(splits[2].primal(), &[0, 1, 2]);
    assert_eq!(splits[2].dual(), &[3, 4, 5, 6, 7, 8]);
}

#[test]
fn a_fixed_dof_is_excluded_from_both_primal_and_dual() {
    let partition = Partition::new(vec![vec![0, 1, 2], vec![2, 3, 4]]);
    let corners = CornerSelection::from_partition(&partition);
    // No automatic corners here (only 2 subdomains share node 2), but node 0
    // component 1 is pinned externally.
    let boundary_conditions = BoundaryConditions::new(vec![(0, 1)]);
    let (splits, _) = build_splits(&partition, &corners, &boundary_conditions, 3);
    // Node 0 is local index 0 in subdomain 0: dofs 0,1,2. Component 1 (dof 1)
    // is fixed and should appear in neither primal nor dual.
    assert!(!splits[0].primal().contains(&1));
    assert!(!splits[0].dual().contains(&1));
    assert!(splits[0].dual().contains(&0));
    assert!(splits[0].dual().contains(&2));
}

#[test]
fn a_boundary_condition_on_a_corner_component_shrinks_the_coarse_problem() {
    let partition = Partition::new(vec![vec![0, 1, 2], vec![2, 3, 4], vec![2, 5, 6]]);
    let corners = CornerSelection::from_partition(&partition);
    assert_eq!(corners.nodes(), &[2]);
    // Node 2 is the only corner (3 free components, dimension 3). Pinning
    // one of its components should shrink the coarse problem by exactly
    // one DOF, not leave a permanently-unassembled (singular) slot for it.
    let boundary_conditions = super::BoundaryConditions::new(vec![(2, 1)]);
    let (splits, corner_dofs) = build_splits(&partition, &corners, &boundary_conditions, 3);
    assert_eq!(corner_dofs.count(), 2);
    // No subdomain's primal set should have 3 corner DOFs anymore, only 2.
    splits
        .iter()
        .for_each(|split| assert_eq!(split.primal().len(), 2));
}
