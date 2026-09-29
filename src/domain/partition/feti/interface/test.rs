use super::build_interfaces;
use crate::domain::feti::dual_primal::CornerSelection;
use crate::geometry::mesh::Partition;
use crate::math::Vector;

#[test]
fn two_subdomains_one_shared_node() {
    let partition = Partition::from_parts_nodes(vec![vec![0, 1, 2], vec![2, 3, 4]]);
    let (interfaces, num_multipliers) =
        build_interfaces(&partition, &CornerSelection::new(vec![]), 3);
    assert_eq!(num_multipliers, 3);
    assert_eq!(interfaces.len(), 2);
    assert_eq!(interfaces[0].multipliers(), &[0, 1, 2]);
    assert_eq!(interfaces[0].dofs(), &[6, 7, 8]);
    assert_eq!(interfaces[1].multipliers(), &[0, 1, 2]);
    assert_eq!(interfaces[1].dofs(), &[0, 1, 2]);
}

#[test]
fn continuity_zeros_the_jump() {
    let partition = Partition::from_parts_nodes(vec![vec![0, 1, 2], vec![2, 3, 4]]);
    let (interfaces, num_multipliers) =
        build_interfaces(&partition, &CornerSelection::new(vec![]), 3);
    let mut local_a = Vector::zero(9);
    let mut local_b = Vector::zero(9);
    [1.0, 2.0, 3.0].into_iter().enumerate().for_each(|(i, v)| {
        local_a[6 + i] = v;
        local_b[i] = v;
    });
    let jump = interfaces[0].apply(&local_a, num_multipliers)
        + interfaces[1].apply(&local_b, num_multipliers);
    (0..num_multipliers).for_each(|i| assert_eq!(jump[i], 0.0));
}

#[test]
fn no_shared_nodes_yields_no_multipliers() {
    let partition = Partition::from_parts_nodes(vec![vec![0, 1], vec![2, 3]]);
    let (interfaces, num_multipliers) =
        build_interfaces(&partition, &CornerSelection::new(vec![]), 3);
    assert_eq!(num_multipliers, 0);
    assert!(interfaces[0].dofs().is_empty());
    assert!(interfaces[1].dofs().is_empty());
}

#[test]
fn scaling_takes_each_copy_of_a_node_to_its_deviation_from_the_average() {
    [2, 3, 5].into_iter().for_each(|copies| {
        let partition = Partition::from_parts_nodes(vec![vec![0]; copies]);
        let (interfaces, num_multipliers) =
            build_interfaces(&partition, &CornerSelection::new(vec![]), 1);
        assert_eq!(num_multipliers, copies - 1);
        let values: Vec<f64> = (0..copies).map(|copy| (copy * copy) as f64 - 1.5).collect();
        let jump = interfaces
            .iter()
            .zip(values.iter())
            .map(|(interface, &value)| {
                interface.apply(&[value].into_iter().collect::<Vector>(), num_multipliers)
            })
            .fold(Vector::zero(num_multipliers), |sum, part| sum + part);
        let mean = values.iter().sum::<f64>() / copies as f64;
        interfaces
            .iter()
            .zip(values.iter())
            .for_each(|(interface, &value)| {
                let scaled = interface.apply_transpose_scaled(&jump, 1);
                assert!((scaled[0] - (value - mean)).abs() < 1e-12);
            });
    });
}

#[test]
fn a_corner_node_gets_no_multiplier() {
    let partition = Partition::from_parts_nodes(vec![vec![0, 1, 2], vec![2, 3, 4]]);
    let (interfaces, num_multipliers) =
        build_interfaces(&partition, &CornerSelection::new(vec![2]), 3);
    assert_eq!(num_multipliers, 0);
    assert!(interfaces[0].dofs().is_empty());
    assert!(interfaces[1].dofs().is_empty());
}
