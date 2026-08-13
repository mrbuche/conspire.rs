use crate::geometry::{
    Coordinate, Coordinates,
    mesh::{Connectivity, Mesh, Verdict},
};

fn polyhedron() -> Mesh<3> {
    (
        vec![Connectivity::Polyhedral(
            (
                vec![vec![0_usize, 1, 2, 3, 4, 5]],
                vec![
                    vec![0_usize, 1, 2, 3],
                    vec![4, 5, 6, 7],
                    vec![0, 1, 5, 4],
                    vec![1, 2, 6, 5],
                    vec![2, 3, 7, 6],
                    vec![3, 0, 4, 7],
                ],
            )
                .into(),
        )],
        Coordinates::from([
            Coordinate::const_from([0.0, 0.0, 0.0]),
            Coordinate::const_from([1.0, 0.0, 0.0]),
            Coordinate::const_from([1.0, 1.0, 0.0]),
            Coordinate::const_from([0.0, 1.0, 0.0]),
            Coordinate::const_from([0.0, 0.0, 1.0]),
            Coordinate::const_from([1.0, 0.0, 1.0]),
            Coordinate::const_from([1.0, 1.0, 1.0]),
            Coordinate::const_from([0.0, 1.0, 1.0]),
        ]),
    )
        .into()
}

#[test]
fn polyhedra_have_no_metrics_yet() {
    let mesh = polyhedron();
    assert!(mesh.maximum_edge_ratios()[0][0].is_nan());
    assert!(mesh.maximum_skews()[0][0].is_nan());
    assert!(mesh.minimum_jacobians()[0][0].is_nan());
    assert!(mesh.minimum_scaled_jacobians()[0][0].is_nan());
    assert!(mesh.volumes()[0][0].is_nan());
}

#[test]
fn unsupported_blocks_keep_one_entry_per_element() {
    let mesh = polyhedron();
    let volumes = mesh.volumes();
    assert_eq!(volumes.len(), 1);
    assert_eq!(volumes[0].len(), 1);
}
