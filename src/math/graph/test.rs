use super::{Graph, Incidence};

#[test]
fn transpose() {
    let incidence = Incidence::new(vec![vec![0, 1], vec![1, 2]], 4);
    let transposed = incidence.transpose(4);
    assert_eq!(
        transposed.as_slice(),
        [vec![0], vec![0, 1], vec![1], vec![]]
    );
    assert_eq!(transposed.number_of_incident_vertices(), 2);
}

#[test]
fn transpose_keeps_unrelated_trailing_vertices() {
    let incidence = Incidence::new(vec![vec![0]], 5);
    assert_eq!(incidence.transpose(5).as_slice().len(), 5);
}

#[test]
fn transpose_involution() {
    let incidence = Incidence::new(vec![vec![0, 2], vec![], vec![1, 2]], 3);
    assert_eq!(
        incidence.transpose(3).transpose(3).as_slice(),
        incidence.as_slice()
    );
}

#[test]
fn adjacency() {
    assert_eq!(
        Incidence::adjacency(vec![vec![1], vec![0]]).number_of_incident_vertices(),
        2
    );
}
