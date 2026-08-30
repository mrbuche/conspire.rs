use crate::{
    geometry::{
        Coordinate,
        csg::{Sphere, ops::UnionAll},
        mesh::{Connectivity, Fitting, Verdict},
        ntree::Balancing,
        solid::{Solid, SolidOracle, Uniform},
    },
    math::Quantity,
    units::Length,
};

fn length(value: f64) -> Quantity<Length> {
    Quantity::new(value)
}

fn point(entries: [f64; 3]) -> Coordinate<3> {
    Coordinate::from(entries)
}

fn sphere_pair() -> UnionAll<Sphere> {
    UnionAll::new(vec![
        Sphere::new(point([-1.0, 0.0, 0.0]), 2.0).unwrap(),
        Sphere::new(point([1.0, 0.0, 0.0]), 2.0).unwrap(),
    ])
}

#[test]
fn an_empty_union_is_the_empty_set() {
    let empty = UnionAll::<Sphere>::new(vec![]).oracle().unwrap();
    assert_eq!(empty.signed_distance(&point([0.0; 3])), f64::NEG_INFINITY);
    assert!(empty.project(&point([0.0; 3])).is_none());
}

#[test]
fn signed_distance_takes_the_nearer_interior() {
    let oracle = sphere_pair().oracle().unwrap();
    assert!(oracle.signed_distance(&point([-2.5, 0.0, 0.0])) > 0.0);
    assert!(oracle.signed_distance(&point([5.0, 0.0, 0.0])) < 0.0);
    assert!(oracle.signed_distance(&point([-3.0, 0.0, 0.0])).abs() < 1e-12);
}

#[test]
fn meshes_the_blob() {
    let mesh = sphere_pair()
        .mesh(
            &Uniform(length(0.6)),
            Some(6),
            0.1,
            Balancing::Strong(1),
            Fitting::Soft,
        )
        .unwrap();
    assert_eq!(mesh.connectivities().len(), 1);
    assert!(matches!(
        mesh.connectivities()[0],
        Connectivity::Hexahedral(_)
    ));
    let jacobians = mesh.minimum_scaled_jacobians();
    assert!(
        jacobians[0].iter().all(|&j| j > 0.0),
        "inverted hex: worst scaled Jacobian {}",
        jacobians[0].iter().cloned().fold(f64::INFINITY, f64::min)
    );

    let mut low = [f64::INFINITY; 3];
    let mut high = [f64::NEG_INFINITY; 3];
    for coordinate in mesh.coordinates() {
        for k in 0..3 {
            low[k] = low[k].min(coordinate[k].value());
            high[k] = high[k].max(coordinate[k].value());
        }
    }
    assert!((low[0] + 3.0).abs() < 0.3 && (high[0] - 3.0).abs() < 0.3);
    assert!(low[1] > -2.3 && high[1] < 2.3);
}
