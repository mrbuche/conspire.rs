use crate::{
    geometry::{
        Coordinate,
        csg::Sphere,
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

#[test]
fn signed_distance_is_radial() {
    let sphere = Sphere::new(point([1.0, 2.0, 3.0]), 4.0).unwrap();
    let oracle = sphere.oracle().unwrap();
    assert!((oracle.signed_distance(&point([1.0, 2.0, 3.0])) - 4.0).abs() < 1e-12);
    assert!((oracle.signed_distance(&point([1.0, 2.0, 9.0])) + 2.0).abs() < 1e-12);
    assert!(oracle.signed_distance(&point([5.0, 2.0, 3.0])).abs() < 1e-12);
}

#[test]
fn projects_along_the_radius() {
    let sphere = Sphere::new(point([0.0; 3]), 3.0).unwrap();
    let oracle = sphere.oracle().unwrap();
    let (surface, normal) = oracle.project(&point([6.0, 0.0, 0.0])).unwrap();
    assert!((surface[0].value() - 3.0).abs() < 1e-12);
    assert!(surface[1].value().abs() < 1e-12 && surface[2].value().abs() < 1e-12);
    assert!((normal[0].value() - 1.0).abs() < 1e-12);
}

#[test]
fn meshes_the_ball() {
    let sphere = Sphere::new(point([0.0; 3]), 3.0).unwrap();
    let mesh = sphere
        .mesh(
            &Uniform(length(0.6)),
            6,
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

    // No boundary node sits far outside the sphere, and the fit reaches it.
    let mut maximum = 0.0f64;
    for coordinate in mesh.coordinates() {
        let radius = (0..3)
            .map(|k| coordinate[k].value().powi(2))
            .sum::<f64>()
            .sqrt();
        maximum = maximum.max(radius);
    }
    assert!(
        (maximum - 3.0).abs() < 0.25,
        "outer radius reached {maximum}"
    );
}
