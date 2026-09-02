use crate::{
    geometry::{
        Coordinate,
        csg::{Sphere, ops::Intersection},
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
fn project_from_outside_lands_on_the_intersection_surface() {
    let oracle = Intersection::new(
        Sphere::new(point([0.0; 3]), 2.0).unwrap(),
        Sphere::new(point([1.0, 0.0, 0.0]), 2.0).unwrap(),
    )
    .oracle()
    .unwrap();

    // q is outside both spheres. Testing the query rather than the candidate
    // point against the other operand rejected both feet, and the fallback
    // returned the first sphere's foot (0, 0, 2) -- 0.24 outside the second.
    let (foot, _) = oracle.project(&point([0.0, 0.0, 3.0])).unwrap();
    assert!(
        oracle.signed_distance(&foot).abs() < 1.0e-6,
        "project returned a point {} off the intersection surface",
        oracle.signed_distance(&foot),
    );
}

fn lens() -> Intersection<Sphere, Sphere> {
    Intersection::new(
        Sphere::new(point([-1.0, 0.0, 0.0]), 2.0).unwrap(),
        Sphere::new(point([1.0, 0.0, 0.0]), 2.0).unwrap(),
    )
}

#[test]
fn disjoint_solids_have_no_bounding_box() {
    let apart = Intersection::new(
        Sphere::new(point([0.0; 3]), 1.0).unwrap(),
        Sphere::new(point([10.0, 0.0, 0.0]), 1.0).unwrap(),
    );
    assert!(apart.bounding_box().is_err());
}

#[test]
fn signed_distance_needs_both() {
    let oracle = lens().oracle().unwrap();
    // The shared centre is inside both spheres.
    assert!(oracle.signed_distance(&point([0.0; 3])) > 0.0);
    // Inside the left sphere only: outside the intersection.
    assert!(oracle.signed_distance(&point([-2.5, 0.0, 0.0])) < 0.0);
}

#[test]
fn meshes_the_lens() {
    let mesh = lens()
        .mesh(
            &Uniform(length(0.4)),
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

    // The lens is far narrower on x than either sphere: it closes at x = +-1.
    let mut low = [f64::INFINITY; 3];
    let mut high = [f64::NEG_INFINITY; 3];
    for coordinate in mesh.coordinates() {
        for k in 0..3 {
            low[k] = low[k].min(coordinate[k].value());
            high[k] = high[k].max(coordinate[k].value());
        }
    }
    assert!(low[0] > -1.1 && high[0] < 1.1, "x span {} .. {}", low[0], high[0]);
    assert!(low[1] > -1.9 && high[1] < 1.9);
}
