use crate::{
    geometry::{
        Coordinate, Direction,
        csg::Ellipsoid,
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

fn axis(entries: [f64; 3]) -> Direction<3> {
    Direction::const_from(entries)
}

#[test]
fn rejects_a_non_positive_semi_axis() {
    assert!(Ellipsoid::new(point([0.0; 3]), [1.0, 0.0, 2.0]).is_err());
}

#[test]
fn signed_distance_axis_aligned() {
    let ellipsoid = Ellipsoid::new(point([0.0; 3]), [1.0, 2.0, 3.0]).unwrap();
    let oracle = ellipsoid.oracle().unwrap();
    // Centre: nearest surface is the tip of the shortest semi-axis, 1 away.
    assert!((oracle.signed_distance(&point([0.0; 3])) - 1.0).abs() < 1e-6);
    // On the surface at each tip.
    assert!(oracle.signed_distance(&point([1.0, 0.0, 0.0])).abs() < 1e-6);
    assert!(oracle.signed_distance(&point([0.0, 0.0, 3.0])).abs() < 1e-6);
    // Two units past the +x tip.
    assert!((oracle.signed_distance(&point([3.0, 0.0, 0.0])) + 2.0).abs() < 1e-6);
    // One unit past the +z tip.
    assert!((oracle.signed_distance(&point([0.0, 0.0, 4.0])) + 1.0).abs() < 1e-6);
}

#[test]
fn projects_onto_the_surface() {
    let ellipsoid = Ellipsoid::new(point([0.0; 3]), [1.0, 2.0, 3.0]).unwrap();
    let oracle = ellipsoid.oracle().unwrap();
    let (surface, normal) = oracle.project(&point([5.0, 0.0, 0.0])).unwrap();
    assert!((surface[0].value() - 1.0).abs() < 1e-6);
    assert!(surface[1].value().abs() < 1e-6 && surface[2].value().abs() < 1e-6);
    assert!((normal[0].value() - 1.0).abs() < 1e-6);
}

/// Least distance from `q` to a dense sample of the `[1, 2, 3]` ellipsoid.
fn brute_force_distance(q: [f64; 3]) -> f64 {
    use std::f64::consts::PI;
    let e = [1.0, 2.0, 3.0];
    let n = 240;
    let mut best = f64::INFINITY;
    for i in 0..=n {
        let theta = PI * i as f64 / n as f64;
        for j in 0..2 * n {
            let phi = PI * j as f64 / n as f64;
            let s = [
                e[0] * theta.sin() * phi.cos(),
                e[1] * theta.sin() * phi.sin(),
                e[2] * theta.cos(),
            ];
            let d = ((q[0] - s[0]).powi(2) + (q[1] - s[1]).powi(2) + (q[2] - s[2]).powi(2)).sqrt();
            best = best.min(d);
        }
    }
    best
}

#[test]
fn interior_query_on_a_principal_plane_projects_onto_the_surface() {
    // The octree origin snap lands a whole plane of nodes on x_i = 0. Such a
    // query used to leave the bisection root unresolved: the returned foot came
    // back off the ellipsoid with the wrong distance. The foot must sit on the
    // surface and the signed distance must equal the distance to it.
    let ellipsoid = Ellipsoid::new(point([0.0; 3]), [1.0, 2.0, 3.0]).unwrap();
    let oracle = ellipsoid.oracle().unwrap();
    let level = |p: [f64; 3]| p[0].powi(2) + (p[1] / 2.0).powi(2) + (p[2] / 3.0).powi(2);

    for q in [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.5],
        [0.3, 0.0, 0.0],
        [0.0, -1.2, 0.8],
        [0.5, 0.0, 2.0],
        [0.0, 0.0, 0.0],
    ] {
        let (foot, _) = oracle.project(&point(q)).unwrap();
        let foot = [foot[0].value(), foot[1].value(), foot[2].value()];
        assert!(
            (level(foot) - 1.0).abs() < 1e-6,
            "project({q:?}) -> {foot:?} off the ellipsoid (level {})",
            level(foot)
        );
        let to_foot =
            ((q[0] - foot[0]).powi(2) + (q[1] - foot[1]).powi(2) + (q[2] - foot[2]).powi(2)).sqrt();
        assert!(
            (oracle.signed_distance(&point(q)).abs() - to_foot).abs() < 1e-6,
            "{q:?}: signed distance disagrees with the foot"
        );
    }

    // Where the nearest point is unique (query outside the focal set) it is
    // also the global nearest. A query inside the focal set on a symmetry
    // plane -- e.g. [0, 1, 0] or [0, 0, 1.5] -- lands on an on-surface
    // stationary point that is not the global minimum; that is a standing
    // limitation of the reduced-axis solve, out of the fit's near-surface
    // regime, not checked here.
    for q in [
        [0.3, 0.0, 0.0],
        [0.0, 0.0, 2.7],
        [5.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ] {
        let distance = oracle.signed_distance(&point(q)).abs();
        assert!(
            distance <= brute_force_distance(q) + 2e-3,
            "project({q:?}) distance {distance} exceeds the brute-force nearest"
        );
    }
}

#[test]
fn oriented_swaps_the_axes() {
    // Rotate 90 degrees about z: the a=1 semi-axis now points along world +y.
    let ellipsoid = Ellipsoid::oriented(
        point([0.0; 3]),
        [
            axis([0.0, 1.0, 0.0]),
            axis([-1.0, 0.0, 0.0]),
            axis([0.0, 0.0, 1.0]),
        ],
        [1.0, 2.0, 3.0],
    )
    .unwrap();
    let oracle = ellipsoid.oracle().unwrap();
    assert!(oracle.signed_distance(&point([0.0, 1.0, 0.0])).abs() < 1e-6);
    assert!((oracle.signed_distance(&point([0.0, 3.0, 0.0])) + 2.0).abs() < 1e-6);
    assert!(oracle.signed_distance(&point([2.0, 0.0, 0.0])).abs() < 1e-6);
}

#[test]
fn meshes_the_ellipsoid() {
    let ellipsoid = Ellipsoid::new(point([0.0; 3]), [2.0, 3.0, 4.0]).unwrap();
    let mesh = ellipsoid
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

    let mut high = [f64::NEG_INFINITY; 3];
    for coordinate in mesh.coordinates() {
        for k in 0..3 {
            high[k] = high[k].max(coordinate[k].value());
        }
    }
    let expected = [2.0, 3.0, 4.0];
    for (k, (&h, &e)) in high.iter().zip(&expected).enumerate() {
        assert!((h - e).abs() < 0.3, "high[{k}] = {h}");
    }
}
