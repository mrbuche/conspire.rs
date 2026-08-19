use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::tessellation::{Tessellation, trim::test::torus},
        primitive::Solid,
    },
    math::{Scalar, Tensor},
};

const MAJOR: Scalar = 1.0;
const MINOR: Scalar = 0.3;

/// The chord a facet cuts across the surface, which is as far as a tessellation
/// may sit inside the shape it stands for, and so how far the two may disagree.
const FACETING: Scalar = 2.0e-3;

fn tessellated() -> Tessellation {
    torus(MAJOR, MINOR, 128, 64)
}

/// A torus knows its own distance, so a tessellation of one can be held to it.
fn exact(point: &Coordinate<3>) -> Scalar {
    let radial = (point[0].value().powi(2) + point[1].value().powi(2)).sqrt();
    ((radial - MAJOR).powi(2) + point[2].value().powi(2)).sqrt() - MINOR
}

/// Faceting keeps the two from agreeing exactly, a tessellation cutting corners
/// off the surface it stands for, but no further apart than a facet is wide.
#[test]
fn distance_follows_the_torus_it_tessellates() {
    let tessellation = tessellated();
    [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.2],
        [1.25, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 1.4, 0.1],
        [-1.1, 0.1, 0.1],
        [0.0, 0.0, 3.0],
    ]
    .into_iter()
    .for_each(|point| {
        let point = Coordinate::const_from(point);
        let found = tessellation.signed_distance(&point).value();
        let expected = exact(&point);
        assert!(
            (found - expected).abs() < FACETING,
            "at {point}: tessellation says {found}, torus says {expected}"
        )
    })
}

/// Which side of the surface a point falls on is what trimming turns upon, and
/// the tessellation has to place it where the torus itself does.
#[test]
fn inside_and_outside_agree_with_the_torus() {
    let tessellation = tessellated();
    [
        ([1.0, 0.0, 0.0], true),
        ([1.0, 0.0, 0.25], true),
        ([-1.0, 0.0, 0.0], true),
        ([0.0, 0.0, 0.0], false),
        ([2.0, 0.0, 0.0], false),
        ([1.0, 0.0, 0.5], false),
    ]
    .into_iter()
    .for_each(|(point, within)| {
        let point = Coordinate::const_from(point);
        assert_eq!(exact(&point) <= 0.0, within, "the torus itself, at {point}");
        assert_eq!(tessellation.contains(&point), within, "at {point}")
    })
}

/// The bulk answer and the one-at-a-time answer come by different paths, one
/// across threads and one not, and may never come apart.
#[test]
fn asking_in_bulk_matches_asking_one_at_a_time() {
    let tessellation = tessellated();
    let points = Coordinates::from(vec![
        [1.0, 0.0, 0.0],
        [1.3, 0.2, 0.1],
        [0.0, 0.0, 0.0],
        [2.0, 2.0, 2.0],
        [-1.0, 0.0, 0.28],
        [0.7, -0.7, 0.0],
        [0.0, 1.0, -0.2],
    ]);
    tessellation
        .signed_distances(&points)
        .into_iter()
        .zip(points.iter())
        .for_each(|(bulk, point)| assert_eq!(bulk, tessellation.signed_distance(point)))
}

#[test]
fn closest_points_land_on_the_tessellation() {
    let tessellation = tessellated();
    [
        [2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.1],
        [0.0, 0.0, 3.0],
        [-1.4, 0.3, -0.2],
    ]
    .into_iter()
    .for_each(|point| {
        let point = Coordinate::const_from(point);
        let (closest, normal) = tessellation.closest_point(&point);
        assert!(exact(&closest).abs() < FACETING, "at {point}");
        assert!((normal.norm().value() - 1.0).abs() < 1.0e-12)
    })
}

#[test]
fn the_box_holds_the_whole_torus() {
    let extent = tessellated().bounding_box();
    let reach = MAJOR + MINOR;
    (0..2).for_each(|axis| {
        assert!((extent.minimum()[axis].value() + reach).abs() < FACETING);
        assert!((extent.maximum()[axis].value() - reach).abs() < FACETING)
    });
    assert!((extent.minimum()[2].value() + MINOR).abs() < FACETING);
    assert!((extent.maximum()[2].value() - MINOR).abs() < FACETING)
}
