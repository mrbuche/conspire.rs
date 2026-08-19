use crate::{
    geometry::{
        Coordinate,
        primitive::{
            Cylinder, Solid, Union,
            test::{assert_length, upright},
        },
    },
    units::Length,
};

/// A cylinder along the x-axis long enough to run clear through [`upright`],
/// so that the two bury one another's surfaces where they cross.
fn spanning() -> Cylinder {
    Cylinder::new(
        [
            Coordinate::const_from([-2.0, 0.0, 1.0]),
            Coordinate::const_from([2.0, 0.0, 1.0]),
        ],
        Length::meters(1.0),
    )
}

fn crossed() -> Union<Cylinder> {
    Union::new(vec![upright(), spanning()])
}

#[test]
fn outside_takes_the_nearest_member() {
    // Beside the upright cylinder by two, but only one past the spanning cap.
    assert_length(
        crossed().signed_distance(&Coordinate::const_from([3.0, 0.0, 1.0])),
        1.0,
    )
}

#[test]
fn inside_either_member_is_inside_the_union() {
    assert_length(
        crossed().signed_distance(&Coordinate::const_from([1.5, 0.0, 1.0])),
        -0.5,
    )
}

#[test]
fn outside_both_members_is_outside_the_union() {
    assert_length(
        crossed().signed_distance(&Coordinate::const_from([0.0, 3.0, 1.0])),
        2.0,
    )
}

/// Uniting is order-independent, being a minimum over the members.
#[test]
fn order_does_not_matter() {
    let point = Coordinate::const_from([0.7, 0.4, 1.3]);
    assert_eq!(
        crossed().signed_distance(&point),
        Union::new(vec![spanning(), upright()]).signed_distance(&point)
    )
}

#[test]
fn closest_point_outside_takes_the_nearest_member() {
    let (point, normal) = crossed().closest_point(&Coordinate::const_from([3.0, 0.0, 1.0]));
    assert_eq!(point, Coordinate::const_from([2.0, 0.0, 1.0]));
    assert_eq!(normal, [1.0, 0.0, 0.0].into())
}

/// At the heart of the crossing the upright cylinder's nearest surface lies
/// buried a full radius inside the spanning one, so it is no longer boundary
/// and the spanning cylinder's exposed surface has to be taken instead.
#[test]
fn closest_point_rejects_a_buried_candidate() {
    let center = Coordinate::const_from([0.0, 0.0, 1.0]);
    let buried = upright().closest_point(&center).0;
    assert_length(spanning().signed_distance(&buried), -1.0);
    let (point, _) = crossed().closest_point(&center);
    assert_length(crossed().signed_distance(&point), 0.0);
    assert_ne!(point, buried)
}

/// Deep in the crossing the two members bury each other's nearest surface by
/// the same `1 - sqrt(3)/2`, leaving no exposed candidate to choose between,
/// so the answer has to come from projecting rather than from choosing.
#[test]
fn closest_point_projects_where_every_candidate_is_buried() {
    let crossed = crossed();
    let point = Coordinate::const_from([0.5, 0.5, 1.5]);
    crossed.solids().iter().for_each(|solid| {
        let candidate = solid.closest_point(&point).0;
        assert_length(
            crossed.signed_distance(&candidate),
            3.0_f64.sqrt() / 2.0 - 1.0,
        )
    });
    let (closest, _) = crossed.closest_point(&point);
    assert_length(crossed.signed_distance(&closest), 0.0)
}

/// Every closest point the union reports has to land on its surface, however
/// the members overlap there.
#[test]
fn closest_points_land_on_the_surface() {
    let crossed = crossed();
    [
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 1.5],
        [3.0, 0.0, 1.0],
        [0.0, 0.0, 5.0],
        [1.0, 1.0, 0.0],
        [-1.5, 0.2, 0.9],
    ]
    .into_iter()
    .for_each(|point| {
        let (closest, _) = crossed.closest_point(&Coordinate::const_from(point));
        assert_length(crossed.signed_distance(&closest), 0.0)
    })
}

#[test]
fn bounding_box_covers_every_member() {
    let extent = crossed().bounding_box();
    assert_eq!(extent.minimum(), &[-2.0, -1.0, 0.0].into());
    assert_eq!(extent.maximum(), &[2.0, 1.0, 2.0].into())
}

#[test]
fn contains_follows_the_sign() {
    let crossed = crossed();
    assert!(crossed.contains(&Coordinate::const_from([0.0, 0.0, 1.0])));
    assert!(crossed.contains(&Coordinate::const_from([1.8, 0.0, 1.0])));
    assert!(!crossed.contains(&Coordinate::const_from([2.2, 0.0, 1.0])))
}

#[test]
#[should_panic(expected = "a union needs at least one solid")]
fn empty_union_is_refused() {
    Union::<Cylinder>::new(vec![]);
}
