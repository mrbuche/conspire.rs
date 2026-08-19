use crate::{
    geometry::{
        Coordinate,
        primitive::{
            Solid,
            test::{assert_length, crossing, upright},
        },
    },
    math::Tensor,
};

#[test]
fn center_is_as_deep_as_the_radius() {
    assert_length(
        upright().signed_distance(&Coordinate::const_from([0.0, 0.0, 1.0])),
        -1.0,
    )
}

#[test]
fn lateral_surface_is_the_zero_level() {
    assert_length(
        upright().signed_distance(&Coordinate::const_from([1.0, 0.0, 1.0])),
        0.0,
    )
}

#[test]
fn cap_is_the_zero_level() {
    assert_length(
        upright().signed_distance(&Coordinate::const_from([0.0, 0.0, 2.0])),
        0.0,
    )
}

#[test]
fn beside_the_cylinder_measures_radially() {
    assert_length(
        upright().signed_distance(&Coordinate::const_from([3.0, 0.0, 1.0])),
        2.0,
    )
}

#[test]
fn beyond_the_cap_measures_axially() {
    assert_length(
        upright().signed_distance(&Coordinate::const_from([0.0, 0.0, 5.0])),
        3.0,
    )
}

/// Past the rim both overshoots count, and the distance is the hypotenuse
/// rather than either leg, which a naive maximum of the two would report.
#[test]
fn past_the_rim_combines_both_overshoots() {
    assert_length(
        upright().signed_distance(&Coordinate::const_from([4.0, 0.0, 6.0])),
        5.0,
    )
}

/// Inside, the nearer of the lateral surface and the caps governs, so a point
/// close under a cap is that far in even though it sits on the axis.
#[test]
fn inside_takes_the_nearest_surface() {
    assert_length(
        upright().signed_distance(&Coordinate::const_from([0.0, 0.0, 1.75])),
        -0.25,
    )
}

#[test]
fn closest_point_beside_the_cylinder_is_lateral() {
    let (point, normal) = upright().closest_point(&Coordinate::const_from([3.0, 0.0, 1.0]));
    assert_eq!(point, Coordinate::const_from([1.0, 0.0, 1.0]));
    assert_eq!(normal, [1.0, 0.0, 0.0].into())
}

#[test]
fn closest_point_beyond_the_cap_is_the_cap() {
    let (point, normal) = upright().closest_point(&Coordinate::const_from([0.0, 0.0, 5.0]));
    assert_eq!(point, Coordinate::const_from([0.0, 0.0, 2.0]));
    assert_eq!(normal, [0.0, 0.0, 1.0].into())
}

/// Past the rim the closest point is the rim itself, which clamping reaches
/// only because it clamps radially and axially at once.
#[test]
fn closest_point_past_the_rim_is_the_rim() {
    let (point, _) = upright().closest_point(&Coordinate::const_from([4.0, 0.0, 6.0]));
    assert_eq!(point, Coordinate::const_from([1.0, 0.0, 2.0]))
}

#[test]
fn closest_point_inside_is_the_nearest_surface() {
    let (point, normal) = upright().closest_point(&Coordinate::const_from([0.0, 0.0, 1.75]));
    assert_eq!(point, Coordinate::const_from([0.0, 0.0, 2.0]));
    assert_eq!(normal, [0.0, 0.0, 1.0].into())
}

/// A point on the axis has no radial direction of its own, so the projection
/// has to fall back to the axis' basis rather than divide by a zero radius.
#[test]
fn closest_point_on_the_axis_stays_finite() {
    let (point, _) = upright().closest_point(&Coordinate::const_from([0.0, 0.0, 1.0]));
    assert!(point.iter().all(|entry| !entry.is_nan()));
    assert_length(upright().signed_distance(&point), 0.0)
}

/// The surface leaves no offset to take a normal from, so the nearest face has
/// to supply one instead of a direction normalized from nothing.
#[test]
fn closest_point_on_the_surface_stays_finite() {
    let (point, normal) = upright().closest_point(&Coordinate::const_from([1.0, 0.0, 1.0]));
    assert_eq!(point, Coordinate::const_from([1.0, 0.0, 1.0]));
    assert_eq!(normal, [1.0, 0.0, 0.0].into())
}

#[test]
fn bounding_box_stands_the_radius_off_an_upright_cylinder() {
    let extent = upright().bounding_box();
    assert_eq!(extent.minimum(), &[-1.0, -1.0, 0.0].into());
    assert_eq!(extent.maximum(), &[1.0, 1.0, 2.0].into())
}

/// Along the cylinder's own axis the lateral surface reaches no further than
/// the endpoints, so the box stands off only across it.
#[test]
fn bounding_box_does_not_stand_off_along_the_axis() {
    let extent = crossing().bounding_box();
    assert_eq!(extent.minimum(), &[-1.0, -1.0, 0.0].into());
    assert_eq!(extent.maximum(), &[1.0, 1.0, 2.0].into())
}
