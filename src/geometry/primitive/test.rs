use crate::{
    geometry::{Coordinate, primitive::Cylinder},
    math::{Quantity, Scalar},
    units::Length,
};

pub const D: usize = 3;
pub const TOLERANCE: Scalar = 1.0e-12;

/// A unit-radius cylinder of height two, up the z-axis from the origin.
pub fn upright() -> Cylinder {
    Cylinder::new(
        [
            Coordinate::const_from([0.0, 0.0, 0.0]),
            Coordinate::const_from([0.0, 0.0, 2.0]),
        ],
        Length::meters(1.0),
    )
}

/// A unit-radius cylinder of height two, along the x-axis through the origin,
/// crossing [`upright`] at right angles.
pub fn crossing() -> Cylinder {
    Cylinder::new(
        [
            Coordinate::const_from([-1.0, 0.0, 1.0]),
            Coordinate::const_from([1.0, 0.0, 1.0]),
        ],
        Length::meters(1.0),
    )
}

pub fn assert_length(value: Quantity<Length>, expected: Scalar) {
    assert!(
        (value.value() - expected).abs() < TOLERANCE,
        "expected {expected}, got {value}"
    )
}
