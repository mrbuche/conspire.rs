use crate::{
    geometry::{Coordinate, primitive::Cylinder},
    math::{Quantity, Scalar},
    units::Length,
};
use std::array::from_fn;

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

/// A reproducible spread of numbers, standing in for a random one so that a
/// failing case stays the same case next time it runs.
pub struct Spread(u64);

impl Default for Spread {
    fn default() -> Self {
        Self(0x2545_F491_4F6C_DD1D)
    }
}

impl Spread {
    /// The next number, somewhere between the bounds.
    pub fn between(&mut self, minimum: Scalar, maximum: Scalar) -> Scalar {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        minimum + (maximum - minimum) * ((self.0 >> 11) as Scalar / (1u64 << 53) as Scalar)
    }
    pub fn point(&mut self, reach: Scalar) -> Coordinate<D> {
        Coordinate::from(from_fn::<Scalar, D, _>(|_| self.between(-reach, reach)))
    }
}

/// Many overlapping cylinders strewn through a box, as a diatom's frustule is
/// many struts, so that a query has plenty to pass over.
pub fn thicket(count: usize) -> Vec<Cylinder> {
    let mut spread = Spread::default();
    (0..count)
        .map(|_| {
            let base = spread.point(10.0);
            let span = Coordinate::from(from_fn::<Scalar, D, _>(|_| spread.between(-3.0, 3.0)));
            Cylinder::new(
                [base.clone(), base + span],
                Length::meters(spread.between(0.2, 1.0)),
            )
        })
        .collect()
}

pub fn assert_length(value: Quantity<Length>, expected: Scalar) {
    assert!(
        (value.value() - expected).abs() < TOLERANCE,
        "expected {expected}, got {value}"
    )
}
