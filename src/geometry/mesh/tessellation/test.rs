use super::NORMALIZED_DIRECTIONS;
use crate::{
    geometry::Direction,
    math::assert::{Assert, AssertionError},
};

const DIRECTIONS: [[f64; 3]; 3] = [
    [1.0, 0.140_412_03, 0.092_153_88],
    [0.097_153_2, 1.0, 0.131_771_4],
    [0.123_456_7, 0.087_654_3, 1.0],
];

#[test]
fn normalized_directions_are_the_normalized_off_axis_directions() -> Result<(), AssertionError> {
    DIRECTIONS
        .into_iter()
        .zip(NORMALIZED_DIRECTIONS)
        .try_for_each(|(direction, normalized)| {
            Assert::eq(&normalized, &Direction::from(direction).normalized())
        })
}
