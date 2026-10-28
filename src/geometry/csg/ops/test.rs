use crate::{
    geometry::{Coordinate, Coordinates, solid::SolidOracle},
    math::{
        Quantity, Tensor,
        assert::{Assert, AssertionError},
    },
    units::Length,
};

pub fn grid() -> Coordinates<3> {
    let axis: Vec<f64> = (0..9)
        .map(|i| -0.43 + 0.2 * i as f64 + 0.011 * i as f64)
        .collect();
    let mut points = Vec::new();
    for &x in &axis {
        for &y in &axis {
            for &z in &axis {
                points.push([x, y, z]);
            }
        }
    }
    points.into()
}

pub fn check(
    oracle: &impl SolidOracle,
    expected: impl Fn(&Coordinate<3>) -> Quantity<Length>,
) -> Result<(usize, usize), AssertionError> {
    let mut counts = (0, 0);
    for point in grid().iter() {
        Assert::default().eq_within_tols(oracle.signed_distance(point), &expected(point))?;
        match oracle.project(point) {
            None => counts.1 += 1,
            Some((closest, normal)) => {
                counts.0 += 1;
                Assert::default()
                    .eq_within_tols(oracle.signed_distance(&closest), &Length::meters(0.0))?;
                Assert::default().eq_within_tols(normal.norm(), &Quantity::new(1.0))?;
                assert!(
                    (&closest - point).norm() + Length::meters(1e-12)
                        >= oracle.signed_distance(point).abs()
                );
            }
        }
    }
    Ok(counts)
}
