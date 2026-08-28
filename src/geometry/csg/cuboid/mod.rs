#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, Direction,
        solid::{Solid, SolidOracle},
    },
    math::Scalar,
};
use std::array::from_fn;

const D: usize = 3;

/// An axis-aligned rectangular box, meshed as a solid by the shared driver.
pub struct Cuboid {
    low: [Scalar; D],
    high: [Scalar; D],
}

impl Cuboid {
    /// The box spanning `low` to `high`; every `low[k]` must be strictly below
    /// `high[k]`.
    pub fn new(low: Coordinate<D>, high: Coordinate<D>) -> Result<Self, &'static str> {
        let low: [Scalar; D] = from_fn(|k| low[k].value());
        let high: [Scalar; D] = from_fn(|k| high[k].value());
        if (0..D).any(|k| low[k] >= high[k]) {
            return Err("cuboid low corner must be strictly below the high corner");
        }
        Ok(Self { low, high })
    }
}

impl Solid for Cuboid {
    type Oracle = CuboidOracle;

    fn bounding_box(&self) -> Result<(Coordinate<D>, Coordinate<D>), &'static str> {
        let low: Coordinate<D> = self.low.into();
        let high: Coordinate<D> = self.high.into();
        Ok((low, high))
    }

    fn oracle(&self) -> Result<CuboidOracle, &'static str> {
        Ok(CuboidOracle {
            low: self.low,
            high: self.high,
        })
    }
}

/// [`SolidOracle`] for an axis-aligned box: analytic closest surface point and
/// exact signed distance.
pub struct CuboidOracle {
    low: [Scalar; D],
    high: [Scalar; D],
}

impl SolidOracle for CuboidOracle {
    fn project(&self, query: &Coordinate<D>) -> Option<(Coordinate<D>, Direction<D>)> {
        let q: [Scalar; D] = from_fn(|k| query[k].value());
        let clamped: [Scalar; D] = from_fn(|k| q[k].clamp(self.low[k], self.high[k]));
        if clamped != q {
            // Outside: the clamped point is the closest point on the surface.
            let normal = unit(from_fn(|k| q[k] - clamped[k]))?;
            let point: Coordinate<D> = clamped.into();
            return Some((point, Direction::const_from(normal)));
        }
        // Inside: push to the nearest of the six faces.
        let (mut axis, mut sign, mut best) = (0, 1.0, Scalar::INFINITY);
        for (k, &qk) in q.iter().enumerate() {
            if qk - self.low[k] < best {
                best = qk - self.low[k];
                (axis, sign) = (k, -1.0);
            }
            if self.high[k] - qk < best {
                best = self.high[k] - qk;
                (axis, sign) = (k, 1.0);
            }
        }
        let mut coordinates = q;
        coordinates[axis] = if sign < 0.0 {
            self.low[axis]
        } else {
            self.high[axis]
        };
        let mut normal = [0.0; D];
        normal[axis] = sign;
        let point: Coordinate<D> = coordinates.into();
        Some((point, Direction::const_from(normal)))
    }

    fn signed_distance(&self, query: &Coordinate<D>) -> Scalar {
        let q: [Scalar; D] = from_fn(|k| query[k].value());
        let d: [Scalar; D] = from_fn(|k| (self.low[k] - q[k]).max(q[k] - self.high[k]));
        let outside = (0..D).map(|k| d[k].max(0.0).powi(2)).sum::<Scalar>().sqrt();
        let inside = d.into_iter().fold(Scalar::NEG_INFINITY, Scalar::max).min(0.0);
        // Exterior box SDF is `outside + inside` (negative within the box); flip
        // so the result is positive inside the solid.
        -(outside + inside)
    }
}

/// The unit vector along `v`, or `None` when `v` is degenerate.
fn unit(v: [Scalar; D]) -> Option<[Scalar; D]> {
    let norm = v.iter().map(|x| x * x).sum::<Scalar>().sqrt();
    (norm > 0.0).then(|| v.map(|x| x / norm))
}
