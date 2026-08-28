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

/// A ball, meshed as a solid by the shared driver.
pub struct Sphere {
    center: [Scalar; D],
    radius: Scalar,
}

impl Sphere {
    /// The ball of `radius` about `center`; `radius` must be positive.
    pub fn new(center: Coordinate<D>, radius: Scalar) -> Result<Self, &'static str> {
        if radius <= 0.0 {
            return Err("sphere radius must be positive");
        }
        Ok(Self {
            center: from_fn(|k| center[k].value()),
            radius,
        })
    }
}

impl Solid for Sphere {
    type Oracle = SphereOracle;

    fn bounding_box(&self) -> Result<(Coordinate<D>, Coordinate<D>), &'static str> {
        let low: Coordinate<D> = from_fn(|k| self.center[k] - self.radius).into();
        let high: Coordinate<D> = from_fn(|k| self.center[k] + self.radius).into();
        Ok((low, high))
    }

    fn oracle(&self) -> Result<SphereOracle, &'static str> {
        Ok(SphereOracle {
            center: self.center,
            radius: self.radius,
        })
    }
}

/// [`SolidOracle`] for a ball: exact radial projection and signed distance.
pub struct SphereOracle {
    center: [Scalar; D],
    radius: Scalar,
}

impl SolidOracle for SphereOracle {
    fn project(&self, query: &Coordinate<D>) -> Option<(Coordinate<D>, Direction<D>)> {
        let delta: [Scalar; D] = from_fn(|k| query[k].value() - self.center[k]);
        let distance = delta.iter().map(|x| x * x).sum::<Scalar>().sqrt();
        let normal = if distance > 0.0 {
            delta.map(|x| x / distance)
        } else {
            let mut n = [0.0; D];
            n[0] = 1.0;
            n
        };
        let point: Coordinate<D> = from_fn(|k| self.center[k] + self.radius * normal[k]).into();
        Some((point, Direction::const_from(normal)))
    }

    fn signed_distance(&self, query: &Coordinate<D>) -> Scalar {
        let distance = (0..D)
            .map(|k| (query[k].value() - self.center[k]).powi(2))
            .sum::<Scalar>()
            .sqrt();
        self.radius - distance
    }
}
