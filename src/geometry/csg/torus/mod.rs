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

/// A ring torus, meshed as a solid by the shared driver: a tube of radius
/// `minor_radius` swept around the axis at distance `major_radius`.
pub struct Torus {
    center: [Scalar; D],
    axis: [Scalar; D],
    major_radius: Scalar,
    minor_radius: Scalar,
}

impl Torus {
    /// The torus about `axis` through `center`. `minor_radius` must be positive
    /// and strictly less than `major_radius` (a ring torus, no self-crossing).
    pub fn new(
        center: Coordinate<D>,
        axis: Direction<D>,
        major_radius: Scalar,
        minor_radius: Scalar,
    ) -> Result<Self, &'static str> {
        if minor_radius <= 0.0 || major_radius <= minor_radius {
            return Err("torus needs 0 < minor_radius < major_radius");
        }
        let axis = unit(from_fn(|k| axis[k].value())).ok_or("degenerate torus axis")?;
        Ok(Self {
            center: from_fn(|k| center[k].value()),
            axis,
            major_radius,
            minor_radius,
        })
    }
}

impl Solid for Torus {
    type Oracle = TorusOracle;

    fn bounding_box(&self) -> Result<(Coordinate<D>, Coordinate<D>), &'static str> {
        // Farthest along world axis k: `major * sqrt(1 - axisₖ²) + minor`.
        let half: [Scalar; D] = from_fn(|k| {
            self.major_radius * (1.0 - self.axis[k] * self.axis[k]).max(0.0).sqrt()
                + self.minor_radius
        });
        let low: Coordinate<D> = from_fn(|k| self.center[k] - half[k]).into();
        let high: Coordinate<D> = from_fn(|k| self.center[k] + half[k]).into();
        Ok((low, high))
    }

    fn oracle(&self) -> Result<TorusOracle, &'static str> {
        Ok(TorusOracle {
            center: self.center,
            axis: self.axis,
            major_radius: self.major_radius,
            minor_radius: self.minor_radius,
        })
    }
}

/// [`SolidOracle`] for a [`Torus`]: closed-form signed distance and closest-point
/// projection, both via the nearest point on the tube centre circle.
pub struct TorusOracle {
    center: [Scalar; D],
    axis: [Scalar; D],
    major_radius: Scalar,
    minor_radius: Scalar,
}

impl TorusOracle {
    /// The point on the tube centre circle nearest `query`, and the vector from
    /// it to `query` with that vector's length.
    fn tube_frame(&self, query: &Coordinate<D>) -> ([Scalar; D], [Scalar; D], Scalar) {
        let relative: [Scalar; D] = from_fn(|k| query[k].value() - self.center[k]);
        let axial = (0..D).map(|k| relative[k] * self.axis[k]).sum::<Scalar>();
        let radial: [Scalar; D] = from_fn(|k| relative[k] - axial * self.axis[k]);
        let rho = radial.iter().map(|x| x * x).sum::<Scalar>().sqrt();
        let radial_unit = if rho > 1.0e-30 {
            radial.map(|x| x / rho)
        } else {
            perpendicular(self.axis)
        };
        let ring: [Scalar; D] =
            from_fn(|k| self.center[k] + self.major_radius * radial_unit[k]);
        let offset: [Scalar; D] = from_fn(|k| query[k].value() - ring[k]);
        let distance = offset.iter().map(|x| x * x).sum::<Scalar>().sqrt();
        (ring, offset, distance)
    }
}

impl SolidOracle for TorusOracle {
    fn project(&self, query: &Coordinate<D>) -> Option<(Coordinate<D>, Direction<D>)> {
        let (ring, offset, distance) = self.tube_frame(query);
        let normal = if distance > 1.0e-30 {
            offset.map(|x| x / distance)
        } else {
            perpendicular(self.axis)
        };
        let point: Coordinate<D> = from_fn(|k| ring[k] + self.minor_radius * normal[k]).into();
        Some((point, Direction::const_from(normal)))
    }

    fn signed_distance(&self, query: &Coordinate<D>) -> Scalar {
        let (_, _, distance) = self.tube_frame(query);
        self.minor_radius - distance
    }
}

fn unit(v: [Scalar; D]) -> Option<[Scalar; D]> {
    let norm = v.iter().map(|x| x * x).sum::<Scalar>().sqrt();
    (norm > 0.0).then(|| v.map(|x| x / norm))
}

fn perpendicular(axis: [Scalar; D]) -> [Scalar; D] {
    let seed = if axis[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let dot = (0..D).map(|k| seed[k] * axis[k]).sum::<Scalar>();
    unit(from_fn(|k| seed[k] - dot * axis[k])).unwrap_or([1.0, 0.0, 0.0])
}
