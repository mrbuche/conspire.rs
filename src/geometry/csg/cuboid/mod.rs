#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, Direction,
        solid::{Projection, SolidOracle},
    },
    math::{Quantity, Tensor},
    units::Length,
};
use core::array::from_fn;

/// An axis-aligned rectangular box.
pub struct Cuboid {
    low: Coordinate<3>,
    high: Coordinate<3>,
}

impl Cuboid {
    /// The box spanning `low` to `high`; every `low[k]` must be strictly below `high[k]`.
    pub fn new(low: Coordinate<3>, high: Coordinate<3>) -> Result<Self, &'static str> {
        if (0..3).any(|axis| low[axis] >= high[axis]) {
            return Err("cuboid low corner must be strictly below the high corner");
        }
        Ok(Self { low, high })
    }
}

impl SolidOracle for Cuboid {
    fn signed_distance(&self, point: &Coordinate<3>) -> Quantity<Length> {
        let zero = Quantity::default();
        let past: [Quantity<Length>; 3] =
            from_fn(|axis| (self.low[axis] - point[axis]).max(point[axis] - self.high[axis]));
        let outside: Coordinate<3> = past.map(|distance| distance.max(zero)).into();
        let inside = past
            .into_iter()
            .fold(Quantity::new(f64::NEG_INFINITY), Quantity::max);
        outside.norm() + inside.min(zero)
    }
    fn project(&self, point: &Coordinate<3>) -> Option<Projection> {
        let zero = Quantity::default();
        let clamped: Coordinate<3> =
            from_fn(|axis| point[axis].max(self.low[axis]).min(self.high[axis])).into();
        let away = point - &clamped;
        if away.norm() > zero {
            return Some(Projection {
                point: clamped,
                normal: away.normalized(),
                exact: true,
            });
        }
        let (mut axis, mut sign, mut best) = (0, 1.0, Quantity::new(f64::INFINITY));
        for k in 0..3 {
            if point[k] - self.low[k] < best {
                (axis, sign, best) = (k, -1.0, point[k] - self.low[k]);
            }
            if self.high[k] - point[k] < best {
                (axis, sign, best) = (k, 1.0, self.high[k] - point[k]);
            }
        }
        let mut on_face = point.clone();
        on_face[axis] = if sign < 0.0 {
            self.low[axis]
        } else {
            self.high[axis]
        };
        let mut normal = [0.0; 3];
        normal[axis] = sign;
        Some(Projection {
            point: on_face,
            normal: Direction::const_from(normal),
            exact: true,
        })
    }
}
