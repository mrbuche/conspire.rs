#[cfg(test)]
mod test;

use super::axis::Axis;
use crate::{
    geometry::{
        Coordinate, Direction,
        solid::{Projection, SolidOracle},
    },
    math::{Quantity, Tensor},
    units::Length,
};

/// A finite capped right circular cylinder, from its base a `height` along its axis.
pub struct Cylinder {
    axis: Axis,
    radius: Quantity<Length>,
    height: Quantity<Length>,
}

impl Cylinder {
    /// The cylinder of `radius` from the centre of its `base`, extending `height` along `axis`.
    /// `radius` and `height` must be positive, and `axis` cannot be zero.
    pub fn new(
        base: Coordinate<3>,
        axis: Direction<3>,
        radius: Quantity<Length>,
        height: Quantity<Length>,
    ) -> Result<Self, &'static str> {
        let zero = Quantity::default();
        if radius <= zero || height <= zero {
            return Err("cylinder radius and height must be positive");
        }
        Ok(Self {
            axis: Axis::new(base, axis)?,
            radius,
            height,
        })
    }
}

impl SolidOracle for Cylinder {
    fn signed_distance(&self, point: &Coordinate<3>) -> Quantity<Length> {
        let zero = Quantity::default();
        let (_, distance) = self.axis.across(point);
        let radial = distance - self.radius;
        let half = self.height * 0.5;
        let axial = (self.axis.along(point) - half).abs() - half;
        let outside = Coordinate::<2>::from([radial.max(zero), axial.max(zero)]).norm();
        outside + radial.max(axial).min(zero)
    }
    fn project(&self, point: &Coordinate<3>) -> Option<Projection> {
        let zero = Quantity::default();
        let along = self.axis.along(point);
        let (across, distance) = self.axis.across(point);
        let outward = self.axis.outward(&across, distance);
        let inside = distance <= self.radius && along >= zero && along <= self.height;
        if inside {
            let to_side = self.radius - distance;
            let (to_base, to_tip) = (along, self.height - along);
            if to_side <= to_base && to_side <= to_tip {
                return Some(Projection {
                    point: self.axis.at(along, &outward, self.radius),
                    normal: outward,
                    exact: true,
                });
            }
            let (cap, sign) = if to_base <= to_tip {
                (zero, -1.0)
            } else {
                (self.height, 1.0)
            };
            return Some(Projection {
                point: self.axis.at(cap, &outward, distance),
                normal: self.axis.direction() * sign,
                exact: true,
            });
        }
        let closest = self.axis.at(
            along.max(zero).min(self.height),
            &outward,
            distance.min(self.radius),
        );
        let away = point - &closest;
        let normal = if away.norm() > zero {
            away.normalized()
        } else {
            outward
        };
        Some(Projection {
            point: closest,
            normal,
            exact: true,
        })
    }
}
