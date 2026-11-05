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

/// A piece of the outline of a cone: where it starts and ends, and its outward normal.
type Piece = ([Quantity<Length>; 2], [Quantity<Length>; 2], [f64; 2]);

/// A finite truncated cone, a frustum, or a pointed cone when the tip radius is zero. It runs from
/// its base a `height` along its axis.
pub struct Cone {
    axis: Axis,
    base_radius: Quantity<Length>,
    tip_radius: Quantity<Length>,
    height: Quantity<Length>,
}

impl Cone {
    /// The cone from a disk of `base_radius` at `base` to a disk of `tip_radius`, `height` along
    /// `axis`. The radii cannot be negative and cannot both be zero, `height` must be positive,
    /// and `axis` cannot be zero.
    pub fn new(
        base: Coordinate<3>,
        axis: Direction<3>,
        base_radius: Quantity<Length>,
        tip_radius: Quantity<Length>,
        height: Quantity<Length>,
    ) -> Result<Self, &'static str> {
        let zero = Quantity::default();
        if base_radius < zero || tip_radius < zero || base_radius.max(tip_radius) <= zero {
            return Err("cone radii must be non-negative with at least one positive");
        }
        if height <= zero {
            return Err("cone height must be positive");
        }
        Ok(Self {
            axis: Axis::new(base, axis)?,
            base_radius,
            tip_radius,
            height,
        })
    }
    /// The pieces of the outline of the cone in the half plane of distance from the axis and
    /// distance along it, from the base cap to the side to the tip cap, each with its outward
    /// normal in that plane.
    fn outline(&self) -> [Piece; 3] {
        let zero = Quantity::default();
        let slope = [
            self.height.value(),
            (self.base_radius - self.tip_radius).value(),
        ];
        let size = slope[0].hypot(slope[1]);
        [
            ([zero, zero], [self.base_radius, zero], [0.0, -1.0]),
            (
                [self.base_radius, zero],
                [self.tip_radius, self.height],
                [slope[0] / size, slope[1] / size],
            ),
            (
                [self.tip_radius, self.height],
                [zero, self.height],
                [0.0, 1.0],
            ),
        ]
    }
}

/// The point of the segment from `a` to `b` closest to `point`.
fn closest_on_segment(
    a: &Coordinate<2>,
    b: &Coordinate<2>,
    point: &Coordinate<2>,
) -> Coordinate<2> {
    let span = b - a;
    let length_squared = &span * &span;
    if length_squared <= Quantity::default() {
        return a.clone();
    }
    let t = (((point - a) * &span) / length_squared)
        .value()
        .clamp(0.0, 1.0);
    a + &(&span * t)
}

impl SolidOracle for Cone {
    fn signed_distance(&self, point: &Coordinate<3>) -> Quantity<Length> {
        let zero = Quantity::default();
        let (_, distance) = self.axis.across(point);
        let (base, tip, half) = (self.base_radius, self.tip_radius, self.height * 0.5);
        let height = self.axis.along(point) - half;
        let query = Coordinate::from([distance, height]);
        let k1 = Coordinate::from([tip, half]);
        let k2 = Coordinate::from([tip - base, self.height]);
        let near = if height < zero { base } else { tip };
        let flat = Coordinate::from([distance - distance.min(near), height.abs() - half]);
        let t = (((&k1 - &query) * &k2) / (&k2 * &k2))
            .value()
            .clamp(0.0, 1.0);
        let slant = &(&query - &k1) + &(&k2 * t);
        let nearest = flat.norm().min(slant.norm());
        if slant[0] < zero && flat[1] < zero {
            -nearest
        } else {
            nearest
        }
    }
    fn project(&self, point: &Coordinate<3>) -> Option<Projection> {
        let zero = Quantity::default();
        let (across, distance) = self.axis.across(point);
        let outward = self.axis.outward(&across, distance);
        let query = Coordinate::from([distance, self.axis.along(point)]);
        let (closest, normal) = self
            .outline()
            .into_iter()
            .map(|(a, b, normal)| {
                let closest = closest_on_segment(&a.into(), &b.into(), &query);
                ((&query - &closest).norm(), (closest, normal))
            })
            .min_by(|(a, _), (b, _)| a.total_cmp(b))
            .map(|(_, found)| found)?;
        let on_surface = self.axis.at(closest[1], &outward, closest[0]);
        let away = point - &on_surface;
        let normal = if self.signed_distance(point) > zero && away.norm() > zero {
            away.normalized()
        } else {
            &(&outward * normal[0]) + &(self.axis.direction() * normal[1])
        };
        Some(Projection {
            point: on_surface,
            normal,
            exact: true,
        })
    }
}
