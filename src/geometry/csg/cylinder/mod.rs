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

/// A finite capped right circular cylinder, meshed as a solid by the shared
/// driver. Runs from `base` a distance `height` along the unit `axis`.
pub struct Cylinder {
    base: [Scalar; D],
    axis: [Scalar; D],
    radius: Scalar,
    height: Scalar,
}

impl Cylinder {
    /// The cylinder of `radius` from `base`, extending `height` along `axis`.
    /// `radius` and `height` must be positive.
    pub fn new(
        base: Coordinate<D>,
        axis: Direction<D>,
        radius: Scalar,
        height: Scalar,
    ) -> Result<Self, &'static str> {
        if radius <= 0.0 || height <= 0.0 {
            return Err("cylinder radius and height must be positive");
        }
        let axis = unit(from_fn(|k| axis[k].value())).ok_or("degenerate cylinder axis")?;
        Ok(Self {
            base: from_fn(|k| base[k].value()),
            axis,
            radius,
            height,
        })
    }
}

/// `query` in cylinder coordinates: axial coordinate `h` from `base`, the radial
/// offset vector, and its length `rho`.
fn local(base: [Scalar; D], axis: [Scalar; D], query: &Coordinate<D>) -> (Scalar, [Scalar; D], Scalar) {
    let rel: [Scalar; D] = from_fn(|k| query[k].value() - base[k]);
    let h = (0..D).map(|k| rel[k] * axis[k]).sum::<Scalar>();
    let radial: [Scalar; D] = from_fn(|k| rel[k] - h * axis[k]);
    let rho = radial.iter().map(|x| x * x).sum::<Scalar>().sqrt();
    (h, radial, rho)
}

impl Solid for Cylinder {
    type Oracle = CylinderOracle;

    fn bounding_box(&self) -> Result<(Coordinate<D>, Coordinate<D>), &'static str> {
        let tip: [Scalar; D] = from_fn(|k| self.base[k] + self.height * self.axis[k]);
        // Half-width of the swept disk along axis k.
        let disk = |k: usize| self.radius * (1.0 - self.axis[k] * self.axis[k]).max(0.0).sqrt();
        let low: Coordinate<D> = from_fn(|k| self.base[k].min(tip[k]) - disk(k)).into();
        let high: Coordinate<D> = from_fn(|k| self.base[k].max(tip[k]) + disk(k)).into();
        Ok((low, high))
    }

    fn oracle(&self) -> Result<CylinderOracle, &'static str> {
        Ok(CylinderOracle {
            base: self.base,
            axis: self.axis,
            radius: self.radius,
            height: self.height,
        })
    }
}

/// [`SolidOracle`] for a capped cylinder: closed-form signed distance and
/// closest-point projection onto the lateral surface or an end cap.
pub struct CylinderOracle {
    base: [Scalar; D],
    axis: [Scalar; D],
    radius: Scalar,
    height: Scalar,
}

impl CylinderOracle {
    fn point(&self, h: Scalar, radial_unit: [Scalar; D], rho: Scalar) -> Coordinate<D> {
        from_fn(|k| self.base[k] + h * self.axis[k] + rho * radial_unit[k]).into()
    }
}

impl SolidOracle for CylinderOracle {
    fn project(&self, query: &Coordinate<D>) -> Option<(Coordinate<D>, Direction<D>)> {
        let (h, radial, rho) = local(self.base, self.axis, query);
        let radial_unit = if rho > 1.0e-30 {
            radial.map(|x| x / rho)
        } else {
            perpendicular(self.axis)
        };
        let inside = rho <= self.radius && (0.0..=self.height).contains(&h);
        if inside {
            let to_side = self.radius - rho;
            let to_base = h;
            let to_tip = self.height - h;
            if to_side <= to_base && to_side <= to_tip {
                return Some((
                    self.point(h, radial_unit, self.radius),
                    Direction::const_from(radial_unit),
                ));
            }
            let (cap_h, sign) = if to_base <= to_tip {
                (0.0, -1.0)
            } else {
                (self.height, 1.0)
            };
            return Some((
                self.point(cap_h, radial_unit, rho),
                Direction::const_from(self.axis.map(|x| sign * x)),
            ));
        }
        // Outside: clamp onto the nearest surface point, normal toward `query`.
        let clamped_h = h.clamp(0.0, self.height);
        let clamped_rho = rho.min(self.radius);
        let closest = self.point(clamped_h, radial_unit, clamped_rho);
        let away: [Scalar; D] = from_fn(|k| query[k].value() - closest[k].value());
        let normal = unit(away).unwrap_or(radial_unit);
        Some((closest, Direction::const_from(normal)))
    }

    fn signed_distance(&self, query: &Coordinate<D>) -> Scalar {
        let (h, _, rho) = local(self.base, self.axis, query);
        let radial = rho - self.radius;
        let axial = (h - 0.5 * self.height).abs() - 0.5 * self.height;
        let outside = radial.max(0.0).hypot(axial.max(0.0));
        let interior = radial.max(axial).min(0.0);
        -(outside + interior)
    }
}

/// The unit vector along `v`, or `None` when `v` is degenerate.
fn unit(v: [Scalar; D]) -> Option<[Scalar; D]> {
    let norm = v.iter().map(|x| x * x).sum::<Scalar>().sqrt();
    (norm > 0.0).then(|| v.map(|x| x / norm))
}

/// Some unit vector orthogonal to the unit vector `axis`.
fn perpendicular(axis: [Scalar; D]) -> [Scalar; D] {
    let seed = if axis[0].abs() < 0.9 {
        [1.0, 0.0, 0.0]
    } else {
        [0.0, 1.0, 0.0]
    };
    let dot = (0..D).map(|k| seed[k] * axis[k]).sum::<Scalar>();
    unit(from_fn(|k| seed[k] - dot * axis[k])).unwrap_or([1.0, 0.0, 0.0])
}
