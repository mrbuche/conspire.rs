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

/// A finite truncated cone (a frustum, or a pointed cone when `tip_radius` is
/// zero), meshed as a solid by the shared driver. Runs from `base` a distance
/// `height` along the unit `axis`.
pub struct Cone {
    base: [Scalar; D],
    axis: [Scalar; D],
    base_radius: Scalar,
    tip_radius: Scalar,
    height: Scalar,
}

impl Cone {
    /// The cone from a `base_radius` disk at `base` to a `tip_radius` disk
    /// `height` along `axis`. Radii must be non-negative with at least one
    /// positive, and `height` positive.
    pub fn new(
        base: Coordinate<D>,
        axis: Direction<D>,
        base_radius: Scalar,
        tip_radius: Scalar,
        height: Scalar,
    ) -> Result<Self, &'static str> {
        if base_radius < 0.0 || tip_radius < 0.0 || base_radius.max(tip_radius) <= 0.0 {
            return Err("cone radii must be non-negative with at least one positive");
        }
        if height <= 0.0 {
            return Err("cone height must be positive");
        }
        let axis = unit(from_fn(|k| axis[k].value())).ok_or("degenerate cone axis")?;
        Ok(Self {
            base: from_fn(|k| base[k].value()),
            axis,
            base_radius,
            tip_radius,
            height,
        })
    }
}

impl Solid for Cone {
    type Oracle = ConeOracle;

    fn bounding_box(&self) -> Result<(Coordinate<D>, Coordinate<D>), &'static str> {
        let tip: [Scalar; D] = from_fn(|k| self.base[k] + self.height * self.axis[k]);
        let spread = |k: usize| (1.0 - self.axis[k] * self.axis[k]).max(0.0).sqrt();
        let low: Coordinate<D> = from_fn(|k| {
            (self.base[k] - self.base_radius * spread(k)).min(tip[k] - self.tip_radius * spread(k))
        })
        .into();
        let high: Coordinate<D> = from_fn(|k| {
            (self.base[k] + self.base_radius * spread(k)).max(tip[k] + self.tip_radius * spread(k))
        })
        .into();
        Ok((low, high))
    }

    fn oracle(&self) -> Result<ConeOracle, &'static str> {
        Ok(ConeOracle {
            base: self.base,
            axis: self.axis,
            base_radius: self.base_radius,
            tip_radius: self.tip_radius,
            height: self.height,
        })
    }
}

/// [`SolidOracle`] for a [`Cone`]: closed-form signed distance (Quilez's capped
/// cone) and closest-point projection onto the lateral surface or an end cap.
pub struct ConeOracle {
    base: [Scalar; D],
    axis: [Scalar; D],
    base_radius: Scalar,
    tip_radius: Scalar,
    height: Scalar,
}

impl SolidOracle for ConeOracle {
    fn project(&self, query: &Coordinate<D>) -> Option<(Coordinate<D>, Direction<D>)> {
        let (h, radial, rho) = local(self.base, self.axis, query);
        let radial_unit = if rho > 1.0e-30 {
            radial.map(|x| x / rho)
        } else {
            perpendicular(self.axis)
        };
        let q = [rho, h];
        let lateral_normal = unit2([self.height, self.base_radius - self.tip_radius])?;
        let segments = [
            ([0.0, 0.0], [self.base_radius, 0.0], [0.0, -1.0]),
            (
                [self.base_radius, 0.0],
                [self.tip_radius, self.height],
                lateral_normal,
            ),
            (
                [self.tip_radius, self.height],
                [0.0, self.height],
                [0.0, 1.0],
            ),
        ];
        let (closest, normal_2d) = segments
            .into_iter()
            .map(|(a, b, normal_2d)| {
                let point = closest_on_segment(a, b, q);
                let distance = (q[0] - point[0]).powi(2) + (q[1] - point[1]).powi(2);
                ((point, normal_2d), distance)
            })
            .min_by(|x, y| x.1.total_cmp(&y.1))
            .map(|(pair, _)| pair)?;
        let point: Coordinate<D> =
            from_fn(|k| self.base[k] + closest[1] * self.axis[k] + closest[0] * radial_unit[k])
                .into();
        let normal = unit(from_fn(|k| {
            normal_2d[0] * radial_unit[k] + normal_2d[1] * self.axis[k]
        }))?;
        Some((point, Direction::const_from(normal)))
    }

    fn signed_distance(&self, query: &Coordinate<D>) -> Scalar {
        let (h, _, rho) = local(self.base, self.axis, query);
        let (r1, r2, half) = (self.base_radius, self.tip_radius, 0.5 * self.height);
        let y = h - half;
        let q = [rho, y];
        let k1 = [r2, half];
        let k2 = [r2 - r1, self.height];
        let near_radius = if y < 0.0 { r1 } else { r2 };
        let ca = [rho - rho.min(near_radius), y.abs() - half];
        let t = (dot2([k1[0] - q[0], k1[1] - q[1]], k2) / dot2(k2, k2)).clamp(0.0, 1.0);
        let cb = [q[0] - k1[0] + k2[0] * t, q[1] - k1[1] + k2[1] * t];
        let sign = if cb[0] < 0.0 && ca[1] < 0.0 {
            -1.0
        } else {
            1.0
        };
        -sign * dot2(ca, ca).min(dot2(cb, cb)).sqrt()
    }
}

/// `query` in cone coordinates: axial coordinate `h` from `base`, the radial
/// offset vector, and its length `rho`.
fn local(
    base: [Scalar; D],
    axis: [Scalar; D],
    query: &Coordinate<D>,
) -> (Scalar, [Scalar; D], Scalar) {
    let rel: [Scalar; D] = from_fn(|k| query[k].value() - base[k]);
    let h = (0..D).map(|k| rel[k] * axis[k]).sum::<Scalar>();
    let radial: [Scalar; D] = from_fn(|k| rel[k] - h * axis[k]);
    let rho = radial.iter().map(|x| x * x).sum::<Scalar>().sqrt();
    (h, radial, rho)
}

fn closest_on_segment(a: [Scalar; 2], b: [Scalar; 2], p: [Scalar; 2]) -> [Scalar; 2] {
    let d = [b[0] - a[0], b[1] - a[1]];
    let span = d[0] * d[0] + d[1] * d[1];
    let t = if span > 0.0 {
        (((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1]) / span).clamp(0.0, 1.0)
    } else {
        0.0
    };
    [a[0] + t * d[0], a[1] + t * d[1]]
}

fn dot2(a: [Scalar; 2], b: [Scalar; 2]) -> Scalar {
    a[0] * b[0] + a[1] * b[1]
}

fn unit(v: [Scalar; D]) -> Option<[Scalar; D]> {
    let norm = v.iter().map(|x| x * x).sum::<Scalar>().sqrt();
    (norm > 0.0).then(|| v.map(|x| x / norm))
}

fn unit2(v: [Scalar; 2]) -> Option<[Scalar; 2]> {
    let norm = (v[0] * v[0] + v[1] * v[1]).sqrt();
    (norm > 0.0).then(|| [v[0] / norm, v[1] / norm])
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
