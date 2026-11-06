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

/// How far from orthonormal the axes of an ellipsoid may be.
const ORTHONORMAL: f64 = 1e-9;

/// A triaxial ellipsoid.
pub struct Ellipsoid {
    center: Coordinate<3>,
    /// The unit directions of the principal semi-axes.
    axes: [Direction<3>; 3],
    /// The lengths of the semi-axes, one along each of the `axes`, which are all positive.
    semi: [Quantity<Length>; 3],
}

impl Ellipsoid {
    /// An ellipsoid with its semi-axes along the coordinate axes. Each must be positive.
    pub fn new(
        center: Coordinate<3>,
        semi_axes: [Quantity<Length>; 3],
    ) -> Result<Self, &'static str> {
        Self::oriented(
            center,
            [
                Direction::const_from([1.0, 0.0, 0.0]),
                Direction::const_from([0.0, 1.0, 0.0]),
                Direction::const_from([0.0, 0.0, 1.0]),
            ],
            semi_axes,
        )
    }
    /// An ellipsoid with its semi-axes along `axes`, which must be orthonormal, with the lengths
    /// `semi_axes`, each of which must be positive.
    pub fn oriented(
        center: Coordinate<3>,
        axes: [Direction<3>; 3],
        semi_axes: [Quantity<Length>; 3],
    ) -> Result<Self, &'static str> {
        if semi_axes.iter().any(|&semi| semi <= Quantity::default()) {
            return Err("ellipsoid semi-axes must be positive");
        }
        for i in 0..3 {
            for j in i..3 {
                let wanted = if i == j { 1.0 } else { 0.0 };
                if ((&axes[i] * &axes[j]).value() - wanted).abs() > ORTHONORMAL {
                    return Err("ellipsoid axes must be orthonormal");
                }
            }
        }
        Ok(Self {
            center,
            axes,
            semi: semi_axes,
        })
    }
    /// The surface point nearest a point, and the outward unit normal there, with whether the
    /// point is inside.
    fn nearest(&self, point: &Coordinate<3>) -> (Coordinate<3>, Direction<3>, bool) {
        let scale = self
            .semi
            .into_iter()
            .fold(Quantity::default(), Quantity::max);
        let offset = point - &self.center;
        let local: [Quantity<Length>; 3] = from_fn(|i| &offset * &self.axes[i]);
        let semi: [f64; 3] = from_fn(|i| (self.semi[i] / scale).value());
        let far: [f64; 3] = from_fn(|i| (local[i].abs() / scale).value());
        let inside = (0..3)
            .map(|i| (local[i] / self.semi[i]).value().powi(2))
            .sum::<f64>()
            <= 1.0;
        let closest = closest(semi, far);
        let sign = |i: usize| {
            if local[i] < Quantity::default() {
                -1.0
            } else {
                1.0
            }
        };
        let on_surface = (0..3).fold(self.center.clone(), |sum, i| {
            &sum + &(&self.axes[i] * (scale * (sign(i) * closest[i])))
        });
        let gradient = (0..3).fold(Direction::default(), |sum, i| {
            &sum + &(&self.axes[i] * (sign(i) * closest[i] / semi[i].powi(2)))
        });
        (on_surface, gradient.normalized(), inside)
    }
}

impl SolidOracle for Ellipsoid {
    fn signed_distance(&self, point: &Coordinate<3>) -> Quantity<Length> {
        let (on_surface, _, inside) = self.nearest(point);
        let distance = (point - &on_surface).norm();
        if inside { -distance } else { distance }
    }
    fn project(&self, point: &Coordinate<3>) -> Option<Projection> {
        let (point, normal, _) = self.nearest(point);
        Some(Projection {
            point,
            normal,
            exact: true,
        })
    }
}

/// The point of the ellipsoid with semi-axes `e` nearest the point with coordinates `y`, which
/// are not negative, in the frame of its axes and with the same signs dropped.
fn closest(e: [f64; 3], y: [f64; 3]) -> [f64; 3] {
    let mut order = [0, 1, 2];
    order.sort_by(|&a, &b| e[b].total_cmp(&e[a]));
    let found = sorted(order.map(|i| e[i]), order.map(|i| y[i]));
    let mut closest = [0.0; 3];
    order
        .iter()
        .zip(found)
        .for_each(|(&axis, value)| closest[axis] = value);
    closest
}

/// The same for semi-axes `e` from longest to shortest.
///
/// A point on the surface at `x` is the nearest to `y` when `x[i] = e[i]² y[i] / (t + e[i]²)` for
/// the one `t` that puts it on the surface, which is the greatest root of
/// `F(t) = Σ (e[i] y[i] / (t + e[i]²))² - 1`.[^1] That root is found by bisection, to the last bit
/// of `t`, so there is no tolerance to scale. It needs `y` to be positive along the shortest axis
/// of those it uses. Without that there may be no root, and the nearest point is off the plane of
/// the other axes.
///
/// [^1]: D. Eberly, [Distance from a Point to an Ellipse, an Ellipsoid, or a Hyperellipsoid](https://www.geometrictools.com/Documentation/DistancePointEllipseEllipsoid.pdf).
fn sorted(e: [f64; 3], y: [f64; 3]) -> [f64; 3] {
    if y[2] == 0.0 {
        let numerator = [e[0] * y[0], e[1] * y[1]];
        let denominator = [e[0] * e[0] - e[2] * e[2], e[1] * e[1] - e[2] * e[2]];
        let off_the_plane = (0..2).all(|i| y[i] == 0.0 || numerator[i] < denominator[i]);
        if off_the_plane {
            let ratio: [f64; 2] = from_fn(|i| {
                if y[i] == 0.0 {
                    0.0
                } else {
                    numerator[i] / denominator[i]
                }
            });
            let remainder = 1.0 - ratio[0] * ratio[0] - ratio[1] * ratio[1];
            if remainder > 0.0 {
                return [e[0] * ratio[0], e[1] * ratio[1], e[2] * remainder.sqrt()];
            }
        }
    }
    let used: Vec<usize> = (0..3).filter(|&i| y[i] > 0.0).collect();
    let Some(&shortest) = used.last() else {
        return [0.0, 0.0, e[2]];
    };
    let weighted = |i: usize| e[i] * y[i];
    let gap = |i: usize| e[i] * e[i] - e[shortest] * e[shortest];
    let f = |s: f64| -> f64 {
        used.iter()
            .map(|&i| (weighted(i) / (s + gap(i))).powi(2))
            .sum::<f64>()
            - 1.0
    };
    let mut low = weighted(shortest);
    let mut high = used
        .iter()
        .map(|&i| weighted(i).powi(2))
        .sum::<f64>()
        .sqrt();
    for _ in 0..2000 {
        let middle = 0.5 * (low + high);
        if middle <= low || middle >= high {
            break;
        }
        if f(middle) > 0.0 {
            low = middle;
        } else {
            high = middle;
        }
    }
    let s = 0.5 * (low + high);
    let mut x = [0.0; 3];
    for &i in &used {
        x[i] = e[i] * e[i] * y[i] / (s + gap(i));
    }
    x
}
