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
const TOLERANCE: Scalar = 1.0e-12;
const ITERATIONS: usize = 100;

/// A triaxial ellipsoid, meshed as a solid by the shared driver.
pub struct Ellipsoid {
    center: [Scalar; D],
    /// Row `i` is the unit world direction of the `i`-th principal semi-axis.
    axes: [[Scalar; D]; D],
    semi: [Scalar; D],
}

impl Ellipsoid {
    /// An axis-aligned ellipsoid with the given half-widths; each must be
    /// positive.
    pub fn new(center: Coordinate<D>, semi_axes: [Scalar; D]) -> Result<Self, &'static str> {
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

    /// An ellipsoid whose principal axes point along `axes` (assumed
    /// orthonormal), with half-widths `semi_axes`; each half-width must be
    /// positive.
    pub fn oriented(
        center: Coordinate<D>,
        axes: [Direction<D>; D],
        semi_axes: [Scalar; D],
    ) -> Result<Self, &'static str> {
        if semi_axes.iter().any(|&s| s <= 0.0) {
            return Err("ellipsoid semi-axes must be positive");
        }
        Ok(Self {
            center: from_fn(|k| center[k].value()),
            axes: from_fn(|i| from_fn(|k| axes[i][k].value())),
            semi: semi_axes,
        })
    }
}

impl Solid for Ellipsoid {
    type Oracle = EllipsoidOracle;

    fn bounding_box(&self) -> Result<(Coordinate<D>, Coordinate<D>), &'static str> {
        // Half-width along world axis k is the ellipsoid's support there.
        let half: [Scalar; D] = from_fn(|k| {
            (0..D)
                .map(|i| (self.semi[i] * self.axes[i][k]).powi(2))
                .sum::<Scalar>()
                .sqrt()
        });
        let low: Coordinate<D> = from_fn(|k| self.center[k] - half[k]).into();
        let high: Coordinate<D> = from_fn(|k| self.center[k] + half[k]).into();
        Ok((low, high))
    }

    fn oracle(&self) -> Result<EllipsoidOracle, &'static str> {
        Ok(EllipsoidOracle {
            center: self.center,
            axes: self.axes,
            semi: self.semi,
        })
    }
}

/// [`SolidOracle`] for an [`Ellipsoid`]: signed distance and closest-point
/// projection via a 1-D root find on the canonical ellipsoid.
pub struct EllipsoidOracle {
    center: [Scalar; D],
    axes: [[Scalar; D]; D],
    semi: [Scalar; D],
}

impl EllipsoidOracle {
    /// `query` in the principal frame, measured from the centre.
    fn local(&self, query: &Coordinate<D>) -> [Scalar; D] {
        let relative: [Scalar; D] = from_fn(|k| query[k].value() - self.center[k]);
        from_fn(|i| dot(relative, self.axes[i]))
    }

    /// The closest point of the canonical ellipsoid to the frame-local point
    /// `p`, by bisecting Eberly's `F(t) = Σ (eᵢ pᵢ / (t + eᵢ²))² − 1`.
    fn closest_local(&self, p: [Scalar; D]) -> [Scalar; D] {
        let e = self.semi;
        let normalized: Scalar = (0..D).map(|i| (p[i] / e[i]).powi(2)).sum();
        if normalized < TOLERANCE {
            // At the centre the projection is the tip of the shortest semi-axis.
            let shortest = (0..D).min_by(|&a, &b| e[a].total_cmp(&e[b])).unwrap();
            let mut closest = [0.0; D];
            closest[shortest] = e[shortest];
            return closest;
        }
        let sign: [Scalar; D] = from_fn(|i| if p[i] < 0.0 { -1.0 } else { 1.0 });
        let y: [Scalar; D] = from_fn(|i| p[i].abs().max(TOLERANCE));

        let f = |t: Scalar| -> Scalar {
            (0..D)
                .map(|i| (e[i] * y[i] / (t + e[i] * e[i])).powi(2))
                .sum::<Scalar>()
                - 1.0
        };
        let smallest = (0..D).map(|i| e[i] * e[i]).fold(Scalar::INFINITY, Scalar::min);
        let (mut lo, mut hi) = if normalized <= 1.0 {
            (TOLERANCE - smallest, 0.0)
        } else {
            (0.0, (0..D).map(|i| (e[i] * y[i]).powi(2)).sum::<Scalar>().sqrt())
        };
        for _ in 0..ITERATIONS {
            if hi - lo <= TOLERANCE * (1.0 + hi.abs()) {
                break;
            }
            let mid = 0.5 * (lo + hi);
            if f(mid) > 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let t = 0.5 * (lo + hi);
        from_fn(|i| sign[i] * e[i] * e[i] * y[i] / (t + e[i] * e[i]))
    }

    fn to_world(&self, local: [Scalar; D]) -> [Scalar; D] {
        from_fn(|k| self.center[k] + (0..D).map(|i| local[i] * self.axes[i][k]).sum::<Scalar>())
    }
}

impl SolidOracle for EllipsoidOracle {
    fn project(&self, query: &Coordinate<D>) -> Option<(Coordinate<D>, Direction<D>)> {
        let local = self.local(query);
        let closest = self.closest_local(local);
        let point: Coordinate<D> = self.to_world(closest).into();
        let gradient_local: [Scalar; D] = from_fn(|i| closest[i] / (self.semi[i] * self.semi[i]));
        let gradient: [Scalar; D] =
            from_fn(|k| (0..D).map(|i| gradient_local[i] * self.axes[i][k]).sum());
        let normal = unit(gradient)?;
        Some((point, Direction::const_from(normal)))
    }

    fn signed_distance(&self, query: &Coordinate<D>) -> Scalar {
        let local = self.local(query);
        let closest = self.closest_local(local);
        let distance = (0..D)
            .map(|i| (local[i] - closest[i]).powi(2))
            .sum::<Scalar>()
            .sqrt();
        let inside = (0..D)
            .map(|i| (local[i] / self.semi[i]).powi(2))
            .sum::<Scalar>()
            <= 1.0;
        if inside {
            distance
        } else {
            -distance
        }
    }
}

fn dot(a: [Scalar; D], b: [Scalar; D]) -> Scalar {
    (0..D).map(|k| a[k] * b[k]).sum()
}

fn unit(v: [Scalar; D]) -> Option<[Scalar; D]> {
    let norm = v.iter().map(|x| x * x).sum::<Scalar>().sqrt();
    (norm > 0.0).then(|| v.map(|x| x / norm))
}
