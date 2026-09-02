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
/// A principal coordinate this close to zero (relative to its semi-axis) is
/// treated as exactly zero: the closest point sits on that plane by symmetry,
/// so the axis is dropped and the lower-D problem solved instead. Keeps the
/// bisection root away from the `-e_min²` bracket end, where an
/// epsilon-nudged coordinate left it unresolved.
const AXIS_EPSILON: Scalar = 1.0e-9;

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
    /// `p`, by bisecting Eberly's `F(t) = Σ (eᵢ pᵢ / (t + eᵢ²))² − 1`. A
    /// principal coordinate within [`AXIS_EPSILON`] of zero is dropped and the
    /// remaining-axes problem solved, with that coordinate held at zero.
    fn closest_local(&self, p: [Scalar; D]) -> [Scalar; D] {
        let e = self.semi;
        let shortest_axis = || (0..D).min_by(|&a, &b| e[a].total_cmp(&e[b])).unwrap();

        let normalized: Scalar = (0..D).map(|i| (p[i] / e[i]).powi(2)).sum();
        let mut closest = [0.0; D];
        if normalized < TOLERANCE {
            // At the centre the projection is the tip of the shortest semi-axis.
            closest[shortest_axis()] = e[shortest_axis()];
            return closest;
        }

        let active: Vec<usize> = (0..D)
            .filter(|&i| p[i].abs() > AXIS_EPSILON * e[i].max(1.0))
            .collect();
        if active.is_empty() {
            closest[shortest_axis()] = e[shortest_axis()];
            return closest;
        }

        let es: Vec<Scalar> = active.iter().map(|&i| e[i]).collect();
        let ys: Vec<Scalar> = active.iter().map(|&i| p[i].abs()).collect();
        let sub_normalized: Scalar = active.iter().map(|&i| (p[i] / e[i]).powi(2)).sum();
        let t = eberly_root(&es, &ys, sub_normalized);
        for (slot, &i) in active.iter().enumerate() {
            let sign = if p[i] < 0.0 { -1.0 } else { 1.0 };
            closest[i] = sign * es[slot] * es[slot] * ys[slot] / (t + es[slot] * es[slot]);
        }
        closest
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

/// Bisects Eberly's `F(t) = Σ (eᵢ yᵢ / (t + eᵢ²))² − 1` over the given axes
/// (`y` = `|p|` there, all strictly positive) for the closest-point parameter
/// `t`. `normalized = Σ (yᵢ / eᵢ)²` selects the interior (`≤ 1`) or exterior
/// bracket.
fn eberly_root(e: &[Scalar], y: &[Scalar], normalized: Scalar) -> Scalar {
    let f = |t: Scalar| -> Scalar {
        e.iter()
            .zip(y)
            .map(|(&e, &y)| (e * y / (t + e * e)).powi(2))
            .sum::<Scalar>()
            - 1.0
    };
    let smallest = e.iter().map(|&e| e * e).fold(Scalar::INFINITY, Scalar::min);
    let (mut lo, mut hi) = if normalized <= 1.0 {
        (TOLERANCE - smallest, 0.0)
    } else {
        (
            0.0,
            e.iter().zip(y).map(|(&e, &y)| (e * y).powi(2)).sum::<Scalar>().sqrt(),
        )
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
    0.5 * (lo + hi)
}

fn dot(a: [Scalar; D], b: [Scalar; D]) -> Scalar {
    (0..D).map(|k| a[k] * b[k]).sum()
}

fn unit(v: [Scalar; D]) -> Option<[Scalar; D]> {
    let norm = v.iter().map(|x| x * x).sum::<Scalar>().sqrt();
    (norm > 0.0).then(|| v.map(|x| x / norm))
}
