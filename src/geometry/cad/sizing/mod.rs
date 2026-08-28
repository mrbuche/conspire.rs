//! Volume sizing fields derived from B-rep geometry.

#[cfg(test)]
mod test;

use crate::{
    geometry::{Coordinate, cad::brep::Brep, solid::Sizing},
    math::{Quantity, Scalar, Tensor},
    units::{Dimensionless, Length},
};

const D: usize = 3;

/// A scalar sizing field driven by B-rep edge lengths.
///
/// Every edge contributes a target size of its length over `segments_per_edge`
/// at its location; away from the edge that size grows at rate `gradation`.
/// The field is the clamped minimum of those contributions, so it is defined
/// everywhere and already satisfies the gradation constraint.
pub struct FeatureSizing {
    segments: Vec<[Coordinate<D>; 2]>,
    sizes: Vec<Quantity<Length>>,
    minimum: Quantity<Length>,
    maximum: Quantity<Length>,
    gradation: Scalar,
}

impl FeatureSizing {
    pub fn of(
        brep: &Brep,
        segments_per_edge: usize,
        minimum: Quantity<Length>,
        maximum: Quantity<Length>,
        gradation: Scalar,
    ) -> Self {
        let divisor = segments_per_edge.max(1) as Scalar;
        let (segments, sizes) = brep
            .edges
            .iter()
            .map(|edge| {
                let [a, b] = edge.vertices;
                let segment = [brep.vertices[a].clone(), brep.vertices[b].clone()];
                let size = ((&segment[1] - &segment[0]).norm() / divisor)
                    .max(minimum)
                    .min(maximum);
                (segment, size)
            })
            .unzip();
        Self {
            segments,
            sizes,
            minimum,
            maximum,
            gradation,
        }
    }

    /// The target element size at `point`.
    pub fn at(&self, point: &Coordinate<D>) -> Quantity<Length> {
        let mut size = self.maximum;
        for (segment, source) in self.segments.iter().zip(&self.sizes) {
            let candidate = *source + distance(point, segment) * self.gradation;
            if candidate < size {
                size = candidate;
            }
        }
        size.max(self.minimum).min(self.maximum)
    }
}

impl Sizing for FeatureSizing {
    fn at(&self, point: &Coordinate<D>) -> Quantity<Length> {
        FeatureSizing::at(self, point)
    }
}

/// Distance from `point` to the closest point of the segment.
fn distance(point: &Coordinate<D>, segment: &[Coordinate<D>; 2]) -> Quantity<Length> {
    let along = &segment[1] - &segment[0];
    let length = &along * &along;
    let closest = if length > Quantity::new(0.0) {
        let fraction = Quantity::<Dimensionless>::new(
            ((point - &segment[0]) * &along / length)
                .value()
                .clamp(0.0, 1.0),
        );
        &segment[0] + &(along * fraction)
    } else {
        segment[0].clone()
    };
    (point - &closest).norm()
}
