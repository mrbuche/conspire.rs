//! Volume sizing fields derived from B-rep geometry.

#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate,
        cad::brep::{Brep, Edge, curve::Curve, oracle::BrepOracle},
        solid::Sizing,
    },
    math::{Quantity, Scalar, Tensor},
    units::{Dimensionless, Length},
};
use std::{array::from_fn, f64::consts::TAU};

const D: usize = 3;

/// The local-feature-size term: cap the target at the local through-dimension
/// (wall thickness / cavity width) over `cells`, so thin walls and narrow
/// cavities get that many elements across.
struct Proximity {
    oracle: BrepOracle,
    cells: Scalar,
}

/// A scalar sizing field driven by B-rep feature edges.
///
/// Only sharp edges ([`Brep::features`] creases) contribute — seams and other
/// parameterization artifacts are ignored. A curved edge is sampled into a
/// chord polyline so the whole rim drives refinement, not just its endpoints.
/// Each contributing chord carries a target size of the edge's arc length over
/// `segments_per_edge`. `gradation` is the rate that size may grow per unit
/// distance from the edge (the mesh-smoothness bound); `Some(0.0)` pins the
/// whole field to the finest feature size, `None` lets it grow as fast as it
/// likes — the feature size within one target-size of the edge, `maximum`
/// beyond, a single fine layer per feature.
///
/// [`with_proximity`](Self::with_proximity) adds a local-feature-size term for
/// thin geometry the crease term is blind to (a thin wall or a small cavity
/// need not be near any sharp edge). The field is the clamped minimum of all
/// contributions, so it is defined everywhere.
pub struct FeatureSizing {
    segments: Vec<[Coordinate<D>; 2]>,
    sizes: Vec<Quantity<Length>>,
    minimum: Quantity<Length>,
    maximum: Quantity<Length>,
    gradation: Option<Scalar>,
    proximity: Option<Proximity>,
}

impl FeatureSizing {
    pub fn of(
        brep: &Brep,
        segments_per_edge: usize,
        minimum: Quantity<Length>,
        maximum: Quantity<Length>,
        gradation: Option<Scalar>,
    ) -> Self {
        let samples = segments_per_edge.max(1);
        let divisor = samples as Scalar;
        let mut segments = Vec::new();
        let mut sizes = Vec::new();
        for &edge in &brep.features().creases {
            let polyline = sample_edge(brep, &brep.edges[edge], samples);
            let length: Scalar = polyline
                .windows(2)
                .map(|pair| (&pair[1] - &pair[0]).norm().value())
                .sum();
            if length <= 0.0 {
                continue;
            }
            let size = Quantity::<Length>::new(length / divisor)
                .max(minimum)
                .min(maximum);
            for pair in polyline.windows(2) {
                segments.push([pair[0].clone(), pair[1].clone()]);
                sizes.push(size);
            }
        }
        Self {
            segments,
            sizes,
            minimum,
            maximum,
            gradation,
            proximity: None,
        }
    }

    /// Adds a local-feature-size term: at each point the target is also capped
    /// at the local through-dimension — the wall thickness or cavity width the
    /// point sits in — over `cells_across`, so a thin wall or a narrow cavity
    /// that the crease term (which sees only distance to a sharp edge) misses
    /// still gets `cells_across` elements across it.
    pub fn with_proximity(
        mut self,
        brep: &Brep,
        cells_across: usize,
    ) -> Result<Self, &'static str> {
        self.proximity = Some(Proximity {
            oracle: brep.oracle()?,
            cells: cells_across.max(1) as Scalar,
        });
        Ok(self)
    }

    /// The target element size at `point`.
    pub fn at(&self, point: &Coordinate<D>) -> Quantity<Length> {
        let mut size = self.maximum;
        for (segment, source) in self.segments.iter().zip(&self.sizes) {
            let reach = distance(point, segment);
            let candidate = match self.gradation {
                Some(rate) => *source + reach * rate,
                None if reach <= *source => *source,
                None => self.maximum,
            };
            if candidate < size {
                size = candidate;
            }
        }
        if let Some(proximity) = &self.proximity {
            let candidate = Quantity::<Length>::new(
                proximity.oracle.local_diameter(point) / proximity.cells,
            );
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

/// A chord polyline tracing `edge` between its two vertices: the arc for a
/// circle or ellipse (the full loop when the endpoints coincide), otherwise the
/// straight segment.
fn sample_edge(brep: &Brep, edge: &Edge, samples: usize) -> Vec<Coordinate<D>> {
    let [ia, ib] = edge.vertices;
    let (a, b) = (&brep.vertices[ia], &brep.vertices[ib]);
    match &edge.curve {
        Curve::Circle(circle) => arc_polyline(
            point(&circle.center),
            axis(&circle.axis),
            axis(&circle.reference_direction),
            circle.radius,
            circle.radius,
            point(a),
            point(b),
            ia == ib,
            samples,
        ),
        Curve::Ellipse(ellipse) => arc_polyline(
            point(&ellipse.center),
            axis(&ellipse.axis),
            axis(&ellipse.reference_direction),
            ellipse.major_radius,
            ellipse.minor_radius,
            point(a),
            point(b),
            ia == ib,
            samples,
        ),
        Curve::Line(_) | Curve::BSpline(_) => vec![a.clone(), b.clone()],
    }
}

#[expect(clippy::too_many_arguments)]
fn arc_polyline(
    centre: [Scalar; D],
    normal: [Scalar; D],
    reference: [Scalar; D],
    major: Scalar,
    minor: Scalar,
    start: [Scalar; D],
    end: [Scalar; D],
    closed: bool,
    samples: usize,
) -> Vec<Coordinate<D>> {
    let u = normalize(sub(reference, project(reference, normal)));
    let w = cross(normal, u);
    let angle = |p: [Scalar; D]| {
        let rel = sub(p, centre);
        (dot(rel, w) / minor.max(1.0e-12)).atan2(dot(rel, u) / major.max(1.0e-12))
    };
    let start_angle = angle(start);
    let mut sweep = angle(end) - start_angle;
    if closed || sweep.abs() < 1.0e-9 {
        sweep = TAU;
    } else {
        sweep -= TAU * (sweep / TAU).round(); // shortest arc, into (-pi, pi]
    }
    (0..=samples)
        .map(|i| {
            let theta = start_angle + sweep * (i as Scalar) / (samples as Scalar);
            let (c, s) = (theta.cos(), theta.sin());
            Coordinate::from(from_fn(|k| {
                centre[k] + major * c * u[k] + minor * s * w[k]
            }))
        })
        .collect()
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

fn point(coordinate: &Coordinate<D>) -> [Scalar; D] {
    from_fn(|k| coordinate[k].value())
}

fn axis(direction: &crate::geometry::Direction<D>) -> [Scalar; D] {
    from_fn(|k| direction[k].value())
}

fn sub(a: [Scalar; D], b: [Scalar; D]) -> [Scalar; D] {
    from_fn(|k| a[k] - b[k])
}

fn dot(a: [Scalar; D], b: [Scalar; D]) -> Scalar {
    (0..D).map(|k| a[k] * b[k]).sum()
}

fn cross(a: [Scalar; D], b: [Scalar; D]) -> [Scalar; D] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn project(v: [Scalar; D], onto: [Scalar; D]) -> [Scalar; D] {
    let scale = dot(v, onto) / dot(onto, onto).max(1.0e-30);
    from_fn(|k| scale * onto[k])
}

fn normalize(v: [Scalar; D]) -> [Scalar; D] {
    let norm = dot(v, v).sqrt();
    if norm > 1.0e-30 {
        v.map(|x| x / norm)
    } else {
        [1.0, 0.0, 0.0]
    }
}
