#[cfg(test)]
mod test;

use crate::geometry::{Coordinate, Direction};

const D: usize = 3;

pub enum Curve {
    Line(Line),
    Circle(Circle),
    Ellipse(Ellipse),
    BSpline(BSpline),
}

pub struct Line {
    pub origin: Coordinate<D>,
    pub direction: Direction<D>,
}

pub struct Circle {
    pub center: Coordinate<D>,
    pub axis: Direction<D>,
    pub reference_direction: Direction<D>,
    pub radius: f64,
}

pub struct Ellipse {
    pub center: Coordinate<D>,
    /// The plane normal.
    pub axis: Direction<D>,
    /// The major-axis direction.
    pub reference_direction: Direction<D>,
    pub major_radius: f64,
    pub minor_radius: f64,
}

/// The 3D chord polyline replacing `curve` from `start` to `end`, both
/// included, refined until no chord strays further from the curve than
/// `RELATIVE_SAGITTA` of its own length.
///
/// Every trim that cannot carry an edge in closed form calls this — the
/// planar frame and the spherical/toroidal charts alike — so two faces
/// sharing an edge chord it into the *same* polyline. Sampling them
/// independently leaves their trimmed regions disagreeing by the difference
/// of two approximations, a crack a ray-parity test falls straight through.
pub(in crate::geometry::cad) fn chords(
    curve: &Curve,
    start: &Coordinate<D>,
    end: &Coordinate<D>,
    forward: bool,
    closed: bool,
) -> Vec<Coordinate<D>> {
    /// Chord deviation allowed, as a fraction of the edge's own length — well
    /// under anything the mesh resolves, and since every face chords a shared
    /// edge through this same function they agree exactly regardless.
    const RELATIVE_SAGITTA: f64 = 1.0e-3;
    const FIRST: usize = 17;
    const CAP: usize = 513;

    // The reader orients a circle/ellipse axis so the *edge* runs CCW about
    // it, and `arc_polyline` always takes the positive turn; a half-edge
    // walked backwards has to turn about the negated axis, or the chord goes
    // the long way round.
    let sense = if forward { 1.0 } else { -1.0 };
    let raw = |point: &Coordinate<D>| std::array::from_fn::<f64, D, _>(|k| point[k].value());
    let sample = |count: usize| -> Vec<Coordinate<D>> {
        match curve {
            Curve::Line(_) => vec![start.clone(), end.clone()],
            Curve::BSpline(bspline) => bspline.segment(start, end, count),
            Curve::Circle(circle) => crate::geometry::cad::sizing::arc_polyline(
                raw(&circle.center),
                std::array::from_fn(|k| sense * circle.axis[k].value()),
                std::array::from_fn(|k| circle.reference_direction[k].value()),
                circle.radius,
                circle.radius,
                raw(start),
                raw(end),
                closed,
                count - 1,
            ),
            Curve::Ellipse(ellipse) => crate::geometry::cad::sizing::arc_polyline(
                raw(&ellipse.center),
                std::array::from_fn(|k| sense * ellipse.axis[k].value()),
                std::array::from_fn(|k| ellipse.reference_direction[k].value()),
                ellipse.major_radius,
                ellipse.minor_radius,
                raw(start),
                raw(end),
                closed,
                count - 1,
            ),
        }
    };
    if matches!(curve, Curve::Line(_)) {
        return sample(2);
    }
    let mut count = FIRST;
    loop {
        let points = sample(count);
        // Halving again would put a sample at each chord's midpoint; the
        // deviation there bounds how far this polyline strays.
        let length: f64 = points
            .windows(2)
            .map(|pair| distance(&pair[0], &pair[1]))
            .sum();
        if count >= CAP || length == 0.0 {
            return points;
        }
        let refined = sample(count * 2 - 1);
        let worst = refined
            .windows(3)
            .step_by(2)
            .map(|triple| point_segment_distance(&triple[1], &triple[0], &triple[2]))
            .fold(0.0, f64::max);
        if worst <= length * RELATIVE_SAGITTA {
            return refined;
        }
        count = count * 2 - 1;
    }
}

fn distance(a: &Coordinate<D>, b: &Coordinate<D>) -> f64 {
    (0..D).map(|k| (a[k].value() - b[k].value()).powi(2)).sum::<f64>().sqrt()
}

/// Distance from `p` to the segment `a`-`b`.
fn point_segment_distance(p: &Coordinate<D>, a: &Coordinate<D>, b: &Coordinate<D>) -> f64 {
    let ab = std::array::from_fn::<f64, D, _>(|k| b[k].value() - a[k].value());
    let ap = std::array::from_fn::<f64, D, _>(|k| p[k].value() - a[k].value());
    let span = (0..D).map(|k| ab[k] * ab[k]).sum::<f64>();
    let t = if span > 0.0 {
        ((0..D).map(|k| ap[k] * ab[k]).sum::<f64>() / span).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (0..D).map(|k| (ap[k] - t * ab[k]).powi(2)).sum::<f64>().sqrt()
}

/// A B-spline (or, when `weights` is set, NURBS) curve, with the knots stored
/// compressed as STEP writes them: distinct `knots` and their
/// `multiplicities`.
pub struct BSpline {
    pub degree: usize,
    pub control_points: Vec<Coordinate<D>>,
    pub knots: Vec<f64>,
    pub multiplicities: Vec<usize>,
    pub weights: Option<Vec<f64>>,
}

impl BSpline {
    /// The knots expanded to one entry per multiplicity.
    fn knot_vector(&self) -> Vec<f64> {
        self.knots
            .iter()
            .zip(self.multiplicities.iter())
            .flat_map(|(&knot, &multiplicity)| std::iter::repeat_n(knot, multiplicity))
            .collect()
    }

    /// The valid parameter range `[U[p], U[n]]` of a clamped curve.
    pub fn span(&self) -> (f64, f64) {
        let knots = self.knot_vector();
        let degree = self.degree.min(self.control_points.len().saturating_sub(1));
        let last = self.control_points.len().max(degree).min(knots.len() - 1);
        (knots[degree.min(last)], knots[last])
    }

    /// De Boor evaluation at `t`, in homogeneous coordinates when rational.
    pub fn point(&self, t: f64) -> Coordinate<D> {
        let points = &self.control_points;
        let knots = self.knot_vector();
        let count = points.len();
        assert!(count > 0, "B-spline has no control points");
        let degree = self.degree.min(count - 1);
        assert!(
            knots.len() > count + degree,
            "B-spline knot vector is too short"
        );
        let (low, high) = self.span();
        let t = t.clamp(low, high);
        let mut span = degree;
        while span + 1 < count && knots[span + 1] <= t {
            span += 1;
        }
        let weight = |i: usize| self.weights.as_ref().map_or(1.0, |w| w[i]);
        let mut work: Vec<[f64; D + 1]> = (0..=degree)
            .map(|j| {
                let i = j + span - degree;
                let w = weight(i);
                let mut point = [0.0; D + 1];
                (0..D).for_each(|k| point[k] = points[i][k].value() * w);
                point[D] = w;
                point
            })
            .collect();
        for r in 1..=degree {
            for j in (r..=degree).rev() {
                let i = j + span - degree;
                let (lower, upper) = (knots[i], knots[i + degree + 1 - r]);
                let alpha = if upper > lower {
                    (t - lower) / (upper - lower)
                } else {
                    0.0
                };
                let previous = work[j - 1];
                work[j]
                    .iter_mut()
                    .zip(previous)
                    .for_each(|(entry, before)| *entry = (1.0 - alpha) * before + alpha * *entry);
            }
        }
        let homogeneous = work[degree];
        let w = if homogeneous[D].abs() > f64::EPSILON {
            homogeneous[D]
        } else {
            1.0
        };
        Coordinate::from(std::array::from_fn::<_, D, _>(|k| homogeneous[k] / w))
    }

    /// `samples` points evenly spaced in the parameter over the whole span.
    pub fn polyline(&self, samples: usize) -> Vec<Coordinate<D>> {
        let (low, high) = self.span();
        self.sample(low, high, samples)
    }

    /// `samples` points evenly spaced in the parameter from `low` to `high`,
    /// both included.
    fn sample(&self, low: f64, high: f64, samples: usize) -> Vec<Coordinate<D>> {
        let samples = samples.max(2);
        (0..samples)
            .map(|i| self.point(low + (high - low) * i as f64 / (samples - 1) as f64))
            .collect()
    }

    /// The parameter whose point is nearest `target`: a dense scan of the span
    /// followed by a ternary-search refinement of the winning bracket.
    fn nearest_parameter(&self, target: &Coordinate<D>) -> f64 {
        let distance = |t: f64| {
            let point = self.point(t);
            (0..D)
                .map(|k| (point[k].value() - target[k].value()).powi(2))
                .sum::<f64>()
        };
        let (low, high) = self.span();
        const SCAN: usize = 128;
        let step = (high - low) / SCAN as f64;
        let mut best = low;
        let mut best_distance = f64::INFINITY;
        for i in 0..=SCAN {
            let t = low + step * i as f64;
            let d = distance(t);
            if d < best_distance {
                best_distance = d;
                best = t;
            }
        }
        let (mut a, mut b) = ((best - step).max(low), (best + step).min(high));
        for _ in 0..40 {
            let (left, right) = (a + (b - a) / 3.0, b - (b - a) / 3.0);
            if distance(left) < distance(right) {
                b = right;
            } else {
                a = left;
            }
        }
        (a + b) / 2.0
    }

    /// The chord polyline of the piece of this curve running from `start` to
    /// `end`, `samples` points including both, with the endpoints replaced by
    /// the exact vertices. Falls back to the whole span when the two vertices
    /// coincide (a closed edge) or land on the same parameter.
    pub fn segment(
        &self,
        start: &Coordinate<D>,
        end: &Coordinate<D>,
        samples: usize,
    ) -> Vec<Coordinate<D>> {
        let (low, high) = self.span();
        let (mut first, mut last) = (self.nearest_parameter(start), self.nearest_parameter(end));
        if (last - first).abs() <= (high - low) * 1.0e-9 {
            (first, last) = (low, high);
        }
        let mut points = self.sample(first, last, samples);
        *points.first_mut().expect("sample yields two or more") = start.clone();
        *points.last_mut().expect("sample yields two or more") = end.clone();
        points
    }
}
