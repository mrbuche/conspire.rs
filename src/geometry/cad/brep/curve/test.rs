use super::{BSpline, D};
use crate::geometry::Coordinate;

fn points(raw: &[[f64; D]]) -> Vec<Coordinate<D>> {
    raw.iter().map(|&p| Coordinate::from(p)).collect()
}

/// A degree-2 B-spline on `[0,0,0,1,1,1]` is the quadratic Bezier of its three
/// control points: `B(t) = (1-t)^2 P0 + 2t(1-t) P1 + t^2 P2`.
#[test]
fn de_boor_matches_quadratic_bezier() {
    let control = [[0.0, 0.0, 0.0], [1.0, 2.0, -1.0], [2.0, 0.0, 3.0]];
    let curve = BSpline {
        degree: 2,
        control_points: points(&control),
        knots: vec![0.0, 1.0],
        multiplicities: vec![3, 3],
        weights: None,
    };
    assert_eq!(curve.span(), (0.0, 1.0));
    for step in 0..=10 {
        let t = step as f64 / 10.0;
        let bernstein = [(1.0 - t).powi(2), 2.0 * t * (1.0 - t), t * t];
        let point = curve.point(t);
        for k in 0..D {
            let expected: f64 = (0..3).map(|i| bernstein[i] * control[i][k]).sum();
            assert!((point[k].value() - expected).abs() < 1.0e-12);
        }
    }
}

/// The standard rational quarter circle: degree 2, weights `[1, 1/sqrt(2), 1]`
/// over the corner control polygon.
#[test]
fn rational_quarter_circle_lies_on_the_circle() {
    let curve = BSpline {
        degree: 2,
        control_points: points(&[[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
        knots: vec![0.0, 1.0],
        multiplicities: vec![3, 3],
        weights: Some(vec![1.0, 0.5_f64.sqrt(), 1.0]),
    };
    for point in curve.polyline(17) {
        let radius = (0..D).map(|k| point[k].value().powi(2)).sum::<f64>().sqrt();
        assert!((radius - 1.0).abs() < 1.0e-9, "radius {radius}");
        assert!(point[2].value().abs() < 1.0e-12);
    }
    let middle = curve.point(0.5);
    assert!((middle[0].value() - 0.5_f64.sqrt()).abs() < 1.0e-12);
    assert!((middle[1].value() - 0.5_f64.sqrt()).abs() < 1.0e-12);
}

/// A multi-span curve: the interior knot must place the parameter in the right
/// span, and the clamped ends must interpolate the first/last control points.
#[test]
fn de_boor_spans_and_clamped_ends() {
    let control = [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, -1.0, 0.0],
        [3.0, 0.0, 0.0],
    ];
    let curve = BSpline {
        degree: 2,
        control_points: points(&control),
        knots: vec![0.0, 0.5, 1.0],
        multiplicities: vec![3, 1, 3],
        weights: None,
    };
    assert_eq!(curve.span(), (0.0, 1.0));
    for (t, expected) in [(0.0, control[0]), (1.0, control[3])] {
        let point = curve.point(t);
        (0..D).for_each(|k| assert!((point[k].value() - expected[k]).abs() < 1.0e-12));
    }
    // At the interior knot the curve is at the midpoint of P1..P2's leg.
    let knot = curve.point(0.5);
    assert!((knot[0].value() - 1.5).abs() < 1.0e-12);
    assert!(knot[1].value().abs() < 1.0e-12);
}

/// `segment` restricts to the sub-range between two points on the curve, in
/// the order given, and pins the exact endpoints.
#[test]
fn segment_restricts_and_reverses() {
    let curve = BSpline {
        degree: 2,
        control_points: points(&[[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [2.0, 0.0, 0.0]]),
        knots: vec![0.0, 1.0],
        multiplicities: vec![3, 3],
        weights: None,
    };
    let quarter = curve.point(0.25);
    let three = curve.point(0.75);
    let forward = curve.segment(&quarter, &three, 9);
    assert_eq!(forward.len(), 9);
    for k in 0..D {
        assert!((forward[0][k].value() - quarter[k].value()).abs() < 1.0e-12);
        assert!((forward[8][k].value() - three[k].value()).abs() < 1.0e-12);
    }
    // Monotone in x over this control polygon, so a genuine restriction.
    assert!(forward.windows(2).all(|w| w[0][0].value() < w[1][0].value()));
    let backward = curve.segment(&three, &quarter, 9);
    assert!(backward.windows(2).all(|w| w[0][0].value() > w[1][0].value()));
    // A closed edge (same vertex twice) falls back to the whole span.
    let whole = curve.segment(&quarter, &quarter, 5);
    assert!((whole[2][0].value() - 1.0).abs() < 1.0e-9);
}
