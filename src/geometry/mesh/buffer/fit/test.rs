use super::{Oracle, crease_term, energy, nearest_on_polylines, scatter, touches_one_of};
use crate::math::assert::perturbation;
use crate::{
    EPSILON,
    geometry::{
        Coordinate, Coordinates, Direction,
        mesh::quality::metrics::{hexahedron, tetrahedron},
    },
    math::{
        Reference, TensorRank1,
        assert::{Assert, AssertionError},
    },
    units::ReciprocalLength,
};
use std::array::from_fn;

/// A stub [`Oracle`] whose `feature` always reports `face` (or nothing, for
/// `None`) regardless of the query -- `touches_one_of` only cares about the
/// reported id, not any real geometry.
struct StubOracle(Option<usize>);

impl Oracle for StubOracle {
    fn project(&self, _query: &Coordinate<3>) -> Option<(Coordinate<3>, Direction<3>)> {
        None
    }
    fn feature(&self, _query: &Coordinate<3>) -> Option<usize> {
        self.0
    }
}

fn one_face_setup() -> (Vec<Vec<usize>>, Vec<usize>, Coordinates<3>) {
    let coordinates = Coordinates::from(vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]);
    (vec![vec![0, 1, 2, 3]], vec![0], coordinates)
}

#[test]
fn touches_one_of_rejects_a_face_on_an_unrelated_surface() {
    let (faces, node_faces, coordinates) = one_face_setup();
    let oracle = StubOracle(Some(7));
    assert!(!touches_one_of(
        &oracle,
        &faces,
        &node_faces,
        &coordinates,
        &[3, 4]
    ));
}

#[test]
fn touches_one_of_accepts_a_face_on_a_bordering_surface() {
    let (faces, node_faces, coordinates) = one_face_setup();
    let oracle = StubOracle(Some(3));
    assert!(touches_one_of(
        &oracle,
        &faces,
        &node_faces,
        &coordinates,
        &[3, 4]
    ));
}

#[test]
fn touches_one_of_disengages_when_the_oracle_reports_no_features() {
    // An oracle with no concept of discrete regions (every CSG primitive, and
    // Facets -- a triangulated target has no per-face id) must not block
    // ownership; the gate only applies when it has something to say.
    let (faces, node_faces, coordinates) = one_face_setup();
    let oracle = StubOracle(None);
    assert!(touches_one_of(
        &oracle,
        &faces,
        &node_faces,
        &coordinates,
        &[3, 4]
    ));
}

#[test]
fn nearest_on_polylines_projects_onto_a_segment() {
    let curves = vec![vec![
        Coordinate::from([0.0; 3]),
        Coordinate::from([0.0, 0.0, 10.0]),
    ]];
    let (index, foot, distance, tangent) =
        nearest_on_polylines(&curves, &Coordinate::from([3.0, 4.0, 5.0])).unwrap();
    assert_eq!(index, 0);
    assert!(foot[0].value().abs() < 1.0e-9);
    assert!(foot[1].value().abs() < 1.0e-9);
    assert!((foot[2].value() - 5.0).abs() < 1.0e-9);
    assert!((distance - 5.0).abs() < 1.0e-9);
    assert!((tangent[2] - 1.0).abs() < 1.0e-9, "{tangent:?}");
}

#[test]
fn nearest_on_polylines_clamps_to_a_segment_end() {
    let curves = vec![vec![
        Coordinate::from([0.0; 3]),
        Coordinate::from([0.0, 0.0, 10.0]),
    ]];
    let (_, foot, distance, _) =
        nearest_on_polylines(&curves, &Coordinate::from([3.0, 0.0, -4.0])).unwrap();
    assert!(
        foot[2].value().abs() < 1.0e-9,
        "should clamp to the near end, not overshoot"
    );
    assert!((distance - 5.0).abs() < 1.0e-9);
}

#[test]
fn no_curves_projects_to_nothing() {
    assert!(nearest_on_polylines(&[], &Coordinate::from([0.0; 3])).is_none());
}

#[test]
fn nearest_on_polylines_reports_which_curve_it_landed_on() {
    let curves = vec![
        vec![
            Coordinate::from([0.0; 3]),
            Coordinate::from([0.0, 0.0, 10.0]),
        ],
        vec![
            Coordinate::from([1.0, 0.0, 0.0]),
            Coordinate::from([1.0, 0.0, 10.0]),
        ],
    ];
    let (index, ..) = nearest_on_polylines(&curves, &Coordinate::from([0.1, 0.0, 5.0])).unwrap();
    assert_eq!(index, 0);
    let (index, ..) = nearest_on_polylines(&curves, &Coordinate::from([0.9, 0.0, 5.0])).unwrap();
    assert_eq!(index, 1);
    // Restricting the search to a single-curve slice ignores the other curve
    // even when it is nearer -- what `crease_curve` freezing in Mesh::fit
    // relies on to keep a node from flipping targets between two close
    // creases (a thin flange's top and bottom rim) as it moves.
    let (index, ..) =
        nearest_on_polylines(&curves[0..1], &Coordinate::from([0.9, 0.0, 5.0])).unwrap();
    assert_eq!(index, 0);
}

#[test]
fn crease_term_is_zero_exactly_on_the_line() {
    let point = Coordinate::from([1.0, 2.0, 3.0]);
    let tangent = [0.0, 1.0, 0.0];
    let on_line = Coordinate::from([1.0, 5.0, 3.0]);
    let (squared, _) = crease_term(&on_line, &point, &tangent);
    assert!(squared.value() < 1.0e-9, "{squared:?}");
}

#[test]
fn crease_term_falls_back_to_point_attraction_when_the_tangent_is_degenerate() {
    let point = Coordinate::from([0.0; 3]);
    let x = Coordinate::from([3.0, 4.0, 0.0]);
    let (squared, perp) = crease_term(&x, &point, &[0.0; 3]);
    assert!((squared.value() - 25.0).abs() < 1.0e-9);
    assert!((perp[0] - 3.0).abs() < 1.0e-9 && (perp[1] - 4.0).abs() < 1.0e-9);
}

#[test]
fn crease_term_gradient_matches_finite_difference() {
    let point = Coordinate::from([0.2, 0.4, 0.7]);
    let tangent = [0.0, 0.0, 1.0];
    let x = [0.5, -0.3, 1.1];
    let squared_at = |x: [f64; 3]| {
        crease_term(&Coordinate::from(x), &point, &tangent)
            .0
            .value()
    };
    let h = 1.0e-6;
    for k in 0..3 {
        let mut plus = x;
        plus[k] += h;
        let mut minus = x;
        minus[k] -= h;
        let numerical = (squared_at(plus) - squared_at(minus)) / (2.0 * h);
        let (_, perp) = crease_term(&Coordinate::from(x), &point, &tangent);
        let analytic = 2.0 * perp[k];
        assert!(
            (numerical - analytic).abs() < 1.0e-6,
            "k={k}: {numerical} vs {analytic}"
        );
    }
}

fn gradient<const N: usize>(
    corners: &[(usize, [usize; 3]); N],
    mut coordinates: Coordinates<3>,
) -> Result<(), AssertionError> {
    let element: [usize; N] = from_fn(|i| i);
    for epsilon in [1.0, 1.0e-3] {
        let scattered = scatter(corners, &element, &coordinates, epsilon);
        for node in 0..N {
            let analytic = scattered[node].clone();
            let numerical = TensorRank1::<3, Reference, ReciprocalLength>::from(from_fn(|i| {
                coordinates[node][i] += perturbation(EPSILON);
                let above = energy(corners, &element, &coordinates, epsilon);
                coordinates[node][i] -= perturbation(2.0 * EPSILON);
                let below = energy(corners, &element, &coordinates, epsilon);
                coordinates[node][i] += perturbation(EPSILON);
                (above - below) / (2.0 * EPSILON)
            }));
            Assert::default().eq_within_fd_tol(analytic, &numerical)?;
        }
    }
    Ok(())
}

#[test]
fn gradient_matches_finite_difference() -> Result<(), AssertionError> {
    gradient(
        &hexahedron::CORNERS,
        Coordinates::from(vec![
            [0.03, -0.04, 0.01],
            [1.08, 0.05, -0.07],
            [1.02, 0.94, 0.11],
            [-0.06, 1.07, 0.02],
            [0.09, 0.01, 0.88],
            [0.94, -0.08, 1.04],
            [1.11, 1.03, 0.93],
            [0.05, 0.92, 1.09],
        ]),
    )
}

#[test]
fn tetrahedral_gradient_matches_finite_difference() -> Result<(), AssertionError> {
    gradient(
        &tetrahedron::CORNERS,
        Coordinates::from(vec![
            [0.03, -0.04, 0.01],
            [1.08, 0.05, -0.07],
            [0.02, 0.94, 0.11],
            [0.09, 0.01, 0.88],
        ]),
    )
}
