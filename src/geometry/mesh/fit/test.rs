use super::{Oracle, energy, scatter};
use crate::math::assert::perturbation;
use crate::{
    EPSILON,
    geometry::{
        Coordinates,
        mesh::{
            quality::metrics::{hexahedron, tetrahedron},
            test::octahedron,
        },
    },
    math::{
        Reference, TensorRank1,
        assert::{Assert, AssertionError},
    },
    units::ReciprocalLength,
};
use std::array::from_fn;

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

fn targets(queries: &[[f64; 3]]) -> Vec<([f64; 3], [f64; 3])> {
    let tessellation = octahedron(0);
    let oracle = Oracle::new(&tessellation);
    let offsets = [
        [0.01, 0.0, 0.0],
        [-0.01, 0.0, 0.0],
        [0.0, 0.01, 0.0],
        [0.0, -0.01, 0.0],
    ];
    let coordinates = Coordinates::from(
        queries
            .iter()
            .flat_map(|query| offsets.map(|offset| from_fn(|i| query[i] + offset[i])))
            .collect::<Vec<[f64; 3]>>(),
    );
    let faces: Vec<Vec<usize>> = (0..queries.len())
        .map(|query| (4 * query..4 * query + 4).collect())
        .collect();
    oracle
        .targets(&faces, &coordinates, 1)
        .unwrap()
        .into_iter()
        .map(|(point, normal, _)| {
            (
                from_fn(|i| point[i].value()),
                from_fn(|i| normal[i].value()),
            )
        })
        .collect()
}

fn close(a: [f64; 3], b: [f64; 3]) -> bool {
    (0..3).all(|i| (a[i] - b[i]).abs() < 1.0e-12)
}

#[test]
fn equidistant_faces_average_to_a_symmetric_target() {
    let (point, normal) = targets(&[[0.0, 0.2, 0.2]])[0];
    let unit = 0.5_f64.sqrt();
    assert!(close(point, [0.0, 0.4, 0.4]), "{point:?}");
    assert!(close(normal, [0.0, unit, unit]), "{normal:?}");
}

#[test]
fn shared_edge_averages_the_face_normals() {
    let (point, normal) = targets(&[[0.0, 0.8, 0.8]])[0];
    let unit = 0.5_f64.sqrt();
    assert!(close(point, [0.0, 0.5, 0.5]), "{point:?}");
    assert!(close(normal, [0.0, unit, unit]), "{normal:?}");
}

#[test]
fn unique_nearest_face_keeps_its_normal() {
    let (point, normal) = targets(&[[0.3, 0.2, 0.2]])[0];
    let unit = 3.0_f64.sqrt().recip();
    assert!(close(point, [0.4, 0.3, 0.3]), "{point:?}");
    assert!(close(normal, [unit, unit, unit]), "{normal:?}");
}

#[test]
fn targets_are_mirror_symmetric() {
    let queries = [
        [0.0, 0.2, 0.2],
        [0.0, 0.8, 0.8],
        [0.3, 0.2, 0.2],
        [0.0, 0.0, 0.9],
    ];
    let expected = targets(&queries);
    for axis in 0..3 {
        let mirrored: Vec<[f64; 3]> = queries
            .iter()
            .map(|query| {
                let mut query = *query;
                query[axis] = -query[axis];
                query
            })
            .collect();
        for ((point, normal), (reflected_point, reflected_normal)) in
            expected.iter().zip(targets(&mirrored))
        {
            let (mut point, mut normal) = (*point, *normal);
            point[axis] = -point[axis];
            normal[axis] = -normal[axis];
            assert!(
                close(point, reflected_point),
                "{point:?} {reflected_point:?}"
            );
            assert!(
                close(normal, reflected_normal),
                "{normal:?} {reflected_normal:?}"
            );
        }
    }
}
