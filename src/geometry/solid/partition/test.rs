use super::Partition;
use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{Tessellation, test::cuboid},
    },
    math::{
        Quantity, Tensor,
        assert::{Assert, AssertionError},
    },
    units::Length,
};

fn inside(point: &Coordinate<3>, minimum: [f64; 3], maximum: [f64; 3]) -> bool {
    (0..3).all(|axis| minimum[axis] < point[axis].value() && point[axis].value() < maximum[axis])
}

fn grid() -> Coordinates<3> {
    let axis: Vec<f64> = (0..9)
        .map(|i| -0.43 + 0.2 * i as f64 + 0.011 * i as f64)
        .collect();
    let mut points = Vec::new();
    for &x in &axis {
        for &y in &axis {
            for &z in &axis {
                points.push([x, y, z]);
            }
        }
    }
    points.into()
}

fn exactly_one_owner(
    partition: &Partition<Tessellation>,
    expected: impl Fn(&Coordinate<3>) -> Option<usize>,
) {
    grid().iter().for_each(|point| {
        let owners: Vec<usize> = (0..partition.number_of_volumes())
            .filter(|&index| partition.signed_distance(point, index) < Quantity::default())
            .collect();
        assert_eq!(owners.len(), expected(point).iter().count());
        assert_eq!(owners.first().copied(), expected(point));
        assert_eq!(partition.owner(point), expected(point))
    })
}

#[test]
fn nested_volume_takes_priority() {
    let (core_min, core_max) = ([0.3; 3], [0.7; 3]);
    let partition = Partition::new(vec![cuboid(core_min, core_max), cuboid([0.0; 3], [1.0; 3])]);
    exactly_one_owner(&partition, |point| {
        if inside(point, core_min, core_max) {
            Some(0)
        } else if inside(point, [0.0; 3], [1.0; 3]) {
            Some(1)
        } else {
            None
        }
    })
}

#[test]
fn overlap_goes_to_the_earlier_volume() {
    let (first, second) = (([0.0; 3], [0.7; 3]), ([0.3; 3], [1.0; 3]));
    let partition = Partition::new(vec![cuboid(first.0, first.1), cuboid(second.0, second.1)]);
    exactly_one_owner(&partition, |point| {
        if inside(point, first.0, first.1) {
            Some(0)
        } else if inside(point, second.0, second.1) {
            Some(1)
        } else {
            None
        }
    })
}

#[test]
fn priority_order_decides_the_overlap() {
    let core = ([0.3; 3], [0.7; 3]);
    let partition = Partition::new(vec![cuboid([0.0; 3], [1.0; 3]), cuboid(core.0, core.1)]);
    let center = [0.5; 3].into();
    assert_eq!(partition.owner(&center), Some(0));
    assert!(partition.signed_distance(&center, 1) > Quantity::default());
}

#[test]
fn gaps_are_owned_by_nobody() {
    let partition = Partition::new(vec![cuboid([0.0; 3], [1.0; 3]), cuboid([2.0; 3], [3.0; 3])]);
    assert_eq!(partition.owner(&[1.5; 3].into()), None);
    assert_eq!(partition.owner(&[2.5; 3].into()), Some(1));
}

fn projected_points(
    partition: &Partition<Tessellation>,
    index: usize,
) -> Vec<(Coordinate<3>, Coordinate<3>)> {
    grid()
        .into_iter()
        .filter_map(|point| {
            partition
                .project(&point, index)
                .map(|(closest, _)| (point, closest))
        })
        .collect()
}

#[test]
fn projection_lands_on_the_region_boundary() -> Result<(), AssertionError> {
    [
        ([0.3; 3], [0.7; 3], [0.0; 3], [1.0; 3]),
        ([0.0; 3], [0.7; 3], [0.3; 3], [1.0; 3]),
    ]
    .into_iter()
    .try_for_each(|(first_min, first_max, second_min, second_max)| {
        let partition = Partition::new(vec![
            cuboid(first_min, first_max),
            cuboid(second_min, second_max),
        ]);
        (0..2).try_for_each(|index| {
            projected_points(&partition, index)
                .iter()
                .try_for_each(|(_, closest)| {
                    Assert::default().eq_within_tols(
                        partition.signed_distance(closest, index),
                        &Length::meters(0.0),
                    )
                })
        })
    })
}

#[test]
fn projection_is_withheld_at_creases() {
    let partition = Partition::new(vec![cuboid([0.0; 3], [0.7; 3]), cuboid([0.3; 3], [1.0; 3])]);
    let found = projected_points(&partition, 1).len();
    assert!(0 < found && found < grid().len());
}

#[test]
fn projection_distance_is_the_effective_distance_when_nested() -> Result<(), AssertionError> {
    let partition = Partition::new(vec![cuboid([0.3; 3], [0.7; 3]), cuboid([0.0; 3], [1.0; 3])]);
    let projected = projected_points(&partition, 1);
    assert_eq!(projected.len(), grid().len());
    projected.iter().try_for_each(|(point, closest)| {
        Assert::default().eq_within_tols(
            (closest - point).norm(),
            &partition.signed_distance(point, 1).abs(),
        )
    })
}

#[test]
fn projection_is_never_nearer_than_the_effective_distance() {
    let partition = Partition::new(vec![cuboid([0.0; 3], [0.7; 3]), cuboid([0.3; 3], [1.0; 3])]);
    (0..2).for_each(|index| {
        projected_points(&partition, index)
            .iter()
            .for_each(|(point, closest)| {
                let slack = Length::meters(1e-12);
                assert!(
                    (closest - point).norm() + slack
                        >= partition.signed_distance(point, index).abs()
                )
            })
    });
}

fn crease_partition() -> Partition<Tessellation> {
    Partition::new(vec![cuboid([0.0; 3], [0.7; 3]), cuboid([0.3; 3], [1.0; 3])])
}

const CREASE_TOLERANCE: Quantity<Length> = Length::meters(1e-9);

fn hidden_points(partition: &Partition<Tessellation>) -> Vec<Coordinate<3>> {
    grid()
        .into_iter()
        .filter(|point| partition.project(point, 1).is_none())
        .collect()
}

#[test]
fn converged_crease_points_lie_on_both_surfaces() -> Result<(), AssertionError> {
    let partition = crease_partition();
    let hidden = hidden_points(&partition);
    let converged: Vec<(Coordinate<3>, Coordinate<3>)> = hidden
        .iter()
        .filter_map(|point| {
            partition
                .project_to_crease(point, 1, CREASE_TOLERANCE, 100)
                .ok()
                .map(|closest| (point.clone(), closest))
        })
        .collect();
    assert!(2 * converged.len() > hidden.len());
    converged.iter().try_for_each(|(point, closest)| {
        Assert::default()
            .eq_within_tols(partition.signed_distance(closest, 1), &Length::meters(0.0))?;
        Assert::default()
            .eq_within_tols(partition.signed_distance(closest, 0), &Length::meters(0.0))?;
        assert!(
            (closest - point).norm() + Length::meters(1e-12)
                >= partition.signed_distance(point, 1).abs()
        );
        Ok(())
    })
}

#[test]
fn crease_projection_agrees_with_projection_where_exact() {
    let partition = crease_partition();
    grid().iter().for_each(|point| {
        if let Some((exact, _)) = partition.project(point, 1) {
            assert_eq!(
                partition
                    .project_to_crease(point, 1, CREASE_TOLERANCE, 100)
                    .unwrap(),
                exact
            )
        }
    })
}

#[test]
fn crease_projection_reports_non_convergence() {
    let partition = crease_partition();
    hidden_points(&partition).iter().for_each(|point| {
        assert!(
            partition
                .project_to_crease(point, 1, CREASE_TOLERANCE, 0)
                .is_err()
        )
    })
}

fn bisected(
    partition: &Partition<Tessellation>,
    point: &Coordinate<3>,
) -> Result<Coordinate<3>, &'static str> {
    partition.boundary_on_segment(&Coordinate::from([0.9; 3]), point, 1, CREASE_TOLERANCE)
}

#[test]
fn bisection_lands_within_tolerance_of_the_boundary() {
    let partition = crease_partition();
    let outside: Vec<Coordinate<3>> = grid()
        .iter()
        .filter(|point| partition.signed_distance(point, 1) > Quantity::default())
        .cloned()
        .collect();
    assert!(!outside.is_empty());
    outside.iter().for_each(|point| {
        let found = bisected(&partition, point).unwrap();
        let distance = partition.signed_distance(&found, 1);
        assert!(distance <= Quantity::default());
        assert!(distance + CREASE_TOLERANCE > Quantity::default());
    })
}

#[test]
fn bisection_reaches_the_points_the_projections_cannot() {
    let partition = crease_partition();
    let failed: Vec<Coordinate<3>> = hidden_points(&partition)
        .into_iter()
        .filter(|point| {
            partition
                .project_to_crease(point, 1, CREASE_TOLERANCE, 100)
                .is_err()
        })
        .collect();
    assert!(!failed.is_empty());
    failed
        .iter()
        .for_each(|point| assert!(bisected(&partition, point).is_ok()))
}

#[test]
fn bisection_requires_the_ends_to_straddle_the_boundary() {
    let partition = crease_partition();
    let (inside, outside) = ([0.9; 3].into(), [-0.5; 3].into());
    assert!(
        partition
            .boundary_on_segment(&outside, &inside, 1, CREASE_TOLERANCE)
            .is_err()
    );
    assert!(
        partition
            .boundary_on_segment(&inside, &inside, 1, CREASE_TOLERANCE)
            .is_err()
    );
    assert!(
        partition
            .boundary_on_segment(&inside, &outside, 1, Length::meters(0.0))
            .is_err()
    );
}

#[test]
fn bisection_terminates_on_a_coincident_interface() {
    let partition = shared_face_partition();
    let found = partition
        .boundary_on_segment(
            &[1.5, 0.5, 0.5].into(),
            &[0.5, 0.5, 0.5].into(),
            1,
            CREASE_TOLERANCE,
        )
        .unwrap();
    assert!((found[0].value() - 1.0).abs() < 1e-8);
}

fn shared_face_partition() -> Partition<Tessellation> {
    Partition::new(vec![
        cuboid([0.0; 3], [1.0; 3]),
        cuboid([1.0, 0.0, 0.0], [2.0, 1.0, 1.0]),
    ])
}

const ACROSS: [f64; 7] = [0.3, 0.9, 0.999, 1.0, 1.001, 1.1, 1.7];
const ALONG: [f64; 3] = [0.2, 0.5, 0.8];

fn across_the_shared_face() -> Coordinates<3> {
    let mut points = Vec::new();
    for &x in &ACROSS {
        for &y in &ALONG {
            for &z in &ALONG {
                points.push([x, y, z]);
            }
        }
    }
    points.into()
}

#[test]
fn coincident_face_owner_is_consistent_with_the_regions() {
    let partition = shared_face_partition();
    across_the_shared_face().iter().for_each(|point| {
        let x = point[0].value();
        let owner = partition.owner(point).unwrap();
        if x != 1.0 {
            assert_eq!(owner, if x < 1.0 { 0 } else { 1 });
        }
        assert!(partition.signed_distance(point, owner) <= Quantity::default());
        (0..owner).for_each(|earlier| {
            assert!(partition.signed_distance(point, earlier) > Quantity::default())
        })
    })
}

#[test]
fn coincident_face_is_a_zero_of_both_regions() -> Result<(), AssertionError> {
    let partition = shared_face_partition();
    across_the_shared_face()
        .iter()
        .filter(|point| point[0].value() == 1.0)
        .try_for_each(|point| {
            (0..2).try_for_each(|index| {
                Assert::default().eq_within_tols(
                    partition.signed_distance(point, index),
                    &Length::meters(0.0),
                )
            })
        })
}

#[test]
fn coincident_face_projects_onto_the_shared_plane() -> Result<(), AssertionError> {
    let partition = shared_face_partition();
    across_the_shared_face()
        .iter()
        .filter(|point| (point[0].value() - 1.0).abs() <= 0.1)
        .try_for_each(|point| {
            (0..2).try_for_each(|index| {
                let (closest, _) = partition.project(point, index).unwrap();
                Assert::default().eq_within_tols(closest[0], &Length::meters(1.0))?;
                Assert::default().eq_within_tols(
                    (&closest - point).norm(),
                    &Length::meters((point[0].value() - 1.0).abs()),
                )
            })
        })
}

#[test]
fn region_normal_is_reversed_on_the_wall_of_an_earlier_volume() -> Result<(), AssertionError> {
    let partition = Partition::new(vec![cuboid([0.3; 3], [0.7; 3]), cuboid([0.0; 3], [1.0; 3])]);
    let (_, cavity) = partition.project(&[0.5, 0.5, 0.25].into(), 1).unwrap();
    Assert::default().eq_within_tols(&cavity, &[0.0, 0.0, 1.0].into())?;
    let (_, outer) = partition.project(&[0.5, 0.5, 0.05].into(), 1).unwrap();
    Assert::default().eq_within_tols(&outer, &[0.0, 0.0, -1.0].into())
}
