use super::Partition;
use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{
            Connectivity, Mesh, Tessellation,
            test::{CONNECTIVITY, COORDINATES},
        },
    },
    math::{Quantity, Tensor},
};
use std::array::from_fn;

fn cuboid(minimum: [f64; 3], maximum: [f64; 3]) -> Tessellation {
    let coordinates = COORDINATES
        .iter()
        .map(|corner| {
            from_fn(|axis| minimum[axis] + (maximum[axis] - minimum[axis]) * corner[axis].value())
                .into()
        })
        .collect();
    Mesh::from((
        vec![Connectivity::Triangular(CONNECTIVITY.to_vec().into())],
        coordinates,
    ))
    .into()
}

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
