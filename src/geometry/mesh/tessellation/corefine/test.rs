use super::{
    Feature, Key, Triangle,
    pierce::{Meeting, crossing as crossing_ends},
};
use crate::{
    geometry::{
        Coordinate,
        bvh::closest_point_on_triangle,
        mesh::test::{cuboid, solid},
        solid::SolidOracle,
    },
    math::{
        Tensor,
        assert::{Assert, AssertionError},
    },
    units::Length,
};
use core::array::from_fn;

fn crossing(first: Triangle, second: Triangle) -> Result<Option<[Coordinate<3>; 2]>, &'static str> {
    Ok(match crossing_ends(first, second)? {
        Meeting::Segment(ends) => Some(ends.map(|end| end.point)),
        _ => None,
    })
}

fn point(x: f64, y: f64, z: f64) -> Coordinate<3> {
    [x, y, z].into()
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn triangle(&mut self) -> [Coordinate<3>; 3] {
        from_fn(|_| point(self.next(), self.next(), self.next()))
    }
}

fn refs(triangle: &[Coordinate<3>; 3]) -> Triangle<'_> {
    [&triangle[0], &triangle[1], &triangle[2]]
}

fn flat() -> [Coordinate<3>; 3] {
    [
        point(0.0, 0.0, 0.0),
        point(4.0, 0.0, 0.0),
        point(0.0, 4.0, 0.0),
    ]
}

fn upright() -> [Coordinate<3>; 3] {
    [
        point(1.0, -1.0, -1.0),
        point(1.0, 5.0, -1.0),
        point(1.0, 2.0, 3.0),
    ]
}

fn sorted(segment: [Coordinate<3>; 2]) -> [Coordinate<3>; 2] {
    let [a, b] = segment;
    if a[1] <= b[1] { [a, b] } else { [b, a] }
}

#[test]
fn perpendicular_triangles_cross_in_the_overlap_of_their_chords() -> Result<(), AssertionError> {
    let (flat, upright) = (flat(), upright());
    let segment = sorted(crossing(refs(&flat), refs(&upright))?.unwrap());
    Assert::default().eq_within_tols(&segment[0], &point(1.0, 0.0, 0.0))?;
    Assert::default().eq_within_tols(&segment[1], &point(1.0, 3.0, 0.0))
}

#[test]
fn crossing_does_not_depend_on_the_order() -> Result<(), AssertionError> {
    let (flat, upright) = (flat(), upright());
    let segment = sorted(crossing(refs(&upright), refs(&flat))?.unwrap());
    Assert::default().eq_within_tols(&segment[0], &point(1.0, 0.0, 0.0))?;
    Assert::default().eq_within_tols(&segment[1], &point(1.0, 3.0, 0.0))
}

#[test]
fn separated_parallel_and_touching_triangles_do_not_cross() {
    let flat = flat();
    let far = [
        point(5.0, -1.0, -1.0),
        point(5.0, 5.0, -1.0),
        point(5.0, 2.0, 3.0),
    ];
    let lifted = [
        point(0.0, 0.0, 1.0),
        point(4.0, 0.0, 1.0),
        point(0.0, 4.0, 1.0),
    ];
    let touching = [
        point(1.0, 1.0, 0.0),
        point(1.0, 1.0, 2.0),
        point(1.0, 2.0, 2.0),
    ];
    [far, lifted, touching]
        .iter()
        .for_each(|other| assert!(crossing(refs(&flat), refs(other)).unwrap().is_none()))
}

#[test]
fn coplanar_triangles_are_reported_and_degenerate_ones_are_errors() {
    let flat = flat();
    let coplanar = [
        point(1.0, 1.0, 0.0),
        point(2.0, 1.0, 0.0),
        point(1.0, 2.0, 0.0),
    ];
    let collinear = [
        point(0.0, 0.0, -1.0),
        point(1.0, 1.0, 1.0),
        point(2.0, 2.0, 3.0),
    ];
    assert!(matches!(
        crossing_ends(refs(&flat), refs(&coplanar)),
        Ok(Meeting::Coplanar)
    ));
    assert!(crossing_ends(refs(&flat), refs(&collinear)).is_err());
    assert!(crossing_ends(refs(&collinear), refs(&flat)).is_err());
}

#[test]
fn random_crossings_lie_on_both_triangles_in_either_order() {
    let mut random = Lcg(7);
    let slack = Length::meters(1e-12);
    let (mut crossed, mut missed) = (0, 0);
    (0..4000).for_each(|_| {
        let (first, second) = (random.triangle(), random.triangle());
        let forward = crossing(refs(&first), refs(&second)).unwrap();
        let backward = crossing(refs(&second), refs(&first)).unwrap();
        assert_eq!(forward.is_some(), backward.is_some());
        match forward {
            None => missed += 1,
            Some(segment) => {
                crossed += 1;
                segment.iter().for_each(|end| {
                    [&first, &second].iter().for_each(|triangle| {
                        let nearest = closest_point_on_triangle(
                            end,
                            &triangle[0],
                            &triangle[1],
                            &triangle[2],
                        );
                        assert!((&nearest - end).norm() < slack)
                    })
                })
            }
        }
    });
    assert!(crossed > 100 && missed > 100)
}

#[test]
fn overlapping_cubes_cross_along_both_surfaces() {
    let first = solid([0.0; 3], [1.0; 3]);
    let second = solid([0.5; 3], [1.5; 3]);
    let crossings = first
        .tessellation()
        .crossings(second.tessellation())
        .unwrap();
    assert!(!crossings.is_empty());
    crossings.iter().for_each(|crossing| {
        crossing.segment.iter().for_each(|end| {
            assert!(first.signed_distance(end).abs() < Length::meters(1e-12));
            assert!(second.signed_distance(end).abs() < Length::meters(1e-12));
        })
    })
}

#[test]
fn cubes_sharing_a_face_report_the_coplanar_pairs_and_still_cross_at_its_boundary() {
    let first = cuboid([0.0; 3], [1.0; 3]);
    let second = cuboid([1.0, 0.0, 0.0], [2.0, 1.0, 1.0]);
    assert!(!first.coplanar_pairs(&second).unwrap().is_empty());
    assert!(!first.crossings(&second).unwrap().is_empty());
}

#[test]
fn disjoint_cubes_have_no_crossings() {
    let first = cuboid([0.0; 3], [1.0; 3]);
    let second = cuboid([3.0; 3], [4.0; 3]);
    assert!(first.crossings(&second).unwrap().is_empty());
}

#[test]
fn features_and_keys_have_a_fixed_order() {
    assert!(Feature::Vertex(9) < Feature::Edge([0, 1]));
    assert!(Feature::Edge([5, 6]) < Feature::Face(0));
    assert!(Feature::Edge([1, 2]) < Feature::Edge([1, 3]));
    let key = |first, second| Key { first, second };
    assert!(
        key(Feature::Vertex(1), Feature::Face(9)) < key(Feature::Vertex(2), Feature::Vertex(0))
    );
    assert!(
        key(Feature::Vertex(1), Feature::Vertex(3))
            < key(Feature::Vertex(1), Feature::Edge([0, 1]))
    );
}
