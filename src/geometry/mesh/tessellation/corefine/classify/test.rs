use super::{Corefined, Region};
use crate::{
    geometry::{
        Coordinate,
        mesh::{Tessellation, Watertight, test::solid},
        solid::SolidOracle,
    },
    math::{CrossProduct, Tensor},
    units::Length,
};
use std::array::from_fn;

fn area(tessellation: &Tessellation, regions: &[Region], region: Region) -> f64 {
    let coordinates = tessellation.mesh().coordinates();
    tessellation
        .mesh()
        .connectivities()
        .iter()
        .flatten()
        .zip(regions)
        .filter(|(_, found)| **found == region)
        .map(|(nodes, _)| {
            let [a, b, c] = [0, 1, 2].map(|i| &coordinates[nodes[i]]);
            0.5 * (&(b - a)).cross(&(c - a)).norm().value()
        })
        .sum()
}

fn count(tessellation: &Tessellation) -> usize {
    tessellation
        .mesh()
        .connectivities()
        .iter()
        .flatten()
        .count()
}

fn areas(corefined: &Corefined) -> [[f64; 4]; 2] {
    let four = |tessellation: &Tessellation, regions: &[Region]| {
        [
            Region::Outside,
            Region::Inside,
            Region::Coincident { aligned: true },
            Region::Coincident { aligned: false },
        ]
        .map(|region| area(tessellation, regions, region))
    };
    [
        four(&corefined.first, &corefined.first_regions),
        four(&corefined.second, &corefined.second_regions),
    ]
}

fn assert_areas(found: [f64; 4], expected: [f64; 4]) {
    found
        .iter()
        .zip(&expected)
        .for_each(|(f, e)| assert!((f - e).abs() < 1e-12, "{found:?} != {expected:?}"))
}

fn corefine(a: &Watertight, b: &Watertight) -> Corefined {
    let corefined = a.corefine(b).unwrap();
    assert_eq!(corefined.first_regions.len(), count(&corefined.first));
    assert_eq!(corefined.second_regions.len(), count(&corefined.second));
    corefined
}

#[test]
fn disjoint_cubes_are_outside_each_other() {
    let corefined = corefine(&solid([0.0; 3], [1.0; 3]), &solid([3.0; 3], [4.0; 3]));
    let [first, second] = areas(&corefined);
    assert_areas(first, [6.0, 0.0, 0.0, 0.0]);
    assert_areas(second, [6.0, 0.0, 0.0, 0.0]);
}

#[test]
fn a_cube_inside_another_is_inside_and_the_other_outside() {
    let corefined = corefine(&solid([0.0; 3], [3.0; 3]), &solid([1.0; 3], [2.0; 3]));
    let [outer, inner] = areas(&corefined);
    assert_areas(outer, [54.0, 0.0, 0.0, 0.0]);
    assert_areas(inner, [0.0, 6.0, 0.0, 0.0]);
}

#[test]
fn overlapping_cubes_have_a_quarter_of_three_faces_inside_each_other() {
    let corefined = corefine(&solid([0.0; 3], [1.0; 3]), &solid([0.5; 3], [1.5; 3]));
    let [first, second] = areas(&corefined);
    assert_areas(first, [6.0 - 0.75, 0.75, 0.0, 0.0]);
    assert_areas(second, [6.0 - 0.75, 0.75, 0.0, 0.0]);
}

#[test]
fn cubes_sharing_a_face_are_coincident_and_opposed_there() {
    let corefined = corefine(
        &solid([0.0; 3], [1.0; 3]),
        &solid([1.0, 0.0, 0.0], [2.0, 1.0, 1.0]),
    );
    let [first, second] = areas(&corefined);
    assert_areas(first, [5.0, 0.0, 0.0, 1.0]);
    assert_areas(second, [5.0, 0.0, 0.0, 1.0]);
}

#[test]
fn a_small_face_on_a_large_one_is_coincident_on_both_and_the_rest_is_outside() {
    let corefined = corefine(
        &solid([0.0; 3], [1.0; 3]),
        &solid([1.0, 0.25, 0.25], [2.0, 0.75, 0.75]),
    );
    let [first, second] = areas(&corefined);
    assert_areas(first, [6.0 - 0.25, 0.0, 0.0, 0.25]);
    assert_areas(second, [2.25, 0.0, 0.0, 0.25]);
}

#[test]
fn cubes_overlapping_with_faces_in_the_same_planes_are_coincident_and_aligned() {
    let corefined = corefine(
        &solid([0.0; 3], [1.0; 3]),
        &solid([0.5, 0.0, 0.0], [1.5, 1.0, 1.0]),
    );
    let [first, second] = areas(&corefined);
    assert_areas(first, [5.0 - 2.0, 1.0, 2.0, 0.0]);
    assert_areas(second, [5.0 - 2.0, 1.0, 2.0, 0.0]);
}

fn assert_sides(corefined: &Corefined, first: &Watertight, second: &Watertight) {
    for (tessellation, regions, other) in [
        (&corefined.first, &corefined.first_regions, second),
        (&corefined.second, &corefined.second_regions, first),
    ] {
        let coordinates = tessellation.mesh().coordinates();
        for (nodes, region) in tessellation
            .mesh()
            .connectivities()
            .iter()
            .flatten()
            .zip(regions)
        {
            let centre: Coordinate<3> = from_fn(|k| {
                (0..3)
                    .map(|i| coordinates[nodes[i]][k].value())
                    .sum::<f64>()
                    / 3.0
            })
            .into();
            let distance = other.signed_distance(&centre);
            match region {
                Region::Coincident { .. } => {
                    assert!(distance.abs() < Length::meters(1e-9));
                }
                Region::Inside => assert!(distance < Length::meters(1e-9)),
                Region::Outside => assert!(distance > Length::meters(-1e-9)),
            }
        }
    }
}

#[test]
fn a_sphere_and_a_cube_are_classified_by_the_side_their_triangles_are_on() {
    use crate::geometry::mesh::test::{cuboid, sphere};
    let ball = Watertight::new(sphere(12, 16, 0.8)).unwrap();
    let cube = Watertight::new(cuboid([-0.3, -0.3, -0.2], [0.7, 0.6, 0.9])).unwrap();
    let corefined = corefine(&ball, &cube);
    assert_sides(&corefined, &ball, &cube);
    let [first, second] = areas(&corefined);
    assert!(first[0] > 0.0 && first[1] > 0.0 && second[0] > 0.0 && second[1] > 0.0);
}

#[test]
fn random_cuboids_on_a_coarse_grid_are_classified_by_the_side_they_are_on() {
    let mut state = 11u64;
    let mut next = |n: u64| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) % n
    };
    let first = solid([0.0; 3], [1.0; 3]);
    for _ in 0..200 {
        let low: [f64; 3] = std::array::from_fn(|_| next(5) as f64 * 0.25 - 0.5);
        let high: [f64; 3] = std::array::from_fn(|axis| low[axis] + (next(4) + 1) as f64 * 0.25);
        let second = solid(low, high);
        let corefined = corefine(&first, &second);
        assert_sides(&corefined, &first, &second);
    }
}

#[test]
fn the_classification_does_not_depend_on_the_size_of_the_solids() {
    for scale in [1e-14, 1e-6, 1.0, 1e6] {
        let at = |x: f64| [x * scale; 3];
        let corefined = corefine(&solid(at(0.0), at(1.0)), &solid(at(0.5), at(1.5)));
        let [first, second] = areas(&corefined);
        let squared = scale * scale;
        let expected = [(6.0 - 0.75) * squared, 0.75 * squared, 0.0, 0.0];
        for found in [first, second] {
            found
                .iter()
                .zip(&expected)
                .for_each(|(f, e)| assert!((f - e).abs() < 1e-12 * squared, "{scale}"));
        }
    }
}

#[test]
fn the_classification_does_not_depend_on_how_far_the_solids_are_from_the_origin() {
    for shift in [0.0, 1e3, 1e6] {
        let at = |x: f64| [x + shift; 3];
        let corefined = corefine(&solid(at(0.0), at(1.0)), &solid(at(0.5), at(1.5)));
        let [first, second] = areas(&corefined);
        for found in [first, second] {
            assert!((found[0] - 5.25).abs() < 1e-6 && (found[1] - 0.75).abs() < 1e-6);
        }
    }
}
