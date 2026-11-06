use crate::{
    geometry::mesh::{
        Tessellation, Watertight,
        test::{cuboid, solid, sphere},
    },
    math::{
        Quantity,
        assert::{Assert, AssertionError},
    },
    units::Volume,
};
use core::array::from_fn;

fn volume(surface: Result<Tessellation, &'static str>, touching: bool) -> Quantity<Volume> {
    match surface {
        Ok(surface) => {
            let defects = surface.defects();
            if touching {
                assert!(defects.boundary_edges.is_empty(), "{defects:?}");
                assert!(defects.self_intersections.is_empty() && !defects.inward);
            } else {
                assert!(defects.is_empty(), "{defects:?}");
            }
            surface.enclosed_volume()
        }
        Err(message) => {
            assert_eq!(message, "the surface is empty");
            Quantity::default()
        }
    }
}

fn volumes(first: &Watertight, second: &Watertight, touching: bool) -> [Quantity<Volume>; 4] {
    let corefined = first.corefine(second).unwrap();
    [
        volume(corefined.union(), touching),
        volume(corefined.intersection(), touching),
        volume(corefined.first_minus_second(), touching),
        volume(corefined.second_minus_first(), touching),
    ]
}

fn assert_volumes(
    first: &Watertight,
    second: &Watertight,
    expected: [f64; 4],
) -> Result<(), AssertionError> {
    volumes(first, second, false)
        .into_iter()
        .zip(expected)
        .try_for_each(|(found, expected)| {
            Assert::default().eq_within_tols(found, &Quantity::new(expected))
        })
}

#[test]
fn overlapping_cubes() -> Result<(), AssertionError> {
    assert_volumes(
        &solid([0.0; 3], [1.0; 3]),
        &solid([0.5; 3], [1.5; 3]),
        [2.0 - 0.125, 0.125, 1.0 - 0.125, 1.0 - 0.125],
    )
}

#[test]
fn cubes_sharing_a_face_join_and_do_not_overlap() -> Result<(), AssertionError> {
    assert_volumes(
        &solid([0.0; 3], [1.0; 3]),
        &solid([1.0, 0.0, 0.0], [2.0, 1.0, 1.0]),
        [2.0, 0.0, 1.0, 1.0],
    )
}

#[test]
fn a_small_face_on_a_large_one_leaves_a_hole_that_the_small_solid_closes()
-> Result<(), AssertionError> {
    assert_volumes(
        &solid([0.0; 3], [1.0; 3]),
        &solid([1.0, 0.25, 0.25], [2.0, 0.75, 0.75]),
        [1.25, 0.0, 1.0, 0.25],
    )
}

#[test]
fn cubes_with_sides_in_the_same_planes_facing_the_same_way() -> Result<(), AssertionError> {
    assert_volumes(
        &solid([0.0; 3], [1.0; 3]),
        &solid([0.5, 0.0, 0.0], [1.5, 1.0, 1.0]),
        [1.5, 0.5, 0.5, 0.5],
    )
}

#[test]
fn a_cube_inside_another_leaves_a_cavity_when_it_is_taken_away() -> Result<(), AssertionError> {
    assert_volumes(
        &solid([0.0; 3], [3.0; 3]),
        &solid([1.0; 3], [2.0; 3]),
        [27.0, 1.0, 26.0, 0.0],
    )
}

#[test]
fn disjoint_cubes() -> Result<(), AssertionError> {
    assert_volumes(
        &solid([0.0; 3], [1.0; 3]),
        &solid([3.0; 3], [4.0; 3]),
        [2.0, 0.0, 1.0, 1.0],
    )
}

fn assert_inclusion_exclusion(
    first: &Watertight,
    second: &Watertight,
    touching: bool,
) -> Result<(), AssertionError> {
    let [union, intersection, first_minus, second_minus] = volumes(first, second, touching);
    let (a, b) = (
        first.tessellation().enclosed_volume(),
        second.tessellation().enclosed_volume(),
    );
    let tols = Assert {
        abs_tol: 1e-10,
        rel_tol: 1e-10,
        ..Assert::default()
    };
    tols.eq_within_tols(union, &(a + b - intersection))?;
    tols.eq_within_tols(first_minus, &(a - intersection))?;
    tols.eq_within_tols(second_minus, &(b - intersection))
}

#[test]
fn a_sphere_and_a_cube() -> Result<(), AssertionError> {
    let ball = Watertight::new(sphere(12, 16, 0.8)).unwrap();
    let cube = Watertight::new(cuboid([-0.3, -0.3, -0.2], [0.7, 0.6, 0.9])).unwrap();
    assert_inclusion_exclusion(&ball, &cube, false)?;
    let [union, intersection, ..] = volumes(&ball, &cube, false);
    assert!(intersection > Quantity::default());
    assert!(union > ball.tessellation().enclosed_volume());
    Ok(())
}

#[test]
fn random_cuboids_on_a_coarse_grid_obey_inclusion_and_exclusion() -> Result<(), AssertionError> {
    let mut state = 13u64;
    let mut next = |n: u64| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) % n
    };
    let first = solid([0.0; 3], [1.0; 3]);
    for _ in 0..200 {
        let low: [f64; 3] = from_fn(|_| next(5) as f64 * 0.25 - 0.5);
        let high: [f64; 3] = from_fn(|axis| low[axis] + (next(4) + 1) as f64 * 0.25);
        assert_inclusion_exclusion(&first, &solid(low, high), true)?;
    }
    Ok(())
}
