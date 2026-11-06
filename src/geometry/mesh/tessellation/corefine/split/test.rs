use crate::{
    geometry::{
        mesh::{
            Tessellation,
            test::{cuboid, solid},
        },
        solid::SolidOracle,
    },
    math::{
        Tensor,
        assert::{Assert, AssertionError},
    },
};

fn assert_same_volume(split: &Tessellation, original: &Tessellation) -> Result<(), AssertionError> {
    Assert::default().eq_within_tols(split.enclosed_volume(), &original.enclosed_volume())
}

fn assert_closed(tessellation: &Tessellation) {
    let defects = tessellation.defects();
    assert!(defects.boundary_edges.is_empty(), "{defects:?}");
    assert!(defects.non_manifold_edges.is_empty());
    assert!(defects.non_manifold_vertices.is_empty());
    assert!(defects.inconsistent_edges.is_empty());
    assert!(!defects.inward);
}

#[test]
fn disjoint_tessellations_are_unchanged() {
    let first = cuboid([0.0; 3], [1.0; 3]);
    let second = cuboid([3.0; 3], [4.0; 3]);
    let [a, b] = first.corefine(&second).unwrap();
    assert_eq!(a.mesh().connectivities().iter().flatten().count(), 12);
    assert_eq!(b.mesh().connectivities().iter().flatten().count(), 12);
}

#[test]
fn overlapping_cubes_are_split_along_their_curve_and_stay_closed() -> Result<(), AssertionError> {
    let first = cuboid([0.0; 3], [1.0; 3]);
    let second = cuboid([0.5; 3], [1.5; 3]);
    let [a, b] = first.corefine(&second).unwrap();
    for (split, original) in [(&a, &first), (&b, &second)] {
        assert_closed(split);
        assert_same_volume(split, original)?;
        let count = |t: &Tessellation| t.mesh().connectivities().iter().flatten().count();
        assert!(count(split) > count(original));
    }
    Ok(())
}

#[test]
fn cubes_sharing_a_face_are_split_where_the_faces_overlap_and_stay_closed()
-> Result<(), AssertionError> {
    let first = cuboid([0.0; 3], [1.0; 3]);
    let second = cuboid([1.0, 0.25, 0.25], [2.0, 0.75, 0.75]);
    let [a, b] = first.corefine(&second).unwrap();
    for (split, original) in [(&a, &first), (&b, &second)] {
        assert_closed(split);
        assert_same_volume(split, original)?;
    }
    Ok(())
}

#[test]
fn cubes_that_cross_through_vertices_and_edges_stay_closed() -> Result<(), AssertionError> {
    let first = cuboid([0.0; 3], [1.0; 3]);
    for (n, second) in [
        cuboid([0.5, 0.0, 0.0], [1.5, 1.0, 1.0]),
        cuboid([1.0, 1.0, 0.0], [2.0, 2.0, 1.0]),
        cuboid([0.5, 0.5, -0.5], [1.5, 1.5, 0.5]),
        cuboid([0.25, 0.25, 0.25], [0.75, 0.75, 1.5]),
    ]
    .into_iter()
    .enumerate()
    {
        println!("case {n}");
        let [a, b] = first.corefine(&second).unwrap();
        for (split, original) in [(&a, &first), (&b, &second)] {
            assert_closed(split);
            assert_same_volume(split, original)?;
        }
    }
    Ok(())
}

#[test]
fn the_new_vertices_lie_on_the_other_surface() -> Result<(), AssertionError> {
    let (first, second) = (solid([0.0; 3], [1.0; 3]), solid([0.5; 3], [1.5; 3]));
    let [a, b] = first
        .tessellation()
        .corefine(second.tessellation())
        .unwrap();
    for (split, original, against) in [(&a, &first, &second), (&b, &second, &first)] {
        let coordinates = split.mesh().coordinates();
        let old = original.tessellation().mesh().coordinates().len();
        assert!(coordinates.len() > old);
        (old..coordinates.len()).try_for_each(|node| {
            Assert::default().zero_within_tols(&against.signed_distance(&coordinates[node]))
        })?;
    }
    Ok(())
}

#[test]
fn random_cuboids_on_a_coarse_grid_split_into_closed_surfaces() -> Result<(), AssertionError> {
    let mut state = 7u64;
    let mut next = |n: u64| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) % n
    };
    let (mut done, mut failed) = (0, 0);
    for _ in 0..300 {
        let low: [f64; 3] = core::array::from_fn(|_| next(5) as f64 * 0.25 - 0.5);
        let high: [f64; 3] = core::array::from_fn(|axis| low[axis] + (next(4) + 1) as f64 * 0.25);
        let (first, second) = (cuboid([0.0; 3], [1.0; 3]), cuboid(low, high));
        match first.corefine(&second) {
            Ok([a, b]) => {
                done += 1;
                for (split, original) in [(&a, &first), (&b, &second)] {
                    assert_closed(split);
                    assert_same_volume(split, original)?;
                }
            }
            Err(_) => failed += 1,
        }
    }
    println!("{done} done, {failed} failed");
    assert!(done > 250);
    Ok(())
}

#[test]
fn a_sphere_and_a_cube_split_into_closed_surfaces() -> Result<(), AssertionError> {
    use crate::geometry::mesh::test::sphere;
    let cube = cuboid([-0.3, -0.3, -0.2], [0.7, 0.6, 0.9]);
    let ball = sphere(12, 16, 0.8);
    let [a, b] = ball.corefine(&cube).unwrap();
    assert_closed(&a);
    assert_closed(&b);
    assert_same_volume(&a, &ball)?;
    assert_same_volume(&b, &cube)
}
