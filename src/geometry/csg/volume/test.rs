use super::Volume;
use crate::{
    geometry::{
        Coordinate, Coordinates,
        csg::{Cuboid, Cylinder, Sphere},
        mesh::test::solid,
        solid::{Partition, SolidOracle},
    },
    math::{
        Quantity, Tensor,
        assert::{Assert, AssertionError},
    },
    units::Length,
};

fn length(value: f64) -> Quantity<Length> {
    Quantity::new(value)
}

fn point(entries: [f64; 3]) -> Coordinate<3> {
    entries.into()
}

fn grid() -> Coordinates<3> {
    let axis: Vec<f64> = (0..11).map(|i| -0.5 + 0.37 * i as f64).collect();
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

fn ball() -> Sphere {
    Sphere::new(point([2.0, 1.0, 1.0]), length(0.8)).unwrap()
}

fn pin() -> Cylinder {
    Cylinder::new(
        point([-0.5, 1.0, 1.0]),
        [1.0, 0.0, 0.0].into(),
        length(0.3),
        length(1.5),
    )
    .unwrap()
}

fn shapes() -> Vec<Volume> {
    vec![
        solid([0.0; 3], [2.0; 3]).into(),
        ball().into(),
        pin().into(),
    ]
}

#[test]
fn a_volume_answers_as_the_solid_it_holds() -> Result<(), AssertionError> {
    let volumes = shapes();
    let surface = solid([0.0; 3], [2.0; 3]);
    let assert = Assert::default();
    for query in grid().iter() {
        assert.eq_within_tols(
            volumes[0].signed_distance(query),
            &surface.signed_distance(query),
        )?;
        assert.eq_within_tols(
            volumes[1].signed_distance(query),
            &ball().signed_distance(query),
        )?;
        assert.eq_within_tols(
            volumes[2].signed_distance(query),
            &pin().signed_distance(query),
        )?;
    }
    Ok(())
}

#[test]
fn the_later_shape_owns_the_space_where_a_surface_and_primitives_overlap() {
    let partition = Partition::new(shapes());
    let owner = |entries| partition.owner(&point(entries));
    assert_eq!(owner([1.1, 1.0, 1.0]), Some(0));
    assert_eq!(owner([1.8, 1.0, 1.0]), Some(1));
    assert_eq!(owner([2.6, 1.0, 1.0]), Some(1));
    assert_eq!(owner([0.2, 1.0, 1.0]), Some(2));
    assert_eq!(owner([-0.3, 1.0, 1.0]), Some(2));
    assert_eq!(owner([0.2, 0.2, 0.2]), Some(0));
    assert_eq!(owner([4.0, 4.0, 4.0]), None);
}

#[test]
fn the_region_of_a_volume_is_what_the_later_ones_leave() -> Result<(), AssertionError> {
    let partition = Partition::new(shapes());
    let volumes = shapes();
    for query in grid().iter() {
        let expected = volumes[0]
            .signed_distance(query)
            .max(-volumes[1].signed_distance(query))
            .max(-volumes[2].signed_distance(query));
        Assert::default().eq_within_tols(partition.signed_distance(query, 0), &expected)?;
    }
    Ok(())
}

#[test]
fn projections_land_on_the_boundary_of_each_region() -> Result<(), AssertionError> {
    let partition = Partition::new(shapes());
    let tols = Assert {
        abs_tol: 1e-9,
        ..Assert::default()
    };
    for index in 0..3 {
        for query in grid().iter() {
            if let Some(found) = partition.project(query, index) {
                tols.zero_within_tols(&partition.signed_distance(&found.point, index))?;
            }
        }
    }
    Ok(())
}

fn host_and_package() -> Vec<Volume> {
    vec![
        solid([2.0, -1.0, -1.0], [7.0, 2.0, 2.0]).into(),
        Volume::union(vec![
            Cuboid::new(point([3.0, 0.0, 0.0]), point([4.0, 1.0, 1.0]))
                .unwrap()
                .into(),
            solid([4.0, 0.0, 0.0], [5.0, 1.0, 1.0]).into(),
        ]),
    ]
}

#[test]
fn a_package_of_a_surface_and_a_primitive_is_one_material() {
    let partition = Partition::new(host_and_package());
    let owner = |entries| partition.owner(&point(entries));
    assert_eq!(owner([3.5, 0.5, 0.5]), Some(1));
    assert_eq!(owner([4.5, 0.5, 0.5]), Some(1));
    assert_eq!(owner([6.0, 0.5, 0.5]), Some(0));
    assert_eq!(owner([8.0, 0.5, 0.5]), None);
}

#[test]
fn a_union_answers_as_the_nearer_of_its_parts() -> Result<(), AssertionError> {
    let package = host_and_package().pop().unwrap();
    let primitive = Cuboid::new(point([3.0, 0.0, 0.0]), point([4.0, 1.0, 1.0])).unwrap();
    let surface = solid([4.0, 0.0, 0.0], [5.0, 1.0, 1.0]);
    for query in grid().iter() {
        let expected = primitive
            .signed_distance(query)
            .min(surface.signed_distance(query));
        Assert::default().eq_within_tols(package.signed_distance(query), &expected)?;
    }
    Ok(())
}

#[test]
fn a_union_of_unions_is_a_union() -> Result<(), AssertionError> {
    let nested = Volume::union(vec![
        Volume::union(vec![ball().into(), pin().into()]),
        solid([0.0; 3], [2.0; 3]).into(),
    ]);
    let flat = Volume::union(shapes());
    for query in grid().iter() {
        Assert::default()
            .eq_within_tols(nested.signed_distance(query), &flat.signed_distance(query))?;
    }
    Ok(())
}
