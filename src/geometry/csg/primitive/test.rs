use super::Primitive;
use crate::{
    geometry::{
        Coordinate,
        csg::{Cuboid, Sphere, ops::Union},
        solid::{Partition, SolidOracle},
    },
    math::Quantity,
    units::Length,
};

fn length(value: f64) -> Quantity<Length> {
    Quantity::new(value)
}

fn point(entries: [f64; 3]) -> Coordinate<3> {
    entries.into()
}

fn shapes() -> Vec<Primitive> {
    vec![
        Cuboid::new(point([0.0; 3]), point([2.0, 1.0, 1.0]))
            .unwrap()
            .into(),
        Sphere::new(point([2.0, 0.5, 0.5]), length(0.8))
            .unwrap()
            .into(),
    ]
}

#[test]
fn a_primitive_answers_as_the_shape_it_holds() {
    let primitives = shapes();
    let cuboid = Cuboid::new(point([0.0; 3]), point([2.0, 1.0, 1.0])).unwrap();
    let sphere = Sphere::new(point([2.0, 0.5, 0.5]), length(0.8)).unwrap();
    for query in [[1.0, 0.5, 0.5], [3.0, 0.5, 0.5], [2.5, 2.0, -1.0]] {
        let query = point(query);
        assert_eq!(
            primitives[0].signed_distance(&query),
            cuboid.signed_distance(&query)
        );
        assert_eq!(
            primitives[1].signed_distance(&query),
            sphere.signed_distance(&query)
        );
    }
}

#[test]
fn different_shapes_make_a_union_and_a_partition() {
    let [cuboid, sphere]: [Primitive; 2] = shapes().try_into().ok().unwrap();
    let union = Union::new(cuboid, sphere);
    assert!(union.signed_distance(&point([1.0, 0.5, 0.5])) < Quantity::default());
    assert!(union.signed_distance(&point([2.6, 0.5, 0.5])) < Quantity::default());
    assert!(union.signed_distance(&point([4.0, 0.5, 0.5])) > Quantity::default());
    let partition = Partition::new(shapes());
    assert_eq!(partition.owner(&point([0.5, 0.5, 0.5])), Some(0));
    assert_eq!(partition.owner(&point([1.9, 0.5, 0.5])), Some(1));
    assert_eq!(partition.owner(&point([5.0, 0.5, 0.5])), None);
}
