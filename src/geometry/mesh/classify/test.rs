use super::NodeClass::{self, Curve, Interior, Surface, Vertex};
use crate::geometry::{Coordinate, csg::Cuboid, mesh::Mesh, solid::Partition};

fn point(entries: [f64; 3]) -> Coordinate<3> {
    entries.into()
}

fn cuboid(low: [f64; 3], high: [f64; 3]) -> Cuboid {
    Cuboid::new(point(low), point(high)).unwrap()
}

fn classes(volumes: Vec<Cuboid>, upper: [f64; 3], nel: [usize; 3]) -> (Mesh<3>, Vec<NodeClass>) {
    let mesh = Mesh::from_partition(
        &Partition::new(volumes),
        &point([0.0; 3]),
        &point(upper),
        nel,
    )
    .unwrap();
    let classes = mesh.classify_nodes().unwrap();
    (mesh, classes)
}

fn class_at(mesh: &Mesh<3>, classes: &[NodeClass], position: [f64; 3]) -> NodeClass {
    let node = (0..mesh.number_of_nodes())
        .find(|&node| {
            (0..3).all(|a| (mesh.coordinates()[node][a].value() - position[a]).abs() < 1e-9)
        })
        .unwrap();
    classes[node]
}

#[test]
fn a_single_volume_is_interior_inside_and_surface_on_its_boundary() {
    let (mesh, classes) = classes(vec![cuboid([0.0; 3], [4.0; 3])], [4.0; 3], [4; 3]);
    assert_eq!(class_at(&mesh, &classes, [2.0; 3]), Interior);
    assert_eq!(class_at(&mesh, &classes, [0.0, 2.0, 2.0]), Surface);
    assert_eq!(class_at(&mesh, &classes, [0.0, 0.0, 2.0]), Surface);
    assert_eq!(class_at(&mesh, &classes, [0.0; 3]), Surface);
    assert_eq!(classes.iter().filter(|&&c| c == Interior).count(), 27);
}

#[test]
fn nodes_between_two_volumes_are_on_the_surface_between_them() {
    let (mesh, classes) = classes(
        vec![
            cuboid([0.0; 3], [2.0, 4.0, 4.0]),
            cuboid([2.0, 0.0, 0.0], [4.0; 3]),
        ],
        [4.0; 3],
        [4; 3],
    );
    assert_eq!(class_at(&mesh, &classes, [1.0, 2.0, 2.0]), Interior);
    assert_eq!(class_at(&mesh, &classes, [2.0; 3]), Surface);
    assert_eq!(class_at(&mesh, &classes, [2.0, 0.0, 2.0]), Curve);
    assert_eq!(class_at(&mesh, &classes, [2.0, 2.0, 0.0]), Curve);
}

#[test]
fn nodes_where_the_surface_between_two_volumes_turns_stay_on_it() {
    let (mesh, classes) = classes(
        vec![cuboid([0.0; 3], [4.0; 3]), cuboid([1.0; 3], [3.0; 3])],
        [4.0; 3],
        [4; 3],
    );
    assert_eq!(class_at(&mesh, &classes, [1.0; 3]), Surface);
    assert_eq!(class_at(&mesh, &classes, [1.0, 1.0, 2.0]), Surface);
    assert_eq!(class_at(&mesh, &classes, [2.0; 3]), Interior);
}

#[test]
fn four_volumes_around_a_line_meet_on_a_curve_and_end_at_vertices() {
    let (mesh, classes) = classes(
        vec![
            cuboid([0.0; 3], [2.0, 2.0, 2.0]),
            cuboid([2.0, 0.0, 0.0], [4.0, 2.0, 2.0]),
            cuboid([0.0, 2.0, 0.0], [2.0, 4.0, 2.0]),
            cuboid([2.0, 2.0, 0.0], [4.0, 4.0, 2.0]),
        ],
        [4.0, 4.0, 2.0],
        [4, 4, 2],
    );
    assert_eq!(class_at(&mesh, &classes, [2.0, 2.0, 1.0]), Curve);
    assert_eq!(class_at(&mesh, &classes, [2.0, 2.0, 0.0]), Vertex);
    assert_eq!(class_at(&mesh, &classes, [2.0, 2.0, 2.0]), Vertex);
}

#[test]
fn volumes_with_a_gap_between_them_each_have_their_own_outer_surface() {
    let (mesh, classes) = classes(
        vec![
            cuboid([0.0; 3], [4.0, 6.0, 6.0]),
            cuboid([6.0, 0.0, 0.0], [10.0, 6.0, 6.0]),
        ],
        [10.0, 6.0, 6.0],
        [10, 6, 6],
    );
    assert_eq!(class_at(&mesh, &classes, [4.0, 3.0, 3.0]), Surface);
    assert_eq!(class_at(&mesh, &classes, [6.0, 3.0, 3.0]), Surface);
    assert_eq!(class_at(&mesh, &classes, [2.0, 3.0, 3.0]), Interior);
}
