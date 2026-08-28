use crate::{
    geometry::cad::{brep::test::axis_aligned_box, sizing::FeatureSizing},
    geometry::mesh::Class,
    math::Quantity,
    units::Length,
};

fn length(value: f64) -> Quantity<Length> {
    Quantity::new(value)
}

#[test]
fn box_octree_partitions_into_all_three_classes() {
    let extents = [2.0, 4.0, 8.0];
    let brep = axis_aligned_box(extents);
    let sizing = FeatureSizing::of(&brep, 32, length(0.05), length(1.0), 0.25);
    let mesh = brep.sizing_octree(&sizing, 6, 0.1).unwrap();
    let classes = brep.classify(&mesh).unwrap();
    assert_eq!(classes.len(), mesh.number_of_elements());

    let count = |wanted: Class| classes.iter().filter(|&&class| class == wanted).count();
    let (inside, cut, outside) = (
        count(Class::Inside),
        count(Class::Cut),
        count(Class::Outside),
    );
    assert!(
        inside > 0 && cut > 0 && outside > 0,
        "inside {inside}, cut {cut}, outside {outside}"
    );

    // Away from the boundary, the flood fill must agree with a direct test.
    let centroids = mesh.centroids();
    for (index, &class) in classes.iter().enumerate() {
        if class == Class::Cut {
            continue;
        }
        assert_eq!(
            class == Class::Inside,
            brep.encloses(&centroids[index]).unwrap(),
            "cell {index}"
        );
    }
}
