use super::super::Class;
use super::super::test::{dual, hexahedron, sphere};
use crate::math::Tensor;
use std::collections::HashMap;

#[test]
fn classify_single_hexahedra() {
    let tessellation = sphere(3);
    assert_eq!(
        tessellation.classify(&hexahedron([-0.1; 3], [0.1; 3])),
        vec![Class::Inside]
    );
    assert_eq!(
        tessellation.classify(&hexahedron([2.0; 3], [3.0; 3])),
        vec![Class::Outside]
    );
    assert_eq!(
        tessellation.classify(&hexahedron([0.9, -0.1, -0.1], [1.1, 0.1, 0.1])),
        vec![Class::Cut]
    );
}

#[test]
fn classify_sphere_dual() {
    let tessellation = sphere(3);
    let mesh = dual(&tessellation, 8.0);
    let classes = tessellation.classify(&mesh);
    [Class::Inside, Class::Cut, Class::Outside]
        .iter()
        .for_each(|class| assert!(classes.contains(class)));
    let centroids = mesh.centroids();
    classes
        .iter()
        .zip(centroids.iter())
        .for_each(|(class, centroid)| match class {
            Class::Inside => assert!(centroid.norm() < 1.0),
            Class::Outside => assert!(centroid.norm() > 1.0),
            Class::Cut => (),
        });
    let mut faces = HashMap::<Vec<usize>, Vec<Class>>::new();
    mesh.iter().for_each(|block| {
        block
            .iter()
            .zip(classes.iter())
            .for_each(|(element, &class)| {
                block.local_faces().iter().for_each(|face| {
                    let mut key: Vec<usize> = face.iter().map(|&local| element[local]).collect();
                    key.sort_unstable();
                    faces.entry(key).or_default().push(class);
                })
            })
    });
    faces.values().for_each(|classes| {
        assert!(!(classes.contains(&Class::Inside) && classes.contains(&Class::Outside)))
    })
}
