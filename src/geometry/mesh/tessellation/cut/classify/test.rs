use super::super::test::{box_surface, dual, hexahedron, sphere};
use super::super::{Class, RegionClass};
use super::classify_regions;
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
            Class::Inside => assert!(centroid.norm().value() < 1.0),
            Class::Outside => assert!(centroid.norm().value() > 1.0),
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

#[test]
fn classify_regions_disjoint_boxes() {
    let surfaces = [
        box_surface([-1.0; 3], [-0.5; 3]),
        box_surface([0.5; 3], [1.0; 3]),
    ];
    assert_eq!(
        classify_regions(&surfaces, &hexahedron([-0.9; 3], [-0.6; 3])),
        vec![RegionClass::Inside(0)]
    );
    assert_eq!(
        classify_regions(&surfaces, &hexahedron([0.6; 3], [0.9; 3])),
        vec![RegionClass::Inside(1)]
    );
    assert_eq!(
        classify_regions(&surfaces, &hexahedron([-0.1; 3], [0.1; 3])),
        vec![RegionClass::Outside]
    );
    assert_eq!(
        classify_regions(
            &surfaces,
            &hexahedron([-0.6, -0.6, -0.6], [-0.4, -0.4, -0.4])
        ),
        vec![RegionClass::Cut(vec![0])]
    );
}

#[test]
fn classify_regions_nested_priority() {
    let surfaces = [
        box_surface([-1.0; 3], [1.0; 3]),
        box_surface([-2.0; 3], [2.0; 3]),
    ];
    assert_eq!(
        classify_regions(&surfaces, &hexahedron([-0.1; 3], [0.1; 3])),
        vec![RegionClass::Inside(0)]
    );
    assert_eq!(
        classify_regions(&surfaces, &hexahedron([1.2, 1.2, 1.2], [1.4, 1.4, 1.4])),
        vec![RegionClass::Inside(1)]
    );
    assert_eq!(
        classify_regions(&surfaces, &hexahedron([2.5; 3], [2.8; 3])),
        vec![RegionClass::Outside]
    );
}

#[test]
fn classify_regions_shared_boundary_cut() {
    let surfaces = [
        box_surface([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        box_surface([1.0, 0.0, 0.0], [2.0, 1.0, 1.0]),
    ];
    assert_eq!(
        classify_regions(&surfaces, &hexahedron([0.9, 0.4, 0.4], [1.1, 0.6, 0.6])),
        vec![RegionClass::Cut(vec![0, 1])]
    );
}
