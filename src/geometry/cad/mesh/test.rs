use crate::geometry::{
    cad::brep::test::unit_cube,
    mesh::{Class, Connectivity},
    ntree::Balancing,
};

#[test]
fn cube_runs_through_the_octree_dual_pathway() {
    let brep = unit_cube();
    let (mesh, classes) = brep.dual_background(Balancing::Strong(1), 4.0).unwrap();

    let Connectivity::Hexahedral(block) = &mesh.connectivities()[0] else {
        panic!("expected a hexahedral mesh");
    };
    let hexes = block.iter().count();
    assert!(hexes > 0);
    assert_eq!(classes.len(), hexes);
    assert!(classes.contains(&Class::Inside));
    assert!(classes.contains(&Class::Outside));

    let point = |node: usize| {
        let coordinate = &mesh.coordinates()[node];
        [
            coordinate[0].value(),
            coordinate[1].value(),
            coordinate[2].value(),
        ]
    };
    let inside_nodes: Vec<usize> = block
        .iter()
        .zip(&classes)
        .filter(|(_, class)| **class == Class::Inside)
        .flat_map(|(hex, _)| *hex)
        .collect();
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for node in inside_nodes {
        let p = point(node);
        for axis in 0..3 {
            min[axis] = min[axis].min(p[axis]);
            max[axis] = max[axis].max(p[axis]);
        }
    }
    for axis in 0..3 {
        assert!(
            min[axis] > -0.5 && min[axis] < 0.5,
            "min[{axis}] = {}",
            min[axis]
        );
        assert!(
            max[axis] > 0.5 && max[axis] < 1.5,
            "max[{axis}] = {}",
            max[axis]
        );
    }
}

#[test]
fn cut_fits_the_cube_back_to_its_bounds() {
    let brep = unit_cube();
    let tessellation = brep.tessellate().unwrap();
    let (mesh, classes) = tessellation
        .dual_background(Balancing::Strong(1), 4.0)
        .unwrap();
    let fitted = tessellation.cut(mesh, &classes).unwrap();

    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for coordinate in fitted.coordinates() {
        for axis in 0..3 {
            let value = coordinate[axis].value();
            min[axis] = min[axis].min(value);
            max[axis] = max[axis].max(value);
        }
    }
    for axis in 0..3 {
        assert!((min[axis] - 0.0).abs() < 0.2, "min[{axis}] = {}", min[axis]);
        assert!((max[axis] - 1.0).abs() < 0.2, "max[{axis}] = {}", max[axis]);
    }
}
