use crate::geometry::mesh::from::ntree::{
    facets::{Facets, leaves},
    test::octree,
};

fn signed_volume(polygons: &[Vec<Vec<usize>>], facets: &Facets<3>) -> f64 {
    let point = |node: usize| {
        let coordinate = &facets.coordinates()[node];
        [
            coordinate[0].value(),
            coordinate[1].value(),
            coordinate[2].value(),
        ]
    };
    polygons
        .iter()
        .flatten()
        .map(|polygon| {
            let a = point(polygon[0]);
            (1..polygon.len() - 1)
                .map(|k| {
                    let (b, c) = (point(polygon[k]), point(polygon[k + 1]));
                    let (u, v) = (
                        [b[0] - a[0], b[1] - a[1], b[2] - a[2]],
                        [c[0] - a[0], c[1] - a[1], c[2] - a[2]],
                    );
                    (a[0] * (u[1] * v[2] - u[2] * v[1])
                        + a[1] * (u[2] * v[0] - u[0] * v[2])
                        + a[2] * (u[0] * v[1] - u[1] * v[0]))
                        / 6.0
                })
                .sum::<f64>()
        })
        .sum()
}

#[test]
fn leaf_polygons_enclose_each_leaf() {
    let mut tree = octree(8);
    tree.subdivide(0).unwrap();
    tree.subdivide(1).unwrap();
    tree.subdivide(9).unwrap();
    let (leaves, _) = leaves(&tree);
    let facets = Facets::<3>::new(&tree, &leaves);
    let mut hanging = 0;
    leaves.iter().for_each(|&index| {
        let polygons = facets.leaf_polygons(&tree, index);
        assert_eq!(polygons.len(), 6);
        let length = tree.nodes[index].length as f64;
        assert!((signed_volume(&polygons, &facets) - length.powi(3)).abs() < 1e-12);
        hanging += polygons
            .iter()
            .flatten()
            .filter(|polygon| polygon.len() > 4)
            .count()
    });
    assert!(hanging > 0)
}

#[test]
fn corners_are_the_leaf_corners() {
    let mut tree = octree(4);
    tree.subdivide(0).unwrap();
    let (leaves, _) = leaves(&tree);
    let facets = Facets::<3>::new(&tree, &leaves);
    leaves.iter().for_each(|&index| {
        let node = &tree.nodes[index];
        let corner = [
            node.corner[0] as usize,
            node.corner[1] as usize,
            node.corner[2] as usize,
        ];
        let length = node.length as usize;
        facets
            .corners::<8>(corner, length)
            .iter()
            .enumerate()
            .for_each(|(k, &id)| {
                let coordinate = &facets.coordinates()[id];
                (0..3).for_each(|axis| {
                    assert_eq!(
                        coordinate[axis].value(),
                        (corner[axis] + ((k >> axis) & 1) * length) as f64
                    )
                })
            })
    })
}
