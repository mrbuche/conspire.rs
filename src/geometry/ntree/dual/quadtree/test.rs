use crate::geometry::{
    Balance, Balancing, Dualization, Pairing, Quadtree, WriteExodus,
    mesh::{Connectivity, MeshNew, PrimitiveConnectivity},
    ntree::balance::quadtree::test::circle,
};

#[test]
fn from_circle() {
    let coordinates = circle();
    let mut quadtree = Quadtree::<u16, usize>::from((coordinates, 1.0));
    quadtree
        .equilibrate(Balancing::Weak, Pairing::Regular)
        .unwrap();
    let mesh: MeshNew<2, i32> = quadtree.dualize();
    (&mesh).write_exodus("target/dual_quadtree.exo").unwrap();
    let (connectivities, coordinates) = mesh.into();
    let quads: Vec<[i32; 4]> = connectivities
        .into_iter()
        .flat_map(|block| match block {
            Connectivity::Quadrilateral(PrimitiveConnectivity(quads)) => quads,
            _ => panic!("expected only quadrilateral blocks"),
        })
        .collect();
    quads
        .into_iter()
        .enumerate()
        .for_each(|(i, [n0, n1, n2, n3])| {
            let p0 = &coordinates[n0 as usize];
            let p1 = &coordinates[n1 as usize];
            let p2 = &coordinates[n2 as usize];
            let p3 = &coordinates[n3 as usize];
            let area2 = p0[0] * p1[1] - p0[1] * p1[0] + p1[0] * p2[1] - p1[1] * p2[0]
                + p2[0] * p3[1]
                - p2[1] * p3[0]
                + p3[0] * p0[1]
                - p3[1] * p0[0];

            let sign = if area2 > 1e-12 {
                1
            } else if area2 < -1e-12 {
                -1
            } else {
                panic!("degenerate quad {i}: [{n0}, {n1}, {n2}, {n3}] area2={area2}");
            };
            assert_eq!(
                sign, 1,
                "flipped quad at element {i}: [{n0}, {n1}, {n2}, {n3}], area2={area2}"
            )
        })
}