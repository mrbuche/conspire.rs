use crate::geometry::{
    Balance, Balancing, Dualization, Pairing, QuadrilateralMesh, Quadtree, WriteExodus,
    ntree::balance::quadtree::test::circle,
};

#[test]
fn from_circle() {
    let coordinates = circle();
    let mut quadtree = Quadtree::<u16, usize>::from((coordinates, 1.0));
    quadtree
        .equilibrate(Balancing::Weak, Pairing::Regular)
        .unwrap();
    let mesh: QuadrilateralMesh<_, 0, usize> = quadtree.dualize();
    (&mesh).write_exodus("target/dual.exo").unwrap();
    let (connectivity, coordinates) = mesh.into();
    connectivity
        .into_iter()
        .enumerate()
        .for_each(|(i, [n0, n1, n2, n3])| {
            let p0 = &coordinates[n0];
            let p1 = &coordinates[n1];
            let p2 = &coordinates[n2];
            let p3 = &coordinates[n3];
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
