use crate::geometry::{
    Balance, Balancing, Dualization, Pairing, QuadrilateralMesh, Quadtree, WriteExodus,
    ntree::balance::quadtree::test::circle,
};

#[test]
fn from_circle() {
    let coordinates = circle();
    let mut quadtree = Quadtree::<u16, usize>::from((coordinates, 1.0));
    quadtree
        .equilibrate(Balancing::Strong, Pairing::Regular)
        .unwrap();
    let mesh: QuadrilateralMesh<_, 0, usize> = quadtree.dualize();
    mesh.write_exodus("target/dual.exo").unwrap();
}
