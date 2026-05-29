use crate::geometry::{
    Balance, Balancing, Coordinates, Pairing, Quadtree, WriteExodus,
    mesh::{Connectivity, Mesh},
};
use std::f64::consts::TAU;

pub fn circle() -> Coordinates<2> {
    let num_points = 256;
    let radius = 100.0;
    let center = [128.0, 128.0];
    (0..num_points)
        .map(|i| {
            let theta = TAU * (i as f64) / (num_points as f64);
            [
                center[0] + radius * theta.cos(),
                center[1] + radius * theta.sin(),
            ]
            .into()
        })
        .collect()
}

#[test]
fn from_circle() {
    let coordinates = circle();
    let mut quadtree = Quadtree::<u16, usize>::from((coordinates, 1.0));
    quadtree
        .equilibrate(Balancing::Strong, Pairing::Regular)
        .unwrap();
    quadtree.prune();
    let (connectivity, coordinates): (Connectivity<usize>, Coordinates<2>) = quadtree.into();
    let mesh: Mesh<2, usize> = (vec![connectivity], coordinates).into();
    mesh.write_exodus("target/quadtree.exo").unwrap();
}
