use crate::geometry::{
    Coordinates, Write,
    mesh::{Connectivity, Mesh, Output},
    ntree::{
        Octree,
        balance::{Balance, Balancing},
        pair::Pairing,
    },
};
use std::f64::consts::PI;

pub fn sphere() -> Coordinates<3> {
    let num_points = 256;
    let radius = 100.0;
    let center = [128.0, 128.0, 128.0];
    let golden_angle = PI * (3.0 - 5.0_f64.sqrt());
    (0..num_points)
        .map(|i| {
            let y = 1.0 - (i as f64 / (num_points as f64 - 1.0)) * 2.0;
            let r = (1.0 - y * y).sqrt();
            let theta = golden_angle * i as f64;
            [
                center[0] + radius * theta.cos() * r,
                center[1] + radius * y,
                center[2] + radius * theta.sin() * r,
            ]
            .into()
        })
        .collect()
}

#[test]
fn from_sphere() {
    let coordinates = sphere();
    let mut octree = Octree::<u16, usize>::from((coordinates, 1.0));
    octree
        .equilibrate(Balancing::Strong, Pairing::Regular)
        .unwrap();
    octree.prune();
    let (connectivity, coordinates): (Connectivity, Coordinates<3>) = octree.into();
    let mesh: Mesh<3> = (vec![connectivity], coordinates).into();
    mesh.write(Output::Exodus("target/octree.exo")).unwrap();
}
