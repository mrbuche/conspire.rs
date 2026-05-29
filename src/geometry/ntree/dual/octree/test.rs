use crate::geometry::{
    Balance, Balancing, Dualization, Octree, Pairing, WriteExodus,
    mesh::{Connectivity, MeshNew, PrimitiveConnectivity},
    ntree::balance::octree::test::sphere,
};

#[test]
fn from_sphere() {
    let coordinates = sphere();
    let mut octree = Octree::<u16, usize>::from((coordinates, 1.0));
    octree
        .equilibrate(Balancing::Strong, Pairing::Regular)
        .unwrap();
    let mesh: MeshNew<3, usize> = octree.dualize();
    (&mesh).write_exodus("target/dual_octree.exo").unwrap();
    let (connectivities, coordinates) = mesh.into();
    let hexes: Vec<[usize; 8]> = connectivities
        .into_iter()
        .flat_map(|block| match block {
            Connectivity::Hexahedral(PrimitiveConnectivity(hexes)) => hexes,
            _ => panic!("expected only hexahedral blocks"),
        })
        .collect();
    assert!(!hexes.is_empty(), "no hexes produced");
    hexes.into_iter().enumerate().for_each(|(i, hex)| {
        let p: [[f64; 3]; 8] = std::array::from_fn(|k| {
            let v = &coordinates[hex[k]];
            [v[0], v[1], v[2]]
        });
        let tet = |a: usize, b: usize, c: usize, d: usize| -> f64 {
            let ab = [p[b][0] - p[a][0], p[b][1] - p[a][1], p[b][2] - p[a][2]];
            let ac = [p[c][0] - p[a][0], p[c][1] - p[a][1], p[c][2] - p[a][2]];
            let ad = [p[d][0] - p[a][0], p[d][1] - p[a][1], p[d][2] - p[a][2]];
            ab[0] * (ac[1] * ad[2] - ac[2] * ad[1]) - ab[1] * (ac[0] * ad[2] - ac[2] * ad[0])
                + ab[2] * (ac[0] * ad[1] - ac[1] * ad[0])
        };
        let vol6 = tet(0, 1, 2, 6)
            + tet(0, 2, 3, 6)
            + tet(0, 3, 7, 6)
            + tet(0, 7, 4, 6)
            + tet(0, 4, 5, 6)
            + tet(0, 5, 1, 6);
        let sign = if vol6 > 1e-12 {
            1
        } else if vol6 < -1e-12 {
            -1
        } else {
            panic!("degenerate hex {i}: {hex:?} vol6={vol6}");
        };
        assert_eq!(sign, 1, "flipped hex at element {i}: {hex:?}, vol6={vol6}")
    })
}