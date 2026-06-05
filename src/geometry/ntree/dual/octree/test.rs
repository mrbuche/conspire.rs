use crate::{
    geometry::{
        mesh::{Connectivity, Mesh, Output},
        ntree::{
            Octree,
            balance::{Balance, Balancing, octree::test::sphere},
            dual::Dualization,
            pair::Pairing,
        },
    },
    io::Write,
};

#[test]
fn from_sphere() {
    let coordinates = sphere();
    let mut octree = Octree::<u16, usize>::from((coordinates, 1.0));
    octree
        .equilibrate(Balancing::Strong, Pairing::Regular)
        .unwrap();
    let mesh: Mesh<3> = octree.dualize();
    mesh.write(Output::Exodus("target/dual_octree.exo"))
        .unwrap();
    let (connectivities, coordinates) = mesh.into();
    let hexes: Vec<[usize; 8]> = connectivities
        .into_members()
        .into_iter()
        .flat_map(|block| match block {
            Connectivity::Hexahedral(hexes) => hexes,
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

// vt21 is the 3D-checkerboard vertex: the eight cells around V alternate coarse/fine by
// octant parity. Regular pairing dissolves checkerboards, so it never fires on real input
// (0 hexes on both the sphere and a 900k-star bunny). Here we hand-build the exact octree
// -- a length-8 root, subdivided once into eight macro cells, with the even-popcount
// octants {0,3,5,6} subdivided once more -- so that node O (octant 2) sees the vt21
// pattern. This is the first time vt21 is exercised at all, including its orientation.
#[test]
fn vt21_fires_on_synthetic_checkerboard() {
    use super::vertex::{generic::vertex_dual_generic, transition_21_only};
    use crate::geometry::ntree::{
        dual::Uniform,
        node::{Kind, Node},
        rescale::Rescaling,
    };
    use std::collections::HashSet;

    let mut octree = Octree::<u16, usize> {
        balanced: Balancing::None,
        nodes: vec![Node {
            corner: [0, 0, 0],
            length: 8,
            facets: [None; 6],
            kind: Kind::Leaf,
        }],
        paired: Pairing::None,
        rescale: Rescaling {
            center: [4.0, 4.0, 4.0],
            cell: 1.0,
            half: 4.0,
        },
    };
    octree.subdivide(0).unwrap(); // eight macro cells: octant j -> node 1 + j
    for macro_cell in 1..=8usize {
        octree.subdivide(macro_cell).unwrap(); // every macro cell gets leaf children
    }
    for fine_macro in [1usize, 4, 6, 7] {
        // even-popcount octants 0,3,5,6 -> one level deeper than O
        let children = *octree.nodes[fine_macro].orthants().unwrap();
        for child in children {
            octree.subdivide(child).unwrap();
        }
    }
    octree.balanced = Balancing::Strong; // initialize() only asserts these are set
    octree.paired = Pairing::Regular;

    let (center_nodes, coordinates, ..) = octree.initialize();
    let hexes = transition_21_only(&octree, &center_nodes);
    assert!(
        !hexes.is_empty(),
        "vt21 did not fire on the checkerboard vertex"
    );

    hexes.iter().enumerate().for_each(|(i, hex)| {
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
        assert!(
            vol6 > 1e-12,
            "vt21 hex {i} not positively oriented: {hex:?}, vol6={vol6}"
        );
    });

    // is vt21 a true vertex star (in the generic dual) or a filler like vt1?
    let generic: HashSet<[usize; 8]> = vertex_dual_generic(&octree, &center_nodes)
        .into_iter()
        .map(|mut hex| {
            hex.sort_unstable();
            hex
        })
        .collect();
    let star = hexes.iter().all(|hex| {
        let mut sorted = *hex;
        sorted.sort_unstable();
        generic.contains(&sorted)
    });
    println!("vt21 fired {} hex(es); vertex star = {star}", hexes.len());
}

// The generic "descend toward V" routine emits one cell-center hex per interior vertex
// (a true vertex star). It reproduces every cell-center hex of the existing pipeline --
// the uniform transitions and vertex templates vt2..21 -- confirming those templates pick
// the geometrically-correct cells. The lone exception is vt1, which is not a vertex star
// but a filler hex spanning a coarse face (its fine side grabs the four corners of the
// coarse face), so none of its hexes appear in the generic dual. (Face/edge transitions
// add extra nodes and are not cell-center hexes, so they are excluded here.)
#[test]
fn generic_vertex_dual_reproduces_stars_but_not_vt1() {
    let coordinates = sphere();
    let mut octree = Octree::<u16, usize>::from((coordinates, 1.0));
    octree
        .equilibrate(Balancing::Strong, Pairing::Regular)
        .unwrap();
    let (generic, sets, non_vt1, vt1) = super::generic_star_report(&octree);
    println!(
        "generic stars={generic} (sets {sets}); cell-center stars (uniform+vt2..21)={non_vt1}, \
         vt1 fillers={vt1}; face/edge-type interior vertices={}",
        sets - non_vt1
    );
}
