use crate::{
    geometry::{
        Coordinates,
        mesh::{
            Connectivity, Mesh, Tessellation, Verdict, from::ntree::facets::leaves,
            test::sphere as ball,
        },
        ntree::{Balance, Balancing, CurvatureSizing, Octree, Pairing, node::slot::Slot},
    },
    math::{FxHashMap, Scalar},
};
use std::num::NonZeroU32;

type Tree = Octree<u16, NonZeroU32>;

fn box_surface(minimum: [f64; 3], maximum: [f64; 3]) -> Tessellation {
    let [x0, y0, z0] = minimum;
    let [x1, y1, z1] = maximum;
    let coordinates = vec![
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ];
    let faces: Vec<[usize; 3]> = [
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
        [0, 3, 2, 1],
        [4, 5, 6, 7],
    ]
    .iter()
    .flat_map(|&[a, b, c, d]| [[a, b, c], [a, c, d]])
    .collect();
    Tessellation::from(Mesh::from((
        vec![Connectivity::Triangular(faces.into())],
        Coordinates::from(coordinates),
    )))
}

fn tree(tessellation: &Tessellation, scale: Scalar) -> Tree {
    let mut octree =
        Tree::from_features(tessellation, scale, CurvatureSizing::default(), 2).unwrap();
    octree
        .equilibrate(Balancing::Strong(1), Pairing::None)
        .unwrap();
    octree
}

/// Leaf count and occupied volume, before the tree is consumed.
fn occupancy<U: Slot>(octree: &Octree<u16, U>) -> (usize, Scalar) {
    let (leaves, _) = leaves(octree);
    let cell = octree.rescale().cell.value();
    let volume = leaves
        .iter()
        .map(|&index| (cell * octree.nodes[index].length as Scalar).powi(3))
        .sum();
    (leaves.len(), volume)
}

fn tets(mesh: &Mesh<3>) -> Vec<[usize; 4]> {
    mesh.connectivities()
        .iter()
        .flat_map(|block| match block {
            Connectivity::Tetrahedral(elements) => elements.iter().copied(),
            _ => panic!("expected a tetrahedral block"),
        })
        .collect()
}

fn conforming(mesh: &Mesh<3>) {
    let coordinates = mesh.coordinates();
    let mut lo = [Scalar::INFINITY; 3];
    let mut hi = [Scalar::NEG_INFINITY; 3];
    (0..mesh.number_of_nodes()).for_each(|n| {
        (0..3).for_each(|a| {
            lo[a] = lo[a].min(coordinates[n][a].value());
            hi[a] = hi[a].max(coordinates[n][a].value())
        })
    });
    let span = (0..3).fold(0.0_f64, |m, a| m.max(hi[a] - lo[a]));
    let on_hull = |face: &[usize; 3]| {
        (0..3).any(|a| {
            let coincident = |bound: f64| {
                face.iter()
                    .all(|&n| (coordinates[n][a].value() - bound).abs() < 1.0e-9 * span)
            };
            coincident(lo[a]) || coincident(hi[a])
        })
    };
    let mut faces = FxHashMap::<[usize; 3], usize>::default();
    tets(mesh).iter().for_each(|tet| {
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
            .iter()
            .for_each(|local| {
                let mut face = [tet[local[0]], tet[local[1]], tet[local[2]]];
                face.sort_unstable();
                *faces.entry(face).or_default() += 1
            })
    });
    let interior_once = faces
        .iter()
        .filter(|&(face, &count)| count == 1 && !on_hull(face))
        .count();
    assert_eq!(interior_once, 0, "non-conforming: interior face used once");
    let shared = faces.values().filter(|&&count| count == 2).count();
    let boundary = faces.values().filter(|&&count| count == 1).count();
    assert_eq!(
        shared + boundary,
        faces.len(),
        "a face on three or more tets"
    );
    assert_eq!(boundary, mesh.exterior_faces().len())
}

fn check(octree: Tree) {
    let (leaves, volume) = occupancy(&octree);
    let (mesh, plain, templated) = Mesh::tetrahedra(octree);
    assert_eq!(plain + templated, leaves);
    assert!(plain > 0, "no leaf took the plain branch");
    assert!(templated > 0, "no leaf took the template branch");
    conforming(&mesh);
    let worst = mesh
        .minimum_scaled_jacobians()
        .into_iter()
        .flatten()
        .fold(Scalar::INFINITY, Scalar::min);
    assert!(worst > 0.0, "{worst}");
    let total: Scalar = mesh.volumes().into_iter().flatten().sum();
    assert!(
        (total - volume).abs() < 1.0e-9 * volume,
        "{total} vs {volume}"
    );
    // Every templated leaf is config B at eight tetrahedra or config A at
    // fourteen, so the total lies between those multiples.
    let templated_tets = mesh.number_of_elements() - 6 * plain;
    assert!(
        (8 * templated..=14 * templated).contains(&templated_tets),
        "{templated_tets} over {templated}"
    )
}

#[test]
fn graded_sphere() {
    [1.0, 2.5].iter().for_each(|&scale| {
        check(tree(&ball(8, 12, 1.0), scale));
    })
}

#[test]
fn graded_box() {
    [1.0, 2.5].iter().for_each(|&scale| {
        check(tree(
            &box_surface([-1.0, -0.6, -0.2], [1.0, 0.6, 0.2]),
            scale,
        ));
    })
}

#[test]
fn a_uniform_tree_is_six_tets_a_leaf() {
    let mut octree = super::super::test::octree(2);
    octree.subdivide(0).unwrap();
    let (leaves, volume) = occupancy(&octree);
    assert_eq!(leaves, 8);
    let mesh = Mesh::tetrahedra_from(octree);
    assert_eq!(mesh.number_of_elements(), 6 * leaves);
    conforming(&mesh);
    let total: Scalar = mesh.volumes().into_iter().flatten().sum();
    assert!(
        (total - volume).abs() < 1.0e-12 * volume,
        "{total} vs {volume}"
    )
}

/// Tetrahedra and nodes produced when every patch carrying a hanging node was
/// fanned about an added centre, which the Steiner-free rules must stay under.
const CENTRE_FAN: [(usize, usize); 2] = [(4032, 805), (1584, 343)];

/// The same once those rules landed, which cutting a Kuhn cell along its one
/// hanging edge must in turn stay under.
const PATCH_FAN: [(usize, usize); 2] = [(3264, 605), (1296, 271)];

/// And once that cut landed, which the config-A template must stay under.
const KUHN_CUT: [(usize, usize); 2] = [(2952, 557), (1140, 247)];

#[test]
fn steiner_free_patches_shrink_the_mesh() {
    [
        box_surface([-1.0, -0.6, -0.2], [1.0, 0.6, 0.2]),
        ball(8, 12, 1.0),
    ]
    .iter()
    .enumerate()
    .for_each(|(case, tessellation)| {
        let (mesh, _, templated) = Mesh::tetrahedra(tree(tessellation, 1.0));
        assert!(templated > 0);
        conforming(&mesh);
        let (tets, nodes) = (mesh.number_of_elements(), mesh.number_of_nodes());
        // Each stage strictly under the last: centre fan, then the
        // Steiner-free patch rules, then the Kuhn cut, then config A.
        for &(stage_tets, stage_nodes) in &[CENTRE_FAN[case], PATCH_FAN[case], KUHN_CUT[case]] {
            assert!(tets < stage_tets && nodes < stage_nodes, "{tets} / {nodes}");
        }
    })
}

/// KNOWN LIMITATION: a leaf facing a neighbor more than one level finer is not
/// matched against the whole of that neighbor's refinement, so the two sides
/// cut the shared patch differently. Predates the Steiner-free patch rules,
/// which fail this identically. 2:1 balancing avoids it.
#[test]
#[ignore = "tetrahedra requires a 2:1 balanced tree"]
fn an_unbalanced_tree_still_conforms() {
    let mut octree = super::super::test::octree(4);
    octree.subdivide(0).unwrap();
    octree.subdivide(1).unwrap();
    octree.subdivide(9).unwrap();
    let (leaves, volume) = occupancy(&octree);
    let (mesh, plain, templated) = Mesh::tetrahedra(octree);
    assert_eq!(plain + templated, leaves);
    conforming(&mesh);
    let worst = mesh
        .minimum_scaled_jacobians()
        .into_iter()
        .flatten()
        .fold(Scalar::INFINITY, Scalar::min);
    assert!(worst > 0.0, "{worst}");
    let total: Scalar = mesh.volumes().into_iter().flatten().sum();
    assert!(
        (total - volume).abs() < 1.0e-9 * volume,
        "{total} vs {volume}"
    )
}

/// Every config-A template, in each of the six orientations the refined facet
/// can take, must bound exactly the triangles the patch rules give that cell.
///
/// That contract is what lets a config-A cell sit against a plain, a config-B
/// or a fallback neighbor, and it is asserted per orientation because the
/// rules order nodes absolutely: a template cannot be rotated onto another
/// orientation, so each is checked on its own.
#[test]
fn config_a_templates_bound_the_patch_triangles() {
    let mut seen = std::collections::BTreeSet::new();
    for tessellation in [
        box_surface([-1.0, -0.6, -0.2], [1.0, 0.6, 0.2]),
        ball(8, 12, 1.0),
        box_surface([-2.0, -1.0, -0.15], [2.0, 1.0, 0.15]),
    ] {
        for scale in [1.0, 2.5] {
            let octree = tree(&tessellation, scale);
            let (leaves, _) = leaves(&octree);
            let facets = super::Facets::<3>::new(&octree, &leaves);
            let mut builder = super::Builder::new(&facets);
            for &index in &leaves {
                let (corner, length) = super::corner_length(&octree.nodes[index]);
                let polygons = facets.leaf_polygons(&octree, index);
                let refined: Vec<usize> = (0..6).filter(|&f| polygons[f].len() > 1).collect();
                if refined.len() != 1 || length % 2 != 0 {
                    continue;
                }
                let mut hanging = Vec::new();
                for (edge, &(axis, low)) in super::CUBE_EDGES.iter().enumerate() {
                    let mut key: [usize; 3] =
                        std::array::from_fn(|a| corner[a] + ((low >> a) & 1) * length);
                    key[axis] += length / 2;
                    if let Some(node) = facets.node(&key) {
                        hanging.push((edge, node))
                    }
                }
                if hanging.len() != 4 || !super::on_facet(&hanging, refined[0]) {
                    continue;
                }
                let tets =
                    super::config_a(&facets, corner, length, refined[0]).expect("template missing");
                assert_eq!(tets.len(), 14);
                let mut counts = FxHashMap::<[usize; 3], usize>::default();
                tets.iter().for_each(|tet| {
                    [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
                        .iter()
                        .for_each(|local| {
                            let mut face = [tet[local[0]], tet[local[1]], tet[local[2]]];
                            face.sort_unstable();
                            *counts.entry(face).or_default() += 1
                        })
                });
                let bound: std::collections::BTreeSet<[usize; 3]> = counts
                    .iter()
                    .filter(|&(_, &count)| count == 1)
                    .map(|(&face, _)| face)
                    .collect();
                assert!(counts.values().all(|&count| count <= 2));
                let wanted: std::collections::BTreeSet<[usize; 3]> = polygons
                    .iter()
                    .flatten()
                    .flat_map(|polygon| builder.triangles(polygon))
                    .map(|mut face| {
                        face.sort_unstable();
                        face
                    })
                    .collect();
                assert_eq!(wanted.len(), 22);
                assert_eq!(bound, wanted, "facet {}", refined[0]);
                seen.insert(refined[0]);
            }
        }
    }
    assert_eq!(seen.len(), 6, "orientations covered: {seen:?}")
}
