use super::super::{D, N};
use super::{Template, apply, transition_21};
use crate::geometry::ntree::{
    Octree,
    balance::Balancing,
    node::{Kind, split::Split},
};
use std::{array::from_fn, ops::Add};

fn weak_vertex_depths(fine: usize) -> [usize; 8] {
    let mut depths = [2; 8];
    depths[fine] = 3;
    depths[7 - fine] = 1;
    depths
}

#[test]
fn write_weak_vertex_dual() {
    use super::super::test::verify_dual;
    use crate::{
        geometry::{mesh::Output, ntree::Dualization},
        io::Write,
    };
    for fine in 0..8 {
        let mut octree =
            super::super::edge::test::weak_tree(weak_vertex_depths(fine), Balancing::Weak);
        let mesh = octree.dualize();
        if let Err(error) = verify_dual(&mesh) {
            panic!("weak vertex dual (fine orthant {fine}) failed verification: {error}");
        }
        if fine == 7 {
            mesh.write(Output::Exodus("target/weak_vertex.exo"))
                .unwrap();
        }
    }
}

#[test]
fn transition_35_fills_weak_vertex_config_only() {
    use crate::geometry::ntree::dual::Uniform;
    let hexes = |fine, balancing| {
        let octree = super::super::edge::test::weak_tree(weak_vertex_depths(fine), balancing);
        let (center_nodes, ..) = octree.initialize();
        let mut connectivity = Vec::new();
        super::transition_35::template(&octree, &center_nodes, &mut connectivity);
        connectivity.len()
    };
    for fine in 0..8 {
        assert_eq!(
            hexes(fine, Balancing::Weak),
            1,
            "transition_35 should fill the weak vertex star (fine orthant {fine})"
        );
        assert_eq!(
            hexes(fine, Balancing::Strong),
            0,
            "transition_35 fired on the strong tree (the config should be balanced away)"
        );
    }
}

pub(crate) fn vertex_dual_generic<T, U>(
    tree: &Octree<T, U>,
    center_nodes: &[usize],
) -> Vec<[usize; N]>
where
    T: Copy + Into<usize> + Add<Output = T> + PartialOrd + Split,
    U: Copy + Into<usize>,
{
    const WIND: [usize; N] = [0, 1, 3, 2, 4, 5, 7, 6];
    let root = &tree.nodes[0];
    let lo = root.corner;
    let hi: [T; D] = from_fn(|a| root.corner[a] + root.length);
    let mut hexes = Vec::new();
    for node in tree.iter().filter(|node| node.is_leaf()) {
        let v: [T; D] = from_fn(|a| node.corner[a] + node.length);
        if (0..D).all(|a| lo[a] < v[a] && v[a] < hi[a]) {
            let cells: [usize; N] = from_fn(|d| find_leaf_octant(tree, &v, d));
            hexes.push(from_fn(|k| center_nodes[cells[WIND[k]]]));
        }
    }
    hexes
}

fn find_leaf_octant<T, U>(tree: &Octree<T, U>, v: &[T; D], d: usize) -> usize
where
    T: Copy + Add<Output = T> + PartialOrd + Split,
    U: Copy + Into<usize>,
{
    let mut index = 0;
    loop {
        match &tree.nodes[index].kind {
            Kind::Leaf => return index,
            Kind::Tree(orthants) => {
                let corner = tree.nodes[index].corner;
                let half = tree.nodes[index].length.split();
                let child = (0..D).fold(0, |acc, a| {
                    let mid = corner[a] + half;
                    let bit = if v[a] > mid {
                        1
                    } else if v[a] < mid {
                        0
                    } else {
                        (d >> a) & 1
                    };
                    acc | (bit << a)
                });
                index = orthants[child].into();
            }
        }
    }
}

fn one_template<T, U>(
    tree: &Octree<T, U>,
    center_nodes: &[usize],
    data: &[[usize; 11]],
    template: Template<T, U>,
) -> Vec<[usize; N]>
where
    T: Copy + Into<usize>,
    U: Copy + Into<usize>,
{
    let mut connectivity = Vec::new();
    apply(tree, center_nodes, &mut connectivity, data, template);
    connectivity
}

pub(crate) fn transition_21_only<T, U>(
    tree: &Octree<T, U>,
    center_nodes: &[usize],
) -> Vec<[usize; N]>
where
    T: Copy + Into<usize>,
    U: Copy + Into<usize>,
{
    one_template(
        tree,
        center_nodes,
        &transition_21::DATA,
        transition_21::template,
    )
}
