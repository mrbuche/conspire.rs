#[cfg(test)]
pub(crate) mod test;

mod transition_1;
mod transition_10;
mod transition_11;
mod transition_12;
mod transition_13;
mod transition_14;
mod transition_15;
mod transition_16;
mod transition_17;
mod transition_18;
mod transition_19;
mod transition_2;
mod transition_20;
mod transition_21;
mod transition_3;
mod transition_4;
mod transition_5;
mod transition_6;
mod transition_7;
mod transition_8;
mod transition_9;

use super::{D, L, M, N};
use crate::geometry::ntree::{Octree, node::Node};

type Template<T, U> =
    fn(&Octree<T, U>, &Node<D, M, N, T, U>, &[U; N], &[usize], [usize; 11]) -> Option<[usize; N]>;

type Entry<'a, T, U> = (&'a [[usize; 11]], Template<T, U>);

pub fn vertex_transitions<T, U>(
    tree: &Octree<T, U>,
    center_nodes: &[usize],
    connectivity: &mut Vec<[usize; N]>,
) where
    T: Copy + Into<usize>,
    U: Copy + Into<usize>,
{
    let templates: [Entry<T, U>; 21] = [
        (&transition_1::DATA, transition_1::template),
        (&transition_2::DATA, transition_2::template),
        (&transition_3::DATA, transition_3::template),
        (&transition_4::DATA, transition_4::template),
        (&transition_5::DATA, transition_5::template),
        (&transition_6::DATA, transition_6::template),
        (&transition_7::DATA, transition_7::template),
        (&transition_8::DATA, transition_8::template),
        (&transition_9::DATA, transition_9::template),
        (&transition_10::DATA, transition_10::template),
        (&transition_11::DATA, transition_11::template),
        (&transition_12::DATA, transition_12::template),
        (&transition_13::DATA, transition_13::template),
        (&transition_14::DATA, transition_14::template),
        (&transition_15::DATA, transition_15::template),
        (&transition_16::DATA, transition_16::template),
        (&transition_17::DATA, transition_17::template),
        (&transition_18::DATA, transition_18::template),
        (&transition_19::DATA, transition_19::template),
        (&transition_20::DATA, transition_20::template),
        (&transition_21::DATA, transition_21::template),
    ];
    for (data, template) in templates {
        apply(tree, center_nodes, connectivity, data, template)
    }
}

fn apply<T, U>(
    tree: &Octree<T, U>,
    center_nodes: &[usize],
    connectivity: &mut Vec<[usize; N]>,
    data: &[[usize; 11]],
    template: Template<T, U>,
) where
    T: Copy + Into<usize>,
    U: Copy + Into<usize>,
{
    for node in tree.iter() {
        if let Some(cell_subcells) = tree.all_leaves(node) {
            for &row in data {
                if let Some(hex) = template(tree, node, cell_subcells, center_nodes, row) {
                    connectivity.push(hex)
                }
            }
        }
    }
}

fn pick<T, U>(tree: &Octree<T, U>, cell: U, facet: usize, idx: usize, fine: bool) -> Option<U>
where
    T: Copy + Into<usize>,
    U: Copy + Into<usize>,
{
    if fine {
        sub_subnode(tree, cell, facet, idx)
    } else {
        Some(tree.all_leaves(&tree.nodes[cell.into()])?[idx])
    }
}

fn face_plus_two<T, U>(
    tree: &Octree<T, U>,
    node: &Node<D, M, N, T, U>,
    cell_subcells: &[U; N],
    center_nodes: &[usize],
    data: [usize; 11],
    fine: [bool; 7],
    cab_b: bool,
) -> Option<[usize; N]>
where
    T: Copy + Into<usize>,
    U: Copy + Into<usize>,
{
    let [f, fa, fb, sc, x_a, x_ab, x_b, x_c, x_ca, x_cab, x_cb] = data;
    let [is_a, is_ab, is_b, is_c, is_ca, is_cab, is_cb] = fine;
    let cell_a = node.facets[fa]?;
    let cell_b = node.facets[fb]?;
    let cell_ab = tree.nodes[cell_a.into()].facets[fb]?;
    let cell_c = node.facets[f]?;
    let cell_c_a = tree.nodes[cell_a.into()].facets[f]?;
    let cell_c_b = tree.nodes[cell_b.into()].facets[f]?;
    let cell_c_ab = tree.nodes[cell_ab.into()].facets[f]?;
    let cab_facet = if cab_b { fb } else { f };
    Some([
        center_nodes[cell_subcells[sc].into()],
        center_nodes[pick(tree, cell_a, fa, x_a, is_a)?.into()],
        center_nodes[pick(tree, cell_ab, fa, x_ab, is_ab)?.into()],
        center_nodes[pick(tree, cell_b, fb, x_b, is_b)?.into()],
        center_nodes[pick(tree, cell_c, f, x_c, is_c)?.into()],
        center_nodes[pick(tree, cell_c_a, f, x_ca, is_ca)?.into()],
        center_nodes[pick(tree, cell_c_ab, cab_facet, x_cab, is_cab)?.into()],
        center_nodes[pick(tree, cell_c_b, f, x_cb, is_cb)?.into()],
    ])
}

fn three_face<T, U>(
    tree: &Octree<T, U>,
    node: &Node<D, M, N, T, U>,
    cell_subcells: &[U; N],
    center_nodes: &[usize],
    data: [usize; 11],
    fine: [bool; 7],
    cb_c: bool,
) -> Option<[usize; N]>
where
    T: Copy + Into<usize>,
    U: Copy + Into<usize>,
{
    let [fa, fb, fc, sc, x_a, x_ab, x_b, x_c, x_ca, x_cab, x_cb] = data;
    let [is_a, is_ab, is_b, is_c, is_ca, is_cab, is_cb] = fine;
    let cell_a = node.facets[fa]?;
    let cell_b = node.facets[fb]?;
    let cell_ab = tree.nodes[cell_a.into()].facets[fb]?;
    let cell_c = node.facets[fc]?;
    let cell_c_a = tree.nodes[cell_c.into()].facets[fa]?;
    let cell_c_b = tree.nodes[cell_c.into()].facets[fb]?;
    let cell_c_ab = tree.nodes[cell_c_a.into()].facets[fb]?;
    let cb_facet = if cb_c { fc } else { fb };
    Some([
        center_nodes[pick(tree, cell_c, fc, x_c, is_c)?.into()],
        center_nodes[pick(tree, cell_c_a, fa, x_ca, is_ca)?.into()],
        center_nodes[pick(tree, cell_c_ab, fb, x_cab, is_cab)?.into()],
        center_nodes[pick(tree, cell_c_b, cb_facet, x_cb, is_cb)?.into()],
        center_nodes[cell_subcells[sc].into()],
        center_nodes[pick(tree, cell_a, fa, x_a, is_a)?.into()],
        center_nodes[pick(tree, cell_ab, fb, x_ab, is_ab)?.into()],
        center_nodes[pick(tree, cell_b, fb, x_b, is_b)?.into()],
    ])
}

fn sub_subnode<T, U>(tree: &Octree<T, U>, neighbor: U, facet: usize, idx: usize) -> Option<U>
where
    T: Copy + Into<usize>,
    U: Copy + Into<usize>,
{
    tree.orthants_leaves_on_facet(&tree.nodes[neighbor.into()], facet ^ 1)[idx / L]
        .and_then(|inner| inner[idx % L])
}
