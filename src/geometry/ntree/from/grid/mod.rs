#[cfg(test)]
mod test;

use crate::{
    geometry::{
        grid::Grid,
        ntree::{
            Orthotree,
            balance::Balancing,
            node::{Kind, Node, split::Split},
            pair::Pairing,
            rescale::Rescaling,
        },
    },
    math::Scalar,
};
use std::{array::from_fn, ops::Add};

type Pyramid<const D: usize, V> = Vec<([usize; D], Vec<Option<V>>)>;

enum Cell<V> {
    Empty,
    Uniform(V),
    Mixed,
}

impl<const D: usize, const L: usize, const M: usize, const N: usize, T, U, V> From<Grid<D, V>>
    for Orthotree<D, L, M, N, T, U, V>
where
    T: Add<Output = T> + Copy + From<u16> + Into<usize> + Split,
    U: Copy + From<usize> + Into<usize>,
    V: Copy + PartialEq,
{
    fn from(grid: Grid<D, V>) -> Self {
        let nel = *grid.nel();
        let max = nel.iter().copied().max().unwrap_or(0).max(1);
        let mut root_length = 1u16;
        while (root_length as usize) < max {
            root_length <<= 1;
        }
        let half = root_length as Scalar / 2.0;
        let mut tree = Self {
            balanced: Balancing::None,
            nodes: vec![Node {
                corner: from_fn(|_| T::from(0)),
                length: T::from(root_length),
                facets: [None; M],
                kind: Kind::Leaf,
                value: None,
            }],
            paired: Pairing::None,
            rescale: Rescaling {
                center: [half; D],
                cell: 1.0,
                half,
            },
        };
        let pyramid = pyramid(
            &nel,
            root_length.trailing_zeros(),
            grid.data_col_major().into_owned(),
        );
        let mut index = 0;
        while index < tree.len() {
            let node = &tree.nodes[index];
            let corner = from_fn(|ax| node.corner[ax].into());
            let length = node.length.into();
            match classify(corner, length, &nel, &pyramid) {
                Cell::Uniform(value) => tree.nodes[index].value = Some(value),
                Cell::Mixed => {
                    tree.subdivide(U::from(index)).ok();
                }
                Cell::Empty => {}
            }
            index += 1;
        }
        tree
    }
}

fn pyramid<const D: usize, V: Copy + PartialEq>(
    nel: &[usize; D],
    levels: u32,
    data: Vec<V>,
) -> Pyramid<D, V> {
    let mut out: Pyramid<D, V> = vec![(*nel, data.into_iter().map(Some).collect())];
    for _ in 0..levels {
        let (dim, prev) = out.last().unwrap();
        let dim = *dim;
        let next_dim: [usize; D] = from_fn(|ax| dim[ax].div_ceil(2));
        let mut next = vec![None; next_dim.iter().product()];
        for (cell, slot) in next.iter_mut().enumerate() {
            let base = unflatten(cell, &next_dim);
            let mut value = None;
            let mut uniform = true;
            'gather: for child in 0..(1usize << D) {
                let coord = from_fn(|ax| 2 * base[ax] + ((child >> ax) & 1));
                if (0..D).any(|ax| coord[ax] >= dim[ax]) {
                    continue;
                }
                match prev[flatten(&coord, &dim)] {
                    Some(entry) if value.is_none_or(|seen| seen == entry) => value = Some(entry),
                    _ => {
                        uniform = false;
                        break 'gather;
                    }
                }
            }
            *slot = uniform.then_some(value).flatten();
        }
        out.push((next_dim, next));
    }
    out
}

fn flatten<const D: usize>(coord: &[usize; D], dim: &[usize; D]) -> usize {
    let mut offset = 0;
    let mut stride = 1;
    for (c, n) in coord.iter().zip(dim) {
        offset += c * stride;
        stride *= n;
    }
    offset
}

fn unflatten<const D: usize>(mut index: usize, dim: &[usize; D]) -> [usize; D] {
    from_fn(|ax| {
        let coord = index % dim[ax];
        index /= dim[ax];
        coord
    })
}

fn classify<const D: usize, V: Copy>(
    corner: [usize; D],
    length: usize,
    nel: &[usize; D],
    pyramid: &Pyramid<D, V>,
) -> Cell<V> {
    if (0..D).any(|ax| corner[ax] >= nel[ax]) {
        return Cell::Empty;
    }
    if (0..D).any(|ax| corner[ax] + length > nel[ax]) {
        return Cell::Mixed;
    }
    let (dim, data) = &pyramid[length.trailing_zeros() as usize];
    let cell = from_fn(|ax| corner[ax] / length);
    match data[flatten(&cell, dim)] {
        Some(value) => Cell::Uniform(value),
        None => Cell::Mixed,
    }
}
