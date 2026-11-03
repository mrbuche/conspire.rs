#[cfg(test)]
mod test;

use crate::{
    geometry::mesh::quality::metrics::{pyramid, tetrahedron, wedge},
    math::Scalar,
};
use std::{array::from_fn, f64::consts::SQRT_2, sync::LazyLock};

pub(super) type Weights = [[Scalar; 8]; 3];

pub(super) struct Point {
    pub(super) sparse: [Vec<(usize, Scalar)>; 3],
    pub(super) weights: Weights,
}

pub(super) static HEXAHEDRON_POINTS: LazyLock<[Point; 9]> = LazyLock::new(|| {
    let corners = [
        (0, [1, 3, 4]),
        (1, [2, 0, 5]),
        (2, [3, 1, 6]),
        (3, [0, 2, 7]),
        (4, [7, 5, 0]),
        (5, [4, 6, 1]),
        (6, [5, 7, 2]),
        (7, [6, 4, 3]),
    ];
    let faces = [
        ([1, 2, 5, 6], [0, 3, 4, 7]),
        ([2, 3, 6, 7], [0, 1, 4, 5]),
        ([4, 5, 6, 7], [0, 1, 2, 3]),
    ];
    from_fn(|point| {
        let mut weights = [[0.0; 8]; 3];
        if point < 8 {
            let (corner, adjacent) = corners[point];
            weights.iter_mut().zip(adjacent).for_each(|(row, node)| {
                row[node] += 1.0;
                row[corner] -= 1.0;
            })
        } else {
            weights
                .iter_mut()
                .zip(&faces)
                .for_each(|(row, (high, low))| {
                    high.iter().for_each(|&node| row[node] += 0.25);
                    low.iter().for_each(|&node| row[node] -= 0.25)
                })
        }
        let sparse = from_fn(|i| {
            (0..8)
                .filter(|&node| weights[i][node] != 0.0)
                .map(|node| (node, weights[i][node]))
                .collect()
        });
        Point { sparse, weights }
    })
});

const EDGES: [[usize; 2]; 12] = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
];

fn corner_points<const N: usize>(table: &[(usize, [usize; 3])]) -> [Point; N] {
    from_fn(|point| {
        let (corner, adjacent) = table[point];
        let mut weights = [[0.0; 8]; 3];
        weights.iter_mut().zip(adjacent).for_each(|(row, node)| {
            row[node] += 1.0;
            row[corner] -= 1.0;
        });
        let sparse = from_fn(|i| {
            (0..8)
                .filter(|&node| weights[i][node] != 0.0)
                .map(|node| (node, weights[i][node]))
                .collect()
        });
        Point { sparse, weights }
    })
}

static TETRAHEDRON_POINTS: LazyLock<[Point; 4]> =
    LazyLock::new(|| corner_points(&tetrahedron::CORNERS));

static WEDGE_POINTS: LazyLock<[Point; 6]> = LazyLock::new(|| corner_points(&wedge::CORNERS));

static PYRAMID_POINTS: LazyLock<[Point; 8]> = LazyLock::new(|| corner_points(&pyramid::CORNERS));

pub(super) struct Kind {
    pub(super) edges: &'static [[usize; 2]],
    pub(super) normalization: Scalar,
    pub(super) points: &'static [Point],
}

pub(super) const HEXAHEDRON: u8 = 0;
pub(super) const TETRAHEDRON: u8 = 1;
pub(super) const WEDGE: u8 = 2;
pub(super) const PYRAMID: u8 = 3;

pub(super) static KINDS: LazyLock<[Kind; 4]> = LazyLock::new(|| {
    [
        Kind {
            edges: &EDGES,
            normalization: 1.0,
            points: &*HEXAHEDRON_POINTS,
        },
        Kind {
            edges: &[[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
            normalization: SQRT_2,
            points: &*TETRAHEDRON_POINTS,
        },
        Kind {
            edges: &[
                [0, 1],
                [1, 2],
                [2, 0],
                [3, 4],
                [4, 5],
                [5, 3],
                [0, 3],
                [1, 4],
                [2, 5],
            ],
            normalization: 2.0 / 3.0_f64.sqrt(),
            points: &*WEDGE_POINTS,
        },
        Kind {
            edges: &[
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
                [0, 4],
                [1, 4],
                [2, 4],
                [3, 4],
            ],
            normalization: SQRT_2,
            points: &*PYRAMID_POINTS,
        },
    ]
});
