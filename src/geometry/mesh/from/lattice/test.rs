use crate::{
    geometry::{Coordinate, Coordinates, grid::Voxels, mesh::Mesh},
    math::{Tensor, TensorVec},
};
use std::time::Instant;

const RADIUS: usize = 3;
const UNIT: Coordinate<3> = Coordinate::const_from([1.0; 3]);
const ORIGIN: Coordinate<3> = Coordinate::const_from([0.0; 3]);

fn rod(n: usize) -> Vec<([usize; 3], usize)> {
    let mut cells = Vec::new();
    for k in 0..n {
        for j in k.saturating_sub(RADIUS)..(k + RADIUS + 1).min(n) {
            for i in k.saturating_sub(RADIUS)..(k + RADIUS + 1).min(n) {
                cells.push(([i, j, k], 1));
            }
        }
    }
    cells
}

fn rod_voxels(n: usize) -> Voxels<u8> {
    let mut data = vec![0u8; n * n * n];
    rod(n)
        .iter()
        .for_each(|([i, j, k], _)| data[i + n * j + n * n * k] = 1);
    Voxels::new(data, [n, n, n])
}

fn dense(voxels: Voxels<u8>, remove: Option<&[u8]>) -> (usize, usize, Vec<[usize; 8]>) {
    let [nx, ny, nz] = *voxels.nel();
    let (nxp, nyp, nzp) = (nx + 1, ny + 1, nz + 1);
    let layer = nxp * nyp;
    let nodes_unfiltered = layer * nzp;
    let mut connectivity = Vec::new();
    for ([i, j, k], &block) in voxels.logical_iter() {
        if remove.is_none_or(|ids| !ids.contains(&block)) {
            let base = i + nxp * j + layer * k;
            let top = base + layer;
            connectivity.push([
                base,
                base + 1,
                base + nxp + 1,
                base + nxp,
                top,
                top + 1,
                top + nxp + 1,
                top + nxp,
            ]);
        }
    }
    let mut used = vec![false; nodes_unfiltered];
    connectivity
        .iter()
        .for_each(|nodes| nodes.iter().for_each(|&node| used[node] = true));
    let mut mapping = vec![0usize; nodes_unfiltered];
    let mut coordinates = Coordinates::new();
    for (old, &is_used) in used.iter().enumerate() {
        if is_used {
            mapping[old] = coordinates.len();
            coordinates.push(Coordinate::const_from([
                (old % nxp) as f64,
                (old / nxp % nyp) as f64,
                (old / layer) as f64,
            ]));
        }
    }
    connectivity
        .iter_mut()
        .for_each(|nodes| nodes.iter_mut().for_each(|node| *node = mapping[*node]));
    (nodes_unfiltered, coordinates.len(), connectivity)
}

#[test]
fn sparse_node_numbering_matches_dense() {
    let n = 24;
    let (_, dense_nodes, dense_connectivity) = dense(rod_voxels(n), Some(&[0]));
    let mesh = Mesh::from_voxels(rod_voxels(n), Some(&[0]));
    assert_eq!(mesh.number_of_nodes(), dense_nodes);
    assert_eq!(mesh.number_of_elements(), dense_connectivity.len());
    let sparse_connectivity: Vec<&[usize]> = mesh.connectivities().iter().flatten().collect();
    assert_eq!(sparse_connectivity.len(), dense_connectivity.len());
    sparse_connectivity
        .iter()
        .zip(dense_connectivity.iter())
        .for_each(|(sparse, dense)| assert_eq!(*sparse, &dense[..]));
}

#[test]
fn empty_bounding_box_costs_nothing() {
    let mesh = Mesh::from_lattice_cells(
        [([0, 0, 0], 1)],
        [1 << 20, 1 << 20, 1 << 20],
        &UNIT,
        &ORIGIN,
    );
    assert_eq!(mesh.number_of_nodes(), 8);
    assert_eq!(mesh.number_of_elements(), 1);
}

#[test]
#[ignore = "benchmark; run with --release -- --ignored --nocapture --test-threads=1"]
fn bench_dense_versus_sparse() {
    println!(
        "{:>6}  {:>9}  {:>12}  {:>11}  {:>11}  {:>10}  {:>10}  {:>10}",
        "n", "cells", "bbox", "dense (MB)", "sparse (MB)", "dense ms", "voxels ms", "cells ms"
    );
    for n in [64, 128, 256, 512] {
        let cells = rod(n);
        let start = Instant::now();
        let (unfiltered, nodes, _) = dense(rod_voxels(n), Some(&[0]));
        let dense_ms = start.elapsed().as_secs_f64() * 1e3;
        let start = Instant::now();
        let from_voxels = Mesh::from_voxels(rod_voxels(n), Some(&[0]));
        let voxels_ms = start.elapsed().as_secs_f64() * 1e3;
        let start = Instant::now();
        let from_cells = Mesh::from_lattice_cells(cells.iter().copied(), [n, n, n], &UNIT, &ORIGIN);
        let cells_ms = start.elapsed().as_secs_f64() * 1e3;
        assert_eq!(from_voxels.number_of_nodes(), nodes);
        assert_eq!(from_cells.number_of_nodes(), nodes);
        let dense_mb = (unfiltered * 9) as f64 / 1e6;
        let sparse_mb = (cells.len() * 8 * 8) as f64 / 1e6;
        println!(
            "{n:>6}  {:>9}  {unfiltered:>12}  {dense_mb:>11.1}  {sparse_mb:>11.1}  {dense_ms:>10.1}  {voxels_ms:>10.1}  {cells_ms:>10.1}",
            cells.len(),
        );
    }
}
