use crate::{
    geometry::{
        Coordinate, Coordinates,
        grid::Voxels,
        mesh::{Connectivity, Mesh, Verdict},
    },
    math::{FxHashMap, Tensor, TensorVec},
};

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

fn dense(voxels: Voxels<u8>, remove: Option<&[u8]>) -> (usize, Vec<[usize; 8]>) {
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
    (coordinates.len(), connectivity)
}

#[test]
fn sparse_node_numbering_matches_dense() {
    let n = 24;
    let (dense_nodes, dense_connectivity) = dense(rod_voxels(n), Some(&[0]));
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

fn box_cells(nel: [usize; 3]) -> Vec<([usize; 3], usize)> {
    let mut cells = Vec::new();
    for k in 0..nel[2] {
        for j in 0..nel[1] {
            for i in 0..nel[0] {
                cells.push(([i, j, k], 1))
            }
        }
    }
    cells
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

#[test]
fn tets_are_six_per_cell() {
    let nel = [3, 4, 5];
    let cells = box_cells(nel);
    let mesh = Mesh::from_lattice_tets(cells.clone(), nel, &UNIT, &ORIGIN);
    assert_eq!(mesh.number_of_elements(), 6 * cells.len());
    assert_eq!(mesh.number_of_element_blocks(), 1);
    assert_eq!(mesh.number_of_nodes(), 4 * 5 * 6)
}

#[test]
fn tets_are_conforming() {
    let nel = [3, 4, 5];
    let mesh = Mesh::from_lattice_tets(box_cells(nel), nel, &UNIT, &ORIGIN);
    let mut faces = FxHashMap::<[usize; 3], usize>::default();
    tets(&mesh).iter().for_each(|tet| {
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
            .iter()
            .for_each(|local| {
                let mut face = [tet[local[0]], tet[local[1]], tet[local[2]]];
                face.sort_unstable();
                *faces.entry(face).or_default() += 1
            })
    });
    assert!(faces.values().all(|&count| count == 1 || count == 2));
    let boundary = faces.values().filter(|&&count| count == 1).count();
    assert_eq!(boundary, mesh.exterior_faces().len());
    let [nx, ny, nz] = nel;
    assert_eq!(boundary, 4 * (nx * ny + ny * nz + nz * nx))
}

#[test]
fn tets_are_positively_oriented() {
    let nel = [2, 3, 4];
    [
        (UNIT, ORIGIN),
        (
            Coordinate::const_from([0.5, 2.0, 1.5]),
            Coordinate::const_from([-3.0, 1.0, 7.0]),
        ),
    ]
    .iter()
    .for_each(|(scale, translate)| {
        let mesh = Mesh::from_lattice_tets(box_cells(nel), nel, scale, translate);
        let worst = mesh
            .minimum_scaled_jacobians()
            .into_iter()
            .flatten()
            .fold(f64::INFINITY, f64::min);
        assert!(worst > 0.0, "{worst}")
    })
}

#[test]
fn tets_fill_the_lattice() {
    let nel = [3, 4, 5];
    let scale = Coordinate::const_from([0.5, 2.0, 1.5]);
    let cells = box_cells(nel);
    let mesh = Mesh::from_lattice_tets(cells.clone(), nel, &scale, &ORIGIN);
    let total: f64 = mesh.volumes().into_iter().flatten().sum();
    let cell = scale[0].value() * scale[1].value() * scale[2].value();
    assert!((total - cell * cells.len() as f64).abs() < 1e-12, "{total}")
}

#[test]
fn tets_carry_materials_like_cells() {
    let nel = [2, 2, 2];
    let cells: Vec<_> = box_cells(nel)
        .into_iter()
        .enumerate()
        .map(|(index, (cell, _))| (cell, 1 + index % 3))
        .collect();
    let mesh = Mesh::from_lattice_tets(cells, nel, &UNIT, &ORIGIN);
    assert_eq!(mesh.blocks(), Some(&[1, 2, 3][..]));
    mesh.connectivities()
        .iter()
        .for_each(|block| assert!(matches!(block, Connectivity::Tetrahedral(_))))
}
