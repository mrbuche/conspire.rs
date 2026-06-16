use crate::{
    geometry::{
        Coordinate, Coordinates,
        grid::Voxels,
        mesh::{Connectivity, Mesh, Tessellation},
    },
    math::TensorVec,
};

const CONNECTIVITY: [[usize; 3]; 12] = [
    [0, 2, 1],
    [0, 3, 2],
    [4, 5, 6],
    [4, 6, 7],
    [0, 1, 5],
    [0, 5, 4],
    [3, 6, 2],
    [3, 7, 6],
    [0, 4, 7],
    [0, 7, 3],
    [1, 2, 6],
    [1, 6, 5],
];

const UNIT_CUBE: [[f64; 3]; 8] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
];

fn two_cubes() -> Tessellation {
    let mut coordinates = Coordinates::new();
    for shift in [0.0, 2.0] {
        UNIT_CUBE.iter().for_each(|point| {
            coordinates.push(Coordinate::const_from([
                point[0] + shift,
                point[1],
                point[2],
            ]))
        });
    }
    let mut connectivity = CONNECTIVITY.to_vec();
    connectivity.extend(CONNECTIVITY.iter().map(|t| [t[0] + 8, t[1] + 8, t[2] + 8]));
    let connectivities = vec![Connectivity::Triangular(connectivity.into())];
    Tessellation::from(Mesh::from((connectivities, coordinates)))
}

#[test]
fn solid_voxelization_inside_outside() {
    let voxels = Voxels::from_tessellation(&two_cubes(), 1.0);
    assert_eq!(*voxels.nel(), [3, 1, 1]);
    assert_eq!(voxels.data(), [1, 0, 1]);
}
