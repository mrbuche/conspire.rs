use std::f64::consts::FRAC_PI_3;

use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{Connectivity, Mesh, Tessellation},
    },
    math::Tensor,
};
use std::path::Path;

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

const COORDINATES: [Coordinate<3>; 8] = [
    Coordinate::const_from([0.0, 0.0, 0.0]),
    Coordinate::const_from([1.0, 0.0, 0.0]),
    Coordinate::const_from([1.0, 1.0, 0.0]),
    Coordinate::const_from([0.0, 1.0, 0.0]),
    Coordinate::const_from([0.0, 0.0, 1.0]),
    Coordinate::const_from([1.0, 0.0, 1.0]),
    Coordinate::const_from([1.0, 1.0, 1.0]),
    Coordinate::const_from([0.0, 1.0, 1.0]),
];

fn cube() -> Tessellation {
    let connectivities = vec![Connectivity::Triangular(CONNECTIVITY.to_vec().into())];
    let coordinates = Coordinates::from(COORDINATES);
    Tessellation::from(Mesh::from((connectivities, coordinates)))
}

#[test]
fn cube_center_ray_is_unit_thickness() {
    let diameters = cube().shape_diameter_function(FRAC_PI_3, 0, 0);
    assert_eq!(diameters.len(), 8);
    diameters
        .iter()
        .for_each(|&diameter| assert_eq!(diameter, 1.0));
}

#[test]
fn cube_cone_stays_near_unit_thickness() {
    let diameters = cube().shape_diameter_function(FRAC_PI_3, 3, 8);
    assert_eq!(diameters.len(), 8);
    diameters.iter().for_each(|&diameter| {
        assert!(diameter > 0.0 && diameter.is_finite());
        assert!((diameter - 1.0).abs() < 0.5);
    });
}

#[test]
fn bunny() {
    let tessellation =
        Tessellation::try_from(Path::new("/home/mrbuche/Downloads/Stanford_Bunny.stl")).unwrap();
    println!("triangles: {}", tessellation.mesh().connectivities().iter().flatten().count());
    let diameters = tessellation.shape_diameter_function(FRAC_PI_3, 3, 8);
    assert_eq!(diameters.len(), tessellation.mesh().coordinates().len());
    // println!("{:?}", diameters);
}
