use super::{Finish, Marching, Placement};
use crate::{
    geometry::mesh::{
        Connectivity, Fitting, Freedom, Mesh, Verdict,
        quality::metrics::hexahedron::bernstein,
        tessellation::cut::test::{sphere, star},
    },
    math::Quantity,
};
use std::array::from_fn;

fn report(name: &str, mesh: &Mesh<3>) -> (usize, f64, usize) {
    let scaled = &mesh.minimum_scaled_jacobians()[0];
    let minimum = scaled.iter().cloned().fold(f64::INFINITY, f64::min);
    let negative = scaled.iter().filter(|&&value| value <= 0.0).count();
    let certified = match &mesh.connectivities()[0] {
        Connectivity::Hexahedral(hexes) => hexes
            .iter()
            .filter(|hex| bernstein::certifies(hex.as_ref(), mesh.coordinates()))
            .count(),
        _ => panic!(),
    };
    println!(
        "{name:>16}  {:>7} hexes  min SJ {minimum:>8.4}  {negative:>5} inverted  {:>5} uncertified",
        scaled.len(),
        scaled.len() - certified
    );
    (scaled.len(), minimum, negative)
}

#[test]
fn a_sphere_is_all_hexahedra_and_none_inverted() {
    let mesh = sphere(3)
        .marching_hex(
            Quantity::new(0.2),
            Marching {
                placement: Placement::Midpoint,
                finish: Finish::Cut,
            },
        )
        .unwrap();
    assert_eq!(mesh.number_of_element_blocks(), 1);
    assert!(matches!(
        &mesh.connectivities()[0],
        Connectivity::Hexahedral(_)
    ));
    let (count, minimum, negative) = report("sphere", &mesh);
    assert!(count > 0);
    assert_eq!(negative, 0, "min SJ {minimum}");
}

#[test]
fn a_creased_surface_is_all_hexahedra_and_none_inverted() {
    let mesh = star(1, 2.0)
        .marching_hex(
            Quantity::new(0.25),
            Marching {
                placement: Placement::Midpoint,
                finish: Finish::Cut,
            },
        )
        .unwrap();
    assert!(matches!(
        &mesh.connectivities()[0],
        Connectivity::Hexahedral(_)
    ));
    let (count, minimum, negative) = report("star", &mesh);
    assert!(count > 0);
    assert_eq!(negative, 0, "min SJ {minimum}");
}

#[test]
fn the_default_holds_quality_and_draws_the_boundary_close() {
    let tessellation = sphere(3);
    let spacing = Quantity::new(0.1);
    let mesh = tessellation
        .marching_hex(spacing, Marching::default())
        .unwrap();
    let scaled = &mesh.minimum_scaled_jacobians()[0];
    assert!(scaled.iter().all(|&value| value > 0.0));
    let hexes = match &mesh.connectivities()[0] {
        Connectivity::Hexahedral(hexes) => hexes.iter().copied().collect::<Vec<[usize; 8]>>(),
        _ => panic!(),
    };
    let (_, mean) = tessellation.conformance(&hexes, mesh.coordinates(), spacing);
    assert!(mean < 0.02, "{mean}");
}

#[test]
fn inflation_meshes_a_sphere_without_inverting() {
    let mesh = sphere(2)
        .marching_hex(
            Quantity::new(0.35),
            Marching {
                placement: Placement::Crossing(0.2),
                finish: Finish::Fit(Freedom::Shell, Fitting::Soft),
            },
        )
        .unwrap();
    let (count, minimum, negative) = report("sphere", &mesh);
    assert!(count > 0);
    assert_eq!(negative, 0, "min SJ {minimum}");
}

#[test]
fn snapping_puts_the_boundary_on_the_surface_without_inverting() {
    let tessellation = sphere(2);
    let mesh = tessellation
        .marching_hex(
            Quantity::new(0.35),
            Marching {
                placement: Placement::Crossing(0.2),
                finish: Finish::Fit(Freedom::Whole, Fitting::Snap),
            },
        )
        .unwrap();
    let (count, minimum, negative) = report("sphere snapped", &mesh);
    assert!(count > 0);
    assert_eq!(negative, 0, "min SJ {minimum}");
    let hexes = match &mesh.connectivities()[0] {
        Connectivity::Hexahedral(hexes) => hexes.iter().copied().collect::<Vec<[usize; 8]>>(),
        _ => panic!(),
    };
    let (maximum, _) = tessellation.conformance(&hexes, mesh.coordinates(), Quantity::new(0.35));
    assert!(maximum < 1e-6, "{maximum}");
}

#[test]
fn every_configuration_of_signs_splits_into_hexahedra_that_hold_up() {
    use super::{CORNERS, Vertex, polyhedron::cell, split};
    use crate::{
        geometry::{Coordinate, mesh::tessellation::D},
        math::{FxHashMap, Scalar},
    };
    let place = |vertex: &Vertex| {
        let at = |corner: [usize; D]| {
            Coordinate::<D>::from(from_fn::<_, D, _>(|d| Quantity::new(corner[d] as Scalar)))
        };
        match vertex {
            Vertex::Inside(corner) => at(*corner),
            Vertex::Boundary([one, two]) => (&at(*one) + &at(*two)) / 2.0,
        }
    };
    for mask in 1u16..256 {
        let inside: [bool; 8] = from_fn(|corner| mask >> corner & 1 == 1);
        let cells = cell(CORNERS, inside).unwrap_or_else(|error| panic!("{mask:08b}  {error}"));
        let points: FxHashMap<Vertex, Coordinate<D>> = cells
            .iter()
            .flat_map(|polyhedron| polyhedron.vertices())
            .map(|vertex| (vertex, place(&vertex)))
            .collect();
        let mesh = split::hexahedra(cells, &points, None)
            .unwrap_or_else(|error| panic!("{mask:08b}  {error}"));
        let scaled = &mesh.minimum_scaled_jacobians()[0];
        let minimum = scaled.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(minimum > 0.0, "{mask:08b}  min SJ {minimum}");
        match &mesh.connectivities()[0] {
            Connectivity::Hexahedral(hexes) => assert!(
                hexes
                    .iter()
                    .all(|hex| bernstein::certifies(hex.as_ref(), mesh.coordinates())),
                "{mask:08b}  uncertified"
            ),
            _ => panic!(),
        }
    }
}

mod field {
    use super::{Placement, report};
    use crate::{
        geometry::{
            grid::{
                Method,
                marching_cubes::separated::test::{extractor, sample, sphere},
            },
            mesh::{Connectivity, Mesh, Verdict},
        },
        math::Tensor,
    };
    use std::{array::from_fn, collections::HashMap, f64::consts::PI};
    fn volume_of(mesh: &Mesh<3>) -> f64 {
        mesh.volumes().into_iter().flatten().sum()
    }
    #[test]
    fn a_sphere_sampled_at_uneven_spacing_meshes_to_its_volume() {
        let spacing = [0.08, 0.12, 0.2];
        let volume = sphere([30, 20, 12], spacing, 0.9);
        let mesh = extractor(spacing, Method::Lewiner)
            .hexahedra(&volume, Placement::Crossing(0.2))
            .unwrap();
        let (count, minimum, negative) = report("uneven", &mesh);
        assert!(count > 0);
        assert_eq!(negative, 0, "min SJ {minimum}");
        let exact = 4.0 / 3.0 * PI * 0.9_f64.powi(3);
        let found = volume_of(&mesh);
        assert!(
            (found - exact).abs() < 0.03 * exact,
            "{found} against {exact}"
        );
    }
    #[test]
    fn the_surface_is_the_boundary_of_the_hexahedra() {
        let spacing = [0.1, 0.1, 0.1];
        let volume = sphere([24, 24, 24], spacing, 1.0);
        let march = extractor(spacing, Method::Separated);
        let surface = march.extract(&volume, None).unwrap();
        let mesh = march.hexahedra(&volume, Placement::Crossing(0.0)).unwrap();
        let hexes: Vec<[usize; 8]> = match &mesh.connectivities()[0] {
            Connectivity::Hexahedral(block) => block.iter().copied().collect(),
            _ => panic!(),
        };
        let boundary: Vec<usize> = mesh.exterior_faces().into_iter().flatten().collect();
        assert!(!hexes.is_empty() && !boundary.is_empty());
        let coordinates = mesh.coordinates();
        let near = |vertex: usize| {
            boundary.iter().any(|&node| {
                (0..3).all(|axis| {
                    (coordinates[node][axis].value() - surface.vertices[vertex][axis].value()).abs()
                        < 1.0e-9
                })
            })
        };
        assert!((0..surface.vertices.len()).all(near));
    }
    #[test]
    fn a_field_of_noise_gives_conforming_hexahedra() {
        let nel = [8, 8, 8];
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut noise = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        };
        let volume = sample(nel, |[i, j, k]| {
            if [i, j, k]
                .iter()
                .any(|&index| index == 0 || index == nel[0] - 1)
            {
                -1.0
            } else {
                noise()
            }
        });
        let mesh = extractor([1.0, 1.3, 0.7], Method::Separated)
            .hexahedra(&volume, Placement::Midpoint)
            .unwrap();
        let (_, minimum, negative) = report("noise", &mesh);
        assert_eq!(negative, 0, "min SJ {minimum}");
        let hexes: Vec<[usize; 8]> = match &mesh.connectivities()[0] {
            Connectivity::Hexahedral(block) => block.iter().copied().collect(),
            _ => panic!(),
        };
        const FACES: [[usize; 4]; 6] = [
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
        ];
        let mut faces = HashMap::<[usize; 4], u32>::new();
        hexes.iter().for_each(|hex| {
            FACES.iter().for_each(|face| {
                let mut key: [usize; 4] = from_fn(|corner| hex[face[corner]]);
                key.sort_unstable();
                *faces.entry(key).or_default() += 1
            })
        });
        assert!(faces.values().all(|&count| count <= 2));
        let mut edges = HashMap::<[usize; 2], u32>::new();
        faces
            .iter()
            .filter(|&(_, &count)| count == 1)
            .for_each(|(face, _)| {
                for edge in [[0, 1], [1, 3], [3, 2], [2, 0]] {
                    let mut key = [face[edge[0]], face[edge[1]]];
                    key.sort_unstable();
                    *edges.entry(key).or_default() += 1
                }
            });
        assert!(
            edges.values().all(|&count| count % 2 == 0),
            "the boundary is open"
        );
    }
    #[test]
    fn a_grid_without_room_for_a_cell_is_refused() {
        let volume = sample([1, 4, 4], |_| 0.0);
        assert!(
            extractor([1.0; 3], Method::Lewiner)
                .hexahedra(&volume, Placement::Midpoint)
                .is_err()
        );
    }
}
