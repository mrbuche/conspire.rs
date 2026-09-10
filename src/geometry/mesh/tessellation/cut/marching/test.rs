use super::{Finish, Marching, Placement};
use crate::{
    geometry::mesh::{
        Connectivity, Verdict,
        quality::metrics::hexahedron::bernstein,
        tessellation::cut::test::{sphere, star},
    },
    math::Quantity,
};

fn report(name: &str, mesh: &crate::geometry::mesh::Mesh<3>) -> (usize, f64, usize) {
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
fn every_configuration_of_signs_splits_into_hexahedra_that_hold_up() {
    use super::{CORNERS, Vertex, polyhedron::cell, split};
    use crate::{
        geometry::{Coordinate, mesh::tessellation::D},
        math::{FxHashMap, Scalar},
    };
    let place = |vertex: &Vertex| {
        let at = |corner: [usize; D]| {
            Coordinate::<D>::from(std::array::from_fn::<_, D, _>(|d| {
                Quantity::new(corner[d] as Scalar)
            }))
        };
        match vertex {
            Vertex::Inside(corner) => at(*corner),
            Vertex::Boundary([one, two]) => (&at(*one) + &at(*two)) / 2.0,
        }
    };
    for mask in 1u16..256 {
        let inside: [bool; 8] = std::array::from_fn(|corner| mask >> corner & 1 == 1);
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
