use super::{Marching, Placement};
use crate::{
    geometry::mesh::{
        Connectivity, Verdict,
        quality::metrics::hexahedron::bernstein,
        tessellation::cut::test::{box_surface, rotated, sphere, star},
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
#[ignore = "diagnostic; run with --release -- --ignored --nocapture --test-threads=1"]
fn placements_compared() {
    for (name, tessellation, spacing) in [
        ("sphere", sphere(3), 0.2),
        ("sphere fine", sphere(3), 0.1),
        ("box", box_surface([-0.5; 3], [0.5; 3]), 0.2),
        (
            "box tilted",
            rotated(&box_surface([-0.5; 3], [0.5; 3]), [0.3, 0.4, 0.5]),
            0.1,
        ),
        ("star", star(1, 2.0), 0.2),
        ("star fine", star(2, 1.6), 0.08),
        ("spike", star(1, 4.0), 0.2),
        (
            "slab",
            box_surface([-2.0, -2.0, -0.05], [2.0, 2.0, 0.05]),
            0.25,
        ),
    ] {
        for (label, placement) in [
            ("midpoint", Placement::Midpoint),
            ("crossing", Placement::Crossing(0.2)),
        ] {
            match tessellation.marching_hex(
                Quantity::new(spacing),
                Marching {
                    placement,
                    keep: None,
                },
            ) {
                Ok(mesh) => {
                    report(&format!("{name}/{label}"), &mesh);
                }
                Err(error) => println!("{name:>12}/{label}  {error}"),
            }
        }
    }
}

#[test]
fn a_sphere_is_all_hexahedra_and_none_inverted() {
    let mesh = sphere(3)
        .marching_hex(
            Quantity::new(0.2),
            Marching {
                placement: Placement::Midpoint,
                keep: None,
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
                keep: None,
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
#[ignore = "diagnostic; run with --release -- --ignored --nocapture --test-threads=1"]
fn bone_marching() {
    use crate::{
        geometry::mesh::{Connectivity, Output, Tessellation, Vtk},
        io::{Write, write::Compression},
    };
    use std::{path::Path, time::Instant};
    let tessellation = Tessellation::try_from(Path::new("bone_tri.stl")).unwrap();
    println!(
        "{:>28}  {:>8}  {:>7}  {:>8}  {:>5}  {:>8}  {:>8}",
        "case", "hexes", "s", "min SJ", "bad", "max gap", "avg gap"
    );
    for divisions in [16.0, 32.0, 64.0] {
        let spacing = Quantity::new(0.9 / divisions);
        for (label, placement) in [
            ("midpoint", Placement::Midpoint),
            ("crossing", Placement::Crossing(0.2)),
        ] {
            for draw in [false, true] {
                let start = Instant::now();
                let mesh = match tessellation.marching_hex(
                    spacing,
                    Marching {
                        placement,
                        keep: draw.then_some(0.5),
                    },
                ) {
                    Err(error) => {
                        println!("{divisions:>5.0}/{label}  {error}");
                        continue;
                    }
                    Ok(mesh) => mesh,
                };
                let seconds = start.elapsed().as_secs_f64();
                let scaled = &mesh.minimum_scaled_jacobians()[0];
                let minimum = scaled.iter().cloned().fold(f64::INFINITY, f64::min);
                let bad = scaled.iter().filter(|&&value| value <= 0.0).count();
                let hexes = match &mesh.connectivities()[0] {
                    Connectivity::Hexahedral(hexes) => {
                        hexes.iter().copied().collect::<Vec<[usize; 8]>>()
                    }
                    _ => panic!(),
                };
                let (worst, mean) = tessellation.conformance(&hexes, mesh.coordinates(), spacing);
                let drawn = if draw { "drawn" } else { "ascut" };
                println!(
                    "{:>18}/{drawn:>7}  {:>8}  {seconds:>7.2}  {minimum:>8.4}  {bad:>5}  {worst:>8.4}  {mean:>8.4}",
                    format!("{divisions:.0}/{label}"),
                    scaled.len()
                );
                let path = format!("bone_{label}_{divisions:.0}_{drawn}.vtm");
                mesh.write(Output::Vtk(Vtk::MultiBlock(Compression::Off(&path))))
                    .unwrap();
            }
        }
    }
}

#[test]
#[ignore = "diagnostic; run with --release -- --ignored --nocapture --test-threads=1"]
fn guard_swept() {
    use crate::geometry::mesh::{Connectivity, Tessellation};
    use std::path::Path;
    let tessellation = Tessellation::try_from(Path::new("bone_tri.stl")).unwrap();
    let spacing = Quantity::new(0.9 / 64.0);
    println!(
        "{:>10}  {:>8}  {:>8}  {:>8}  {:>8}  {:>5}",
        "guard", "SJ ascut", "gap ascut", "SJ drawn", "gap drawn", "bad"
    );
    for guard in [0.0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.49] {
        let mut row = Vec::new();
        for draw in [false, true] {
            let mesh = tessellation
                .marching_hex(
                    spacing,
                    Marching {
                        placement: Placement::Crossing(guard),
                        keep: draw.then_some(0.5),
                    },
                )
                .unwrap();
            let scaled = &mesh.minimum_scaled_jacobians()[0];
            let minimum = scaled.iter().cloned().fold(f64::INFINITY, f64::min);
            let bad = scaled.iter().filter(|&&value| value <= 0.0).count();
            let hexes = match &mesh.connectivities()[0] {
                Connectivity::Hexahedral(hexes) => {
                    hexes.iter().copied().collect::<Vec<[usize; 8]>>()
                }
                _ => panic!(),
            };
            let (_, mean) = tessellation.conformance(&hexes, mesh.coordinates(), spacing);
            row.push((minimum, mean, bad))
        }
        println!(
            "{guard:>10.2}  {:>8.4}  {:>9.4}  {:>8.4}  {:>9.4}  {:>5}",
            row[0].0,
            row[0].1,
            row[1].0,
            row[1].1,
            row[0].2 + row[1].2
        );
    }
}

#[test]
#[ignore = "diagnostic; run with --release -- --ignored --nocapture --test-threads=1"]
fn guard_against_keep() {
    use crate::geometry::mesh::{Connectivity, Tessellation};
    use std::path::Path;
    let tessellation = Tessellation::try_from(Path::new("bone_tri.stl")).unwrap();
    let spacing = Quantity::new(0.9 / 64.0);
    println!(
        "{:>8}  {:>6}  {:>9}  {:>9}",
        "guard", "keep", "min SJ", "avg gap"
    );
    for guard in [0.0, 0.1, 0.2, 0.3, 0.49] {
        for keep in [0.9, 0.7, 0.5, 0.3, 0.15, 0.05] {
            let mesh = tessellation
                .marching_hex(
                    spacing,
                    Marching {
                        placement: Placement::Crossing(guard),
                        keep: Some(keep),
                    },
                )
                .unwrap();
            let scaled = &mesh.minimum_scaled_jacobians()[0];
            let minimum = scaled.iter().cloned().fold(f64::INFINITY, f64::min);
            let hexes = match &mesh.connectivities()[0] {
                Connectivity::Hexahedral(hexes) => {
                    hexes.iter().copied().collect::<Vec<[usize; 8]>>()
                }
                _ => panic!(),
            };
            let (_, mean) = tessellation.conformance(&hexes, mesh.coordinates(), spacing);
            println!("{guard:>8.2}  {keep:>6.2}  {minimum:>9.5}  {mean:>9.5}");
        }
    }
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
