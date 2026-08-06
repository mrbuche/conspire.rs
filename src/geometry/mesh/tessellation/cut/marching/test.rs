use super::Placement;
use crate::geometry::mesh::{
    Connectivity, Verdict,
    quality::metrics::hexahedron::bernstein,
    tessellation::cut::test::{box_surface, rotated, sphere, star},
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
            match tessellation.marching_hex(spacing, placement) {
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
    let mesh = sphere(3).marching_hex(0.2, Placement::Midpoint).unwrap();
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
#[ignore = "diagnostic; run with --release -- --ignored --nocapture --test-threads=1"]
fn bone_marching() {
    use crate::{
        geometry::mesh::{Output, Tessellation, Vtk},
        io::{Write, write::Compression},
    };
    use std::{path::Path, time::Instant};
    let tessellation = Tessellation::try_from(Path::new("bone_tri.stl")).unwrap();
    for divisions in [16.0, 32.0, 64.0, 128.0] {
        let spacing = 0.9 / divisions;
        for (label, placement) in [
            ("midpoint", Placement::Midpoint),
            ("crossing", Placement::Crossing(0.2)),
        ] {
            let start = Instant::now();
            match tessellation.marching_hex(spacing, placement) {
                Err(error) => println!("{divisions:>5.0}/{label}  {error}"),
                Ok(mesh) => {
                    let seconds = start.elapsed().as_secs_f64();
                    let (count, minimum, negative) =
                        report(&format!("{divisions:.0}/{label}"), &mesh);
                    let path = format!("bone_{label}_{divisions:.0}.vtm");
                    mesh.write(Output::Vtk(Vtk::MultiBlock(Compression::Off(&path))))
                        .unwrap();
                    println!(
                        "                  {seconds:>6.2} s   {count} hexes, min SJ {minimum:.4}, {negative} inverted, wrote {path}"
                    );
                }
            }
        }
    }
}
