use super::{
    super::{
        Class,
        test::{box_surface, sphere},
    },
    Lattice,
};
use crate::math::Scalar;

#[test]
fn box_encloses_its_interior_cells() {
    let lattice = box_surface([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        .lattice(0.1)
        .unwrap();
    let cells = lattice.cells();
    assert!(cells.iter().any(|&(_, class)| class == Class::Cut));
    assert!(cells.iter().any(|&(_, class)| class == Class::Inside));
    // The interior of a unit box at this spacing is a solid run of cells.
    let inside = cells
        .iter()
        .filter(|&&(_, class)| class == Class::Inside)
        .count();
    assert!(inside >= 6 * 6 * 6, "only {inside} enclosed cells");
}

#[test]
fn the_exterior_is_only_ever_one_cell_deep() {
    // A fill that ran outward instead of inward would occupy the exterior in
    // bulk rather than in the single layer the boundary faces need.
    let lattice = sphere(2).lattice(0.15).unwrap();
    lattice
        .cells()
        .iter()
        .filter(|&&(_, class)| class == Class::Outside)
        .for_each(|&(index, _)| {
            assert!(
                lattice.neighbors(index).any(|next| lattice
                    .cells
                    .get(&next)
                    .is_some_and(|&class| class != Class::Outside)),
                "{index:?} is outside but touches nothing"
            )
        });
}

/// The enclosed cells underfill the surface and the occupied cells overfill
/// it, so the two bracket the true volume and the bracket closes with the
/// cut cells.
fn bracket(lattice: &Lattice, spacing: Scalar) -> (Scalar, Scalar) {
    let cells = lattice.cells();
    let inside = cells
        .iter()
        .filter(|&&(_, class)| class == Class::Inside)
        .count();
    let occupied = cells
        .iter()
        .filter(|&&(_, class)| class != Class::Outside)
        .count();
    (
        inside as Scalar * spacing.powi(3),
        occupied as Scalar * spacing.powi(3),
    )
}

#[test]
fn box_volume_is_bracketed_ever_more_tightly() {
    let mut previous = Scalar::INFINITY;
    for spacing in [0.2, 0.1, 0.05] {
        let lattice = box_surface([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
            .lattice(spacing)
            .unwrap();
        let (low, high) = bracket(&lattice, spacing);
        assert!(low <= 1.0 && 1.0 <= high, "[{low}, {high}] at {spacing}");
        assert!(high - low < previous);
        previous = high - low;
    }
}

#[test]
fn cells_are_ascending_and_unique() {
    let lattice = sphere(2).lattice(0.15).unwrap();
    let cells = lattice.cells();
    let keys: Vec<_> = cells.iter().map(|&([i, j, k], _)| (k, j, i)).collect();
    assert!(keys.windows(2).all(|pair| pair[0] < pair[1]));
}

#[test]
fn sphere_fill_does_not_leak() {
    // A leak past the shell reaches the padding, which is outside by
    // construction.
    let lattice = sphere(2).lattice(0.15).unwrap();
    let [nx, ny, nz] = lattice.nel;
    lattice.cells().iter().for_each(|&([i, j, k], _)| {
        assert!(i > 0 && j > 0 && k > 0);
        assert!(i < nx - 1 && j < ny - 1 && k < nz - 1);
    });
}

#[test]
fn meshes_one_hexahedron_per_cell() {
    let lattice = box_surface([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        .lattice(0.2)
        .unwrap();
    let mesh = lattice.mesh();
    assert_eq!(mesh.number_of_elements(), lattice.cells().len());
}

#[test]
fn rejects_nonpositive_spacing() {
    assert!(box_surface([0.0; 3], [1.0; 3]).lattice(0.0).is_err());
    assert!(box_surface([0.0; 3], [1.0; 3]).lattice(-1.0).is_err());
}
#[test]
fn enclosed_cells_of_a_box_are_exactly_its_interior() {
    // Five cells span the box, so one shell cell each side leaves three.
    let lattice = box_surface([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        .lattice(0.2)
        .unwrap();
    assert_eq!(
        lattice
            .cells()
            .iter()
            .filter(|&&(_, class)| class == Class::Inside)
            .count(),
        3 * 3 * 3
    );
}

#[test]
fn a_plate_thinner_than_a_cell_is_all_cut_and_does_not_leak() {
    // Opposite sides of the shell land in the same cells, which is where a
    // fill leaks if the rasterized shell is not watertight.
    let lattice = box_surface([0.0, 0.0, 0.0], [4.0, 4.0, 0.3])
        .lattice(0.5)
        .unwrap();
    let cells = lattice.cells();
    let [nx, ny, nz] = lattice.nel;
    assert!(cells.iter().all(|&(_, class)| class != Class::Inside));
    assert!(cells.len() < nx * ny * nz);
}

#[test]
fn sphere_volume_is_bracketed_ever_more_tightly() {
    // The tessellation inscribes the sphere, so it is the lower bound that
    // must hold exactly; the exact volume only bounds it from above.
    let exact = 4.0 * std::f64::consts::PI / 3.0;
    let mut previous = Scalar::INFINITY;
    for spacing in [0.3, 0.15, 0.075] {
        let lattice = sphere(3).lattice(spacing).unwrap();
        let (low, high) = bracket(&lattice, spacing);
        assert!(
            low <= exact && exact <= high,
            "[{low}, {high}] at {spacing}"
        );
        assert!(high - low < previous);
        previous = high - low;
    }
}
