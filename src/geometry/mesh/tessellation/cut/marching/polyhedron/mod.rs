use super::{CORNERS, Corner, Signs};

pub(super) use crate::geometry::grid::marching_cubes::separated::{Polyhedron, cell};

pub(super) fn polyhedra(
    cells: impl IntoIterator<Item = Corner>,
    signs: &Signs,
) -> Result<Vec<Polyhedron>, &'static str> {
    cells
        .into_iter()
        .filter_map(|[i, j, k]| {
            let corners = CORNERS.map(|[a, b, c]| [i + a, j + b, k + c]);
            let inside = corners.map(|corner| signs.at(corner));
            if inside.iter().all(|&flag| !flag) {
                return None;
            }
            Some(cell(corners, inside))
        })
        .collect::<Result<Vec<Vec<Polyhedron>>, &'static str>>()
        .map(|cells| cells.into_iter().flatten().collect())
}
