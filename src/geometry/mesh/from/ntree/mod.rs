#[cfg(test)]
mod test;

pub(crate) mod facets;
mod dualization;
mod polyhedra;
mod tetrahedra;

pub(crate) use dualization::Dualization;
