use crate::math::Scalar;

/// A skyline matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct SkylineMatrix {
    values: Vec<Scalar>,
    col_heights: Vec<usize>,
    col_offsets: Vec<usize>,
}

//
// Would need block.nodal_stiffnesses() to directly fill and return the skyline matrix in order to retain memory efficiency.
// Could have it return an enum if want to keep both options (such as for all your tests).
// And then the solver parts will just call different impls for the two different options.
// The part in fem/block that builds the skyline matrix for the stiffness will also have to return information about the tangent skyline matrix.
// So that it can be built correctly in the solver. Could handle it in fem/block but that would assume something about how it is being solved.
// For example the separate range/null space method will use it differently.
//
