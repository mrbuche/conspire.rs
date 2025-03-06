#[cfg(test)]
pub mod test;

use super::*;

pub type Tetrahedron<C> = FooFiniteElement<C, 1, 3, 4, 4>;

// need to apply tests to tet4, also do linear tests
// should ensure same number of tests are run before and after this merges in
