use crate::{
    domain::boundary::{Enforcement, Rows, Solid},
    math::optimize::Prescription,
};

pub(crate) fn fixed_dofs(dofs: &[usize]) -> Rows<Solid<3>> {
    dofs.iter()
        .fold(Rows::new(Enforcement::Elimination), |rows, &dof| {
            rows.hold(dof, Prescription::Initial)
        })
}
