use crate::{
    domain::boundary::{Enforcement, Field, Rows},
    math::optimize::Prescription,
    units::Temperature,
};

pub(crate) fn fixed_temperatures(nodes: &[usize]) -> Rows<Field<Temperature>> {
    nodes
        .iter()
        .fold(Rows::new(Enforcement::Elimination), |rows, &node| {
            rows.hold(node, Prescription::Initial)
        })
}
