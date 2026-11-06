pub(crate) mod solid;
#[cfg(test)]
pub(crate) mod test;

use crate::{
    domain::block::add_node_neighbors,
    math::{Style, StyledError, assert::AssertionError, defeat_message},
};
use core::{
    fmt::{self, Debug, Display, Formatter},
    marker::PhantomData,
};

pub trait Elements
where
    Self: Debug,
{
    fn node_neighbors(&self, neighbors: &mut [Vec<usize>]);
}

/// The structure of a block: the items it is made of, and the nodes of each.
///
/// An item is an element, or whatever else contributes to a set of nodes. Every [`Items`] is
/// [`Elements`], and the traits that assemble over the items extend this one with their kernels.
pub trait Items
where
    Self: Debug,
{
    /// The number of items.
    fn number_of_items(&self) -> usize;
    /// The nodes of an item.
    fn nodes_of(&self, item: usize) -> &[usize];
}

impl<T> Elements for T
where
    T: Items,
{
    fn node_neighbors(&self, neighbors: &mut [Vec<usize>]) {
        add_node_neighbors(
            (0..self.number_of_items()).map(|item| self.nodes_of(item)),
            neighbors,
        )
    }
}

pub trait ElementKind {
    const NAME: &'static str;
}

pub enum ElementError<K> {
    Upstream(String, String, PhantomData<K>),
}

impl<K> ElementError<K> {
    pub fn upstream(error: impl Display, context: &(impl Debug + ?Sized)) -> Self {
        Self::Upstream(format!("{error}"), format!("{context:?}"), PhantomData)
    }
}

impl<K: ElementKind> From<ElementError<K>> for AssertionError {
    fn from(error: ElementError<K>) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}

impl<K: ElementKind> StyledError for ElementError<K> {
    fn message(&self, style: &Style) -> String {
        let c = style.frame;
        match self {
            Self::Upstream(error, element, _) => format!(
                "{error}{c}\n\
                In {}: {element}.",
                K::NAME
            ),
        }
    }
}

impl<K: ElementKind> Debug for ElementError<K> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let style = Style::detect();
        write!(
            f,
            "\n{}\n{}{}{}\n",
            self.message(&style),
            style.footer,
            defeat_message(),
            style.reset
        )
    }
}

impl<K: ElementKind> Display for ElementError<K> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let style = Style::detect();
        write!(f, "{}{}", self.message(&style), style.reset)
    }
}

impl<K: ElementKind> core::error::Error for ElementError<K> {}
