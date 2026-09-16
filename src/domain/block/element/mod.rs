pub(crate) mod solid;

use crate::math::{Style, StyledError, assert::AssertionError};
use std::{
    fmt::{self, Debug, Display, Formatter},
    marker::PhantomData,
};

pub trait Elements
where
    Self: Debug,
{
    fn node_neighbors(&self, neighbors: &mut [Vec<usize>]);
}

/// Names the kind of element an [`ElementError`] was raised for, e.g. "finite
/// element" or "virtual element".
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
            crate::math::defeat_message(),
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

impl<K: ElementKind> std::error::Error for ElementError<K> {}
