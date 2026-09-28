#![allow(dead_code)]

#[cfg(test)]
mod test;

pub(crate) mod block;
pub(crate) mod dual;
pub(crate) mod dual_primal;
pub(crate) mod interface;
pub(crate) mod parallel;
pub(crate) mod pcg;
pub(crate) mod subdomain;

pub(crate) const THREADS: usize = 1;
