// Not yet used by cbm alone (only via fem/vem); not dead in the architectural
// sense, so suppress rather than gate out
#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
pub(crate) mod viscoplastic;
