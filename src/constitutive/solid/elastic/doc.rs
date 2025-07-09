pub const DOC: &str = include_str!("doc.md");
pub const ALMANSI_HAMEL: ([&str; 2], [&str; 2], [&str; 2]) = (
    [
        "src/constitutive/solid/elastic/almansi_hamel",
        include_str!("almansi_hamel/doc.md"),
    ],
    [
        "cauchy_stress",
        include_str!("almansi_hamel/cauchy_stress.md"),
    ],
    [
        "cauchy_tangent_stiffness",
        include_str!("almansi_hamel/cauchy_tangent_stiffness.md"),
    ],
);
