pub const EXPLICIT: &str = include_str!("explicit.md");

pub fn bogacki_shampine<'a>() -> Vec<[&'a str; 2]> {
    vec![[
        "math/integrate/bogacki_shampine",
        include_str!("bogacki_shampine/doc.md"),
    ]]
}

pub fn dormand_prince<'a>() -> Vec<[&'a str; 2]> {
    vec![[
        "math/integrate/dormand_prince",
        include_str!("dormand_prince/doc.md"),
    ]]
}

pub fn verner_8<'a>() -> Vec<[&'a str; 2]> {
    vec![["math/integrate/verner_8", include_str!("verner_8/doc.md")]]
}

pub fn verner_9<'a>() -> Vec<[&'a str; 2]> {
    vec![["math/integrate/verner_9", include_str!("verner_9/doc.md")]]
}
