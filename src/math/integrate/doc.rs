pub const EXPLICIT: &str = include_str!("explicit.md");
pub const EXPLICIT_IV: &str = include_str!("explicit_iv.md");
pub const IMPLICIT: &str = include_str!("implicit.md");

pub fn bogacki_shampine<'a>() -> Vec<[&'a str; 2]> {
    vec![[
        "math/integrate/explicit/bogacki_shampine",
        include_str!("bogacki_shampine/doc.md"),
    ]]
}

pub fn dormand_prince<'a>() -> Vec<[&'a str; 2]> {
    vec![[
        "math/integrate/explicit/dormand_prince",
        include_str!("dormand_prince/doc.md"),
    ]]
}

pub fn verner_8<'a>() -> Vec<[&'a str; 2]> {
    vec![[
        "math/integrate/explicit/verner_8",
        include_str!("verner_8/doc.md"),
    ]]
}

pub fn verner_9<'a>() -> Vec<[&'a str; 2]> {
    vec![[
        "math/integrate/explicit/verner_9",
        include_str!("verner_9/doc.md"),
    ]]
}

pub fn backward_euler<'a>() -> Vec<[&'a str; 2]> {
    vec![[
        "math/integrate/implicit/backward_euler",
        include_str!("backward_euler/doc.md"),
    ]]
}
