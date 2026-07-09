use super::{Style, defeat_message_inner};

#[test]
fn defeat_message_inner_covers_all_branches() {
    for n in 0..=14 {
        defeat_message_inner(n);
    }
}

#[test]
fn detect_inner_covers_all_branches() {
    for no_color_unset in [false, true] {
        for is_terminal in [false, true] {
            Style::detect_inner(no_color_unset, is_terminal);
        }
    }
}
