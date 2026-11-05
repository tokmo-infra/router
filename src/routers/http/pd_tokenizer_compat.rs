// Generated from frozen rc3 and upstream tokenizers PR2558.
// Upstream commit: c1b083ba76f92fb51fb32f9eaccde29e4976f878 (unmerged).
// rc3 table SHA256: cc6f5dd0544777c122ba6a768b157a70f9e72e2f16a926de2994bde45c18e6ee
// corrected table SHA256: 500fda798f686250e20ed51f6cfb857b8e1a23d344f1583ae624a349ccdd90d2
// Comparing all 1,112,064 Unicode scalars gives 4,805 changes in 49 ranges.
// Conservative classifier delta, not an arbitrary Unicode whitelist.
pub(super) const INCOMPATIBLE_RANGES: &[(u32, u32)] = &[
    (0x295, 0x295),
    (0x88F, 0x88F),
    (0xC5C, 0xC5C),
    (0xCDC, 0xCDC),
    (0x1ACF, 0x1ADD),
    (0x1AE0, 0x1AEB),
    (0x20C1, 0x20C1),
    (0x2B96, 0x2B96),
    (0xA7CE, 0xA7CF),
    (0xA7D2, 0xA7D2),
    (0xA7D4, 0xA7D4),
    (0xA7F1, 0xA7F1),
    (0xFBC3, 0xFBD2),
    (0xFD90, 0xFD91),
    (0xFDC8, 0xFDCE),
    (0x10940, 0x10959),
    (0x10EC5, 0x10EC7),
    (0x10ED0, 0x10ED8),
    (0x10EFA, 0x10EFB),
    (0x11B60, 0x11B67),
    (0x11DB0, 0x11DDB),
    (0x11DE0, 0x11DE9),
    (0x16EA0, 0x16EB8),
    (0x16EBB, 0x16ED3),
    (0x16FE3, 0x16FE3),
    (0x16FF2, 0x16FF6),
    (0x187F8, 0x187FF),
    (0x18D09, 0x18D1E),
    (0x18D80, 0x18DF2),
    (0x1CCFA, 0x1CCFC),
    (0x1CEBA, 0x1CED0),
    (0x1CEE0, 0x1CEF0),
    (0x1E6C0, 0x1E6DE),
    (0x1E6E0, 0x1E6F5),
    (0x1E6FE, 0x1E6FF),
    (0x1F6D8, 0x1F6D8),
    (0x1F777, 0x1F77A),
    (0x1F8D0, 0x1F8D8),
    (0x1FA54, 0x1FA57),
    (0x1FA8A, 0x1FA8A),
    (0x1FA8E, 0x1FA8E),
    (0x1FAC8, 0x1FAC8),
    (0x1FACD, 0x1FACD),
    (0x1FAEA, 0x1FAEA),
    (0x1FAEF, 0x1FAEF),
    (0x1FBFA, 0x1FBFA),
    (0x2B73A, 0x2B73F),
    (0x2CEA2, 0x2CEAD),
    (0x323B0, 0x33479),
];
pub(super) fn compatible(text: &str) -> bool {
    text.is_ascii() || !text.chars().any(|c| {
        if c.is_ascii() { return false; }
        let cp = c as u32;
        let i = INCOMPATIBLE_RANGES.partition_point(|&(start, _)| start <= cp);
        i > 0 && cp <= INCOMPATIBLE_RANGES[i - 1].1
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_delta_is_sorted_and_matches_all_boundaries() {
        let mut total = 0;
        for (i, &(a, b)) in INCOMPATIBLE_RANGES.iter().enumerate() {
            assert!(a <= b && a > 127);
            if i > 0 { assert!(INCOMPATIBLE_RANGES[i - 1].1 + 1 < a); }
            for cp in a..=b {
                let c = char::from_u32(cp).unwrap();
                assert!(!compatible(&format!("abc{c}xyz")), "U+{cp:X}");
            }
            for cp in [a - 1, b + 1] {
                if let Some(c) = char::from_u32(cp) {
                    assert!(compatible(&c.to_string()), "U+{cp:X}");
                }
            }
            total += b - a + 1;
        }
        assert_eq!(total, 4805);
        assert_eq!(INCOMPATIBLE_RANGES.len(), 49);
    }
    #[test]
    fn normal_unicode_remains_fast_and_unicode17_falls_back() {
        for text in ["", "English code", "中文", "日本語", "é e\u{301}", "🙂", "<｜begin▁of▁sentence｜>"] {
            assert!(compatible(text), "{text:?}");
        }
        assert!(!compatible(" \u{1faea}"));
    }
}
