import regex, unicodedata

PROBLEMATIC_AKSHARAS = {
    'யு', 'றா', 'ஞா', 'யே', 'யை', 'றே', 'கெ', 'றோ', 'யெ', 'யொ',
    'றெ', 'ழா', 'நூ', 'லெ', 'வொ', 'ரெ', 'றொ', 'கீ', 'டே', 'லொ',
    'னொ', 'டொ', 'ணே', 'ளெ', 'ணோ', 'ரொ', 'றூ', 'ணீ', 'றீ', 'ழீ',
    'னூ', 'ணெ', 'வூ', 'யீ', 'டீ', 'ளொ', 'லூ', 'நொ', 'ணொ', 'டூ',
    'நை', 'கௌ', 'ளூ', 'ழே', 'ழூ', 'ஞை', 'ளீ', 'ஜை', 'ணூ', 'ழொ',
    'ழெ', 'ஞு', 'ஞெ', 'ஞூ', 'ஸா', 'ஜூ', 'எா', 'எீ', 'ழோ', 'பௌ',
}


def clean_grapheme_split(text, tokenizer):
    """
    Split Tamil text into grapheme clusters.
    Uses | as intra-word separator, space as inter-word separator.
    Uses pre-computed lookup table for problematic aksharas instead
    of recursive merging — more reliable and handles all edge cases.
    """
    words = text.split(' ')
    result_words = []

    for word in words:
        if not word.strip():
            continue

        clusters = regex.findall(r'\X', word)
        clusters = [
            unicodedata.normalize('NFC', c) for c in clusters
            if c.strip() and any('\u0B80' <= ch <= '\u0BFF' for ch in c)
        ]

        if not clusters:
            result_words.append(word)
            continue

        merged = []
        i = 0
        while i < len(clusters):
            c = clusters[i]
            tokens = tokenizer.tokenize(c)
            if (c in PROBLEMATIC_AKSHARAS or len(tokens) > 1) and i + 1 < len(clusters):
                combined = c + clusters[i+1]
                while len(tokenizer.tokenize(combined)) > 1 and i + 2 < len(clusters):
                    i += 1
                    combined = combined + clusters[i+1]
                merged.append(combined)
                i += 2
            else:
                merged.append(c)
                i += 1

        result_words.append('|'.join(merged))

    return ' '.join(result_words)


def post_process_grapheme(text):
    """Remove | separators after generation to restore normal Tamil text."""
    return text.replace('|', '')
