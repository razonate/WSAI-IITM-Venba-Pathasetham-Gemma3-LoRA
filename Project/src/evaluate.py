import sacrebleu


def compute_exact_match(predictions, references):
    exact_matches = sum(p.strip() == r.strip() for p, r in zip(predictions, references))
    n = len(references)
    return exact_matches, n, 100 * exact_matches / n


def compute_word_accuracy(predictions, references):
    word_correct, word_total = 0, 0
    for pred, ref in zip(predictions, references):
        pw, rw = pred.split(), ref.split()
        word_correct += sum(p == r for p, r in zip(pw, rw))
        word_total += max(len(pw), len(rw))
    return word_correct, word_total, 100 * word_correct / word_total


def compute_bleu(predictions, references):
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return bleu.score
