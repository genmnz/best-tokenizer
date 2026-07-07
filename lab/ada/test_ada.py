"""
Ada tests. Runs under pytest OR as `python3 test_ada.py` (no pytest needed).
Covers: losslessness, determinism, superword crossing spaces, entropy/length ablation wiring,
save/load roundtrip, and the core compression claim (Ada beats plain BPE at equal vocab).
"""
from __future__ import annotations

import os
import tempfile

from ada_tokenizer import AdaTokenizer, AdaConfig, pretokenize, BASE_VOCAB
from corpus import generate_corpus, split_corpus


def _small_train():
    lines = generate_corpus(n_lines=1500, seed=1)
    return split_corpus(lines)


def test_roundtrip_lossless():
    train, test = _small_train()
    tok = AdaTokenizer(AdaConfig(vocab_size=600)).train(train)
    for t in test[:200]:
        assert tok.decode(tok.encode(t)) == t, "encode/decode must be lossless"


def test_unicode_robustness():
    tok = AdaTokenizer(AdaConfig(vocab_size=400)).train(generate_corpus(800, seed=2)[0:600])
    for s in ["héllo wörld", "日本語のテスト", "emoji 🚀🔥", "mixed 123 abc\tTAB\nNL"]:
        assert tok.decode(tok.encode(s)) == s, f"unicode roundtrip failed for {s!r}"


def test_determinism():
    train, _ = _small_train()
    a = AdaTokenizer(AdaConfig(vocab_size=500, seed=7)).train(train)
    b = AdaTokenizer(AdaConfig(vocab_size=500, seed=7)).train(train)
    assert a.merges == b.merges, "training must be deterministic for a fixed config+corpus"


def _has_internal_space(tok):
    # a leading-space token (" the") is normal BPE; an *internal* space ("of the") is a superword
    return any(b" " in tok.id_to_bytes[i][1:] for i in range(BASE_VOCAB, tok.vocab_size))


def test_superword_crosses_whitespace():
    # a corpus where "of the" is very frequent should yield a token with an internal space
    corpus = ["of the model of the byte of the token of the patch"] * 200
    tok = AdaTokenizer(AdaConfig(vocab_size=320, enable_superword=True,
                                 superword_start_frac=0.3)).train(corpus)
    assert tok.transition_merge is not None
    assert _has_internal_space(tok), "superword stage must produce a token with an internal space"


def test_baseline_keeps_whitespace():
    corpus = ["of the model of the byte of the token"] * 200
    tok = AdaTokenizer(AdaConfig(vocab_size=320, enable_superword=False)).train(corpus)
    assert tok.transition_merge is None
    assert not _has_internal_space(tok), "baseline BPE must never merge across an internal space"


def test_save_load_roundtrip():
    train, test = _small_train()
    tok = AdaTokenizer(AdaConfig(vocab_size=500)).train(train)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ada.json")
        tok.save(p)
        tok2 = AdaTokenizer.load(p)
    for t in test[:100]:
        assert tok.encode(t) == tok2.encode(t), "loaded tokenizer must encode identically"


def test_ada_beats_bpe_compression():
    """The headline claim: at equal vocab, Ada emits fewer tokens than plain BPE."""
    train, test = _small_train()
    cfg = AdaConfig(vocab_size=1024)
    bpe = AdaTokenizer(cfg.__class__(vocab_size=1024, enable_superword=False,
                                     enable_entropy_guard=False,
                                     enable_length_bonus=False)).train(train)
    ada = AdaTokenizer(cfg).train(train)
    bpe_toks = sum(len(bpe.encode(t)) for t in test)
    ada_toks = sum(len(ada.encode(t)) for t in test)
    assert ada.vocab_size == bpe.vocab_size
    assert ada_toks < bpe_toks, f"Ada={ada_toks} should be < BPE={bpe_toks} at equal vocab"


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    return passed == len(fns)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run_all() else 1)
