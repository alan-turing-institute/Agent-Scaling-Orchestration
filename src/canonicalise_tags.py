"""Collapse near-duplicate question tags into canonical categories.

`tag_questions.py` asks the model for free-text tags per question, so the raw
vocabulary carries many surface variants of one capability ("step-by-step
reasoning", "step by step reasoning", "sequential reasoning"). The hand-written
TAG_MAPPING in `tag_dataset.py` only covers variants someone happened to notice.

This script builds the mapping from the data instead, in two stages:

1. Lexical normalisation merges variants that differ only in case, punctuation,
   separators, simple plurals or filler words. No model needed, no judgement.
2. An LLM groups what is left by meaning, in batches, then re-clusters the
   canonical labels those batches produced so synonyms split across batches
   still meet. Repeats until a round merges nothing.

Writes a JSON mapping {raw tag: canonical tag} that `tag_dataset.py --tag_mapping`
applies before the frequency filter.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, OrderedDict

from openai import OpenAI


SYSTEM_PROMPT = (
    "You are curating the tag vocabulary for a dataset of reasoning questions. "
    "Tags name a capability, reasoning style or domain that would help choose "
    "which agents should answer a question."
)

CLUSTER_PROMPT = """Below is a list of tags with the number of questions carrying each one.

Group tags that name the same capability into one category. Two tags belong together only when an agent good at one is good at the other by definition - a wording variant, a synonym, or the same skill at a different level of detail.

Rules:
- Keep genuinely different capabilities apart, even when they are related. "arithmetic" and "algebra" are different. "commonsense reasoning" and "physical reasoning" are different.
- Prefer the highest-count tag in a group as the canonical name, unless a lower-count tag names the capability more plainly.
- Use lowercase and the wording already present. Do not invent new vocabulary.
- Every tag in the input must appear in exactly one group, including tags that group alone.

Tags:
{tag_block}

Return only valid JSON, no markdown fences, no commentary, in this shape:
{{"groups": [{{"canonical": "...", "members": ["...", "..."]}}]}}
"""

# Words that carry no distinguishing meaning in a capability tag.
FILLER_WORDS = {"general", "basic", "simple", "task", "tasks", "based", "skills", "ability"}

# Suffixes that mark the same capability written as a different part of speech.
LEMMA_SUFFIXES = (("ing", ""), ("ies", "y"), ("es", ""), ("s", ""))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a raw-tag -> canonical-tag mapping from a tags JSONL"
    )
    parser.add_argument(
        "--tags_file",
        default="data-claude/question_tags/gsm8k_arc_hellaswag_truthfulqa_winogrande_pro_medicine_formal_logic_test_100_tags.jsonl",
        help="JSONL written by tag_questions.py",
    )
    parser.add_argument(
        "--out_file",
        default="data-claude/tag_mapping.json",
        help="Where to write the mapping",
    )
    parser.add_argument(
        "--model",
        default="nvidia/Qwen3.6-35B-A3B-NVFP4",
        help="Model that does the semantic grouping",
    )
    parser.add_argument(
        "--api_base_url",
        default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8001/v1"),
    )
    parser.add_argument("--api_key", default=os.getenv("VLLM_API_KEY", "EMPTY"))
    parser.add_argument(
        "--batch_size",
        type=int,
        default=120,
        help="Tags per grouping call; batches are re-clustered afterwards",
    )
    parser.add_argument("--max_rounds", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument(
        "--no_llm",
        action="store_true",
        help="Lexical normalisation only, no model calls",
    )
    return parser.parse_args()


def read_tag_counts(path):
    counts = Counter()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            for tag in json.loads(line).get("tags", []):
                tag = str(tag).strip()
                if tag:
                    counts[tag] += 1
    return counts


def lexical_key(tag):
    """Normalised form used to merge surface variants of one tag."""
    text = tag.lower().strip()
    text = re.sub(r"[_/]+", " ", text)
    text = re.sub(r"[-‐-―]", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = []
    for word in text.split():
        if word in FILLER_WORDS:
            continue
        for suffix, replacement in LEMMA_SUFFIXES:
            if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                word = word[: len(word) - len(suffix)] + replacement
                break
        words.append(word)

    if not words:
        words = text.split()
    # Word order carries no meaning here: "logic formal" and "formal logic" are
    # the same capability, so sort before comparing.
    return " ".join(sorted(words))


def lexical_groups(counts):
    """Map each lexical key to the surface forms that share it."""
    groups = {}
    for tag, count in counts.most_common():
        groups.setdefault(lexical_key(tag), []).append((tag, count))
    return groups


def strip_fences(text):
    text = text.strip()
    fenced = re.match(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text


def parse_groups(raw_text):
    text = strip_fences(raw_text or "")
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return []
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return []

    groups = parsed.get("groups") if isinstance(parsed, dict) else parsed
    if not isinstance(groups, list):
        return []

    cleaned = []
    for group in groups:
        if not isinstance(group, dict):
            continue
        canonical = str(group.get("canonical", "")).strip().lower()
        members = [
            str(member).strip()
            for member in group.get("members", [])
            if str(member).strip()
        ]
        if canonical and members:
            cleaned.append((canonical, members))
    return cleaned


def cluster_batch(client, args, batch):
    """Ask the model to group one batch of (tag, count) pairs."""
    tag_block = "\n".join(f"- {tag} ({count})" for tag, count in batch)
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": CLUSTER_PROMPT.format(tag_block=tag_block)},
        ],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    raw_text = response.choices[0].message.content or ""
    groups = parse_groups(raw_text)

    known = {tag for tag, _ in batch}
    mapping = {}
    for canonical, members in groups:
        for member in members:
            if member in known and member not in mapping:
                mapping[member] = canonical
    # A tag the model dropped, renamed or hallucinated keeps its own name rather
    # than disappearing from the vocabulary.
    missing = sorted(known - set(mapping))
    for tag in missing:
        mapping[tag] = tag
    if missing:
        print(f"  [warn] {len(missing)} tag(s) not returned by the model, kept as-is")
    return mapping


def cluster_round(client, args, counts):
    """One pass of batched clustering over the current vocabulary."""
    ordered = counts.most_common()
    # Alphabetical within the batch keeps near-identical strings adjacent, so
    # the model sees them together.
    batches = [
        sorted(ordered[i : i + args.batch_size])
        for i in range(0, len(ordered), args.batch_size)
    ]
    mapping = {}
    for index, batch in enumerate(batches, start=1):
        print(f"  batch {index}/{len(batches)} ({len(batch)} tags)")
        mapping.update(cluster_batch(client, args, batch))
    return mapping


def compose(first, second):
    """Apply `second` on top of `first`, keeping the original keys."""
    return {tag: second.get(target, target) for tag, target in first.items()}


def main():
    args = parse_args()

    counts = read_tag_counts(args.tags_file)
    if not counts:
        sys.exit(f"No tags found in {args.tags_file}")
    print(f"{sum(counts.values())} tag occurrences, {len(counts)} unique raw tags")

    # Stage 1 - lexical
    mapping = {}
    current = Counter()
    for _, members in lexical_groups(counts).items():
        canonical = members[0][0].lower().strip()
        for tag, count in members:
            mapping[tag] = canonical
            current[canonical] += count
    print(f"after lexical normalisation: {len(current)} tags")

    # Stage 2 - semantic, repeated until a round merges nothing
    if not args.no_llm:
        client = OpenAI(base_url=args.api_base_url, api_key=args.api_key)
        for round_index in range(1, args.max_rounds + 1):
            if len(current) <= 1:
                break
            print(f"clustering round {round_index}: {len(current)} tags")
            round_mapping = cluster_round(client, args, current)

            merged = Counter()
            for tag, count in current.items():
                merged[round_mapping.get(tag, tag)] += count
            mapping = compose(mapping, round_mapping)

            if len(merged) == len(current):
                current = merged
                print(f"round {round_index} merged nothing, stopping")
                break
            print(f"round {round_index}: {len(current)} -> {len(merged)} tags")
            current = merged

    final_counts = Counter()
    for tag, count in counts.items():
        final_counts[mapping[tag]] += count

    payload = OrderedDict(
        [
            ("source", args.tags_file),
            ("model", None if args.no_llm else args.model),
            ("unique_raw_tags", len(counts)),
            ("unique_canonical_tags", len(final_counts)),
            ("mapping", OrderedDict(sorted(mapping.items()))),
            ("canonical_counts", OrderedDict(final_counts.most_common())),
        ]
    )
    os.makedirs(os.path.dirname(args.out_file) or ".", exist_ok=True)
    with open(args.out_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(f"\n{len(counts)} raw tags -> {len(final_counts)} canonical tags")
    print(f"Mapping written to {args.out_file}\n")
    print("Top canonical tags:")
    for tag, count in final_counts.most_common(25):
        variants = sorted({raw for raw, target in mapping.items() if target == tag})
        extra = f"  <- {', '.join(variants[:6])}" if len(variants) > 1 else ""
        print(f"  {count:4d}  {tag}{extra}")


if __name__ == "__main__":
    main()
