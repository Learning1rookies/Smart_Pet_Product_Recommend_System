from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import md5
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_REVIEWS_PATH = PROJECT_ROOT / "data" / "raw" / "raw_reviews.csv"
EVIDENCE_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "review_tag_evidence.csv"
STATS_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "product_tag_stats.csv"

MIN_REVIEW_TEXT_LENGTH = 4
TARGET_SAMPLE_SIZE = 20
BAYES_PRIOR_SAMPLE_SIZE = 10


@dataclass(frozen=True)
class ReviewRow:
    review_id: str
    product_id: str
    product_type: str
    sku_type: str
    review_content: str


@dataclass(frozen=True)
class EvidenceRule:
    tag_name: str
    mention_keywords: tuple[str, ...]
    advantage_keywords: tuple[str, ...]
    problem_keywords: tuple[str, ...]


TAG_EVIDENCE_RULES = (
    EvidenceRule(
        tag_name="噪音/静音",
        mention_keywords=("声音", "噪音", "静音", "嗡嗡", "吵"),
        advantage_keywords=("声音小", "声音很小", "噪音小", "静音", "不吵", "安静", "晚上也不吵"),
        problem_keywords=("声音大", "噪音大", "太吵", "很吵", "嗡嗡响", "吵人", "吵醒", "有点吵"),
    ),
    EvidenceRule(
        tag_name="联网/APP",
        mention_keywords=("app", "APP", "联网", "远程", "手机", "连接", "wifi", "WiFi", "断网", "掉线"),
        advantage_keywords=("连接稳定", "联网稳定", "远程方便", "远程控制", "手机控制", "操作简单", "app好用", "APP好用"),
        problem_keywords=("连不上", "连接不上", "掉线", "断网", "离线", "网络不稳定", "app不好用", "APP不好用"),
    ),
    EvidenceRule(
        tag_name="清洁/拆洗",
        mention_keywords=("清洗", "清洁", "拆洗", "好洗", "滤芯", "耗材", "换水"),
        advantage_keywords=("好清洗", "方便清洗", "容易清洗", "拆洗方便", "清洁方便", "好拆", "省心"),
        problem_keywords=("不好洗", "难清洗", "难拆", "清洗麻烦", "容易脏", "有水垢", "耗材贵"),
    ),
    EvidenceRule(
        tag_name="容量/空间",
        mention_keywords=("容量", "水箱", "粮桶", "空间", "大容量", "够大", "出差"),
        advantage_keywords=("容量大", "大容量", "够大", "够用", "出差够用", "不用频繁加", "水箱大"),
        problem_keywords=("容量小", "不够用", "水箱小", "频繁加水", "频繁加粮", "占地方", "体积大"),
    ),
    EvidenceRule(
        tag_name="漏水/密封",
        mention_keywords=("漏水", "渗水", "溢水", "溢出来", "密封", "漏"),
        advantage_keywords=("不漏水", "没有漏水", "密封好", "不会漏", "不渗水"),
        problem_keywords=("漏水", "渗水", "溢水", "溢出来", "漏出来", "到处是水", "水漏"),
    ),
    EvidenceRule(
        tag_name="稳定/卡顿",
        mention_keywords=("稳定", "卡", "卡住", "卡粮", "故障", "坏了", "不动", "顺畅"),
        advantage_keywords=("稳定", "很稳定", "不卡", "不卡粮", "出粮顺畅", "运行顺畅", "目前没问题"),
        problem_keywords=("卡住", "卡粮", "经常卡", "故障", "坏了", "不动", "失灵", "不能用", "用不了"),
    ),
    EvidenceRule(
        tag_name="定位/信号",
        mention_keywords=("定位", "信号", "轨迹", "位置", "防丢", "找回", "漂移"),
        advantage_keywords=("定位准确", "定位准", "信号好", "轨迹清楚", "位置准确", "找得到"),
        problem_keywords=("定位不准", "不准确", "信号差", "没信号", "漂移", "位置不对", "找不到"),
    ),
    EvidenceRule(
        tag_name="画质/夜视",
        mention_keywords=("画质", "清晰", "夜视", "像素", "摄像", "监控", "云台"),
        advantage_keywords=("画质清晰", "很清晰", "夜视清楚", "像素高", "看得清", "监控方便"),
        problem_keywords=("不清晰", "看不清", "模糊", "夜视差", "画质差", "卡顿"),
    ),
    EvidenceRule(
        tag_name="宠物兴趣",
        mention_keywords=("喜欢", "爱玩", "不玩", "害怕", "追着", "兴趣", "玩"),
        advantage_keywords=("很喜欢", "特别喜欢", "爱玩", "追着玩", "玩得很开心", "有兴趣"),
        problem_keywords=("不喜欢", "不玩", "没兴趣", "害怕", "吓到", "不敢靠近"),
    ),
    EvidenceRule(
        tag_name="佩戴/舒适",
        mention_keywords=("项圈", "佩戴", "脖子", "勒", "重量", "硅胶", "舒服"),
        advantage_keywords=("佩戴舒服", "不勒", "轻便", "重量轻", "材质舒服", "大小合适"),
        problem_keywords=("勒脖子", "重量大", "比较重", "不舒服", "磨脖子", "容易掉", "尺寸不合适"),
    ),
    EvidenceRule(
        tag_name="安全风险",
        mention_keywords=("安全", "危险", "电击", "夹", "夹猫", "刺激", "吓"),
        advantage_keywords=("安全", "安全可靠", "没有刺激", "不伤", "放心"),
        problem_keywords=("危险", "电击", "夹猫", "夹到", "刺激大", "吓到", "不安全"),
    ),
    EvidenceRule(
        tag_name="性价比",
        mention_keywords=("价格", "便宜", "实惠", "性价比", "划算", "贵", "值"),
        advantage_keywords=("便宜", "实惠", "性价比高", "划算", "值得", "价格合适", "物美价廉"),
        problem_keywords=("贵", "太贵", "不值", "性价比低", "价格高", "不划算"),
    ),
)


NEGATION_ADVANTAGE_PREFIXES = ("不", "没", "没有", "不会", "无")


def normalize_text(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(r"商家回复[:：].*$", "", text)
    return text.strip()


def make_review_id(row: dict[str, str]) -> str:
    row_md5 = normalize_text(row.get("row_md5"))
    if row_md5:
        return row_md5
    raw = "|".join(
        [
            normalize_text(row.get("product_id")),
            normalize_text(row.get("sku_type")),
            normalize_text(row.get("review_content")),
        ]
    )
    return md5(raw.encode("utf-8")).hexdigest()


def read_reviews(path: Path) -> list[ReviewRow]:
    reviews: list[ReviewRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            content = normalize_text(row.get("review_content"))
            if len(content) < MIN_REVIEW_TEXT_LENGTH:
                continue
            reviews.append(
                ReviewRow(
                    review_id=make_review_id(row),
                    product_id=normalize_text(row.get("product_id")),
                    product_type=normalize_text(row.get("product_type")),
                    sku_type=normalize_text(row.get("sku_type")),
                    review_content=content,
                )
            )
    return reviews


def split_sentences(content: str) -> list[str]:
    chunks = re.split(r"(?<=[。！？!?；;])|\s{2,}", content)
    sentences = [normalize_text(chunk.strip("，,。！？!?；; ")) for chunk in chunks if normalize_text(chunk)]
    return sentences or [content]


def contains_keyword(text: str, keyword: str) -> bool:
    if re.search(r"[A-Za-z]", keyword):
        return keyword.lower() in text.lower()
    return keyword in text


def find_first_keyword(text: str, keywords: tuple[str, ...]) -> str:
    matched = [keyword for keyword in keywords if contains_keyword(text, keyword)]
    if not matched:
        return ""
    return sorted(matched, key=len, reverse=True)[0]


def has_negated_problem(text: str, problem_keyword: str) -> bool:
    escaped_keyword = re.escape(problem_keyword)
    if re.search(rf"(不|没|没有|不会|无)(会|有|出现|出现过)?{escaped_keyword}", text):
        return True
    for prefix in NEGATION_ADVANTAGE_PREFIXES:
        if f"{prefix}{problem_keyword}" in text:
            return True
    return False


def classify_sentence(sentence: str, rule: EvidenceRule) -> tuple[str, str]:
    advantage_keyword = find_first_keyword(sentence, rule.advantage_keywords)
    problem_keyword = find_first_keyword(sentence, rule.problem_keywords)
    mention_keyword = find_first_keyword(sentence, rule.mention_keywords)

    if advantage_keyword and problem_keyword:
        if advantage_keyword in problem_keyword and len(problem_keyword) > len(advantage_keyword):
            return "problem", problem_keyword
        if has_negated_problem(sentence, problem_keyword):
            return "advantage", advantage_keyword
        return "mixed", f"{advantage_keyword}|{problem_keyword}"
    if advantage_keyword:
        return "advantage", advantage_keyword
    if problem_keyword:
        if has_negated_problem(sentence, problem_keyword):
            return "advantage", f"否定问题:{problem_keyword}"
        return "problem", problem_keyword
    if mention_keyword:
        return "mention", mention_keyword
    return "", ""


def build_context(sentences: list[str], index: int) -> tuple[str, str]:
    sentence = sentences[index]
    if len(sentence) >= 12:
        return sentence[:180], "sentence_match"
    left = sentences[index - 1] if index > 0 else ""
    right = sentences[index + 1] if index + 1 < len(sentences) else ""
    context = "。".join(part for part in (left, sentence, right) if part)
    return context[:180], "context_window"


def extract_evidence_for_review(review: ReviewRow) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    sentences = split_sentences(review.review_content)
    seen: set[tuple[str, str, str]] = set()

    for index, sentence in enumerate(sentences):
        for rule in TAG_EVIDENCE_RULES:
            evidence_type, matched_keyword = classify_sentence(sentence, rule)
            if not evidence_type:
                continue
            evidence_text, evidence_quality = build_context(sentences, index)
            key = (rule.tag_name, evidence_type, evidence_text)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "review_id": review.review_id,
                    "product_id": review.product_id,
                    "product_type": review.product_type,
                    "sku_type": review.sku_type,
                    "tag_name": rule.tag_name,
                    "evidence_type": evidence_type,
                    "matched_keyword": matched_keyword,
                    "evidence_text": evidence_text,
                    "evidence_quality": evidence_quality,
                    "source_method": "rule_sentence_keyword",
                }
            )
    return rows


def confidence_score(mention_count: int) -> float:
    if mention_count <= 0:
        return 0.0
    return round(min(1.0, math.log1p(mention_count) / math.log1p(TARGET_SAMPLE_SIZE)), 4)


def bayesian_smoothed_rate(count: int, total: int, prior_rate: float, prior_sample_size: int) -> float:
    if total < 0 or count < 0:
        return 0.0
    numerator = count + prior_rate * prior_sample_size
    denominator = total + prior_sample_size
    return numerator / denominator if denominator else 0.0


def build_product_tag_stats(reviews: list[ReviewRow], evidence_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    review_count_by_product = Counter(review.product_id for review in reviews)
    review_ids_by_product_tag: dict[tuple[str, str], set[str]] = defaultdict(set)
    advantage_ids_by_product_tag: dict[tuple[str, str], set[str]] = defaultdict(set)
    problem_ids_by_product_tag: dict[tuple[str, str], set[str]] = defaultdict(set)
    mixed_ids_by_product_tag: dict[tuple[str, str], set[str]] = defaultdict(set)
    product_type_by_product: dict[str, str] = {}

    for review in reviews:
        product_type_by_product[review.product_id] = review.product_type

    for row in evidence_rows:
        key = (str(row["product_id"]), str(row["tag_name"]))
        review_id = str(row["review_id"])
        review_ids_by_product_tag[key].add(review_id)
        if row["evidence_type"] == "advantage":
            advantage_ids_by_product_tag[key].add(review_id)
        elif row["evidence_type"] == "problem":
            problem_ids_by_product_tag[key].add(review_id)
        elif row["evidence_type"] == "mixed":
            mixed_ids_by_product_tag[key].add(review_id)

    prior_counts_by_type_tag: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"mention": 0, "advantage": 0, "problem": 0})
    for product_id, tag_name in review_ids_by_product_tag:
        prior_key = (product_type_by_product.get(product_id, ""), tag_name)
        stat_key = (product_id, tag_name)
        prior_counts_by_type_tag[prior_key]["mention"] += len(review_ids_by_product_tag[stat_key])
        prior_counts_by_type_tag[prior_key]["advantage"] += len(advantage_ids_by_product_tag[stat_key])
        prior_counts_by_type_tag[prior_key]["problem"] += len(problem_ids_by_product_tag[stat_key])

    rows: list[dict[str, object]] = []
    for product_id, tag_name in sorted(review_ids_by_product_tag):
        mentioned = review_ids_by_product_tag[(product_id, tag_name)]
        advantage = advantage_ids_by_product_tag[(product_id, tag_name)]
        problem = problem_ids_by_product_tag[(product_id, tag_name)]
        mixed = mixed_ids_by_product_tag[(product_id, tag_name)]
        mention_count = len(mentioned)
        product_review_count = review_count_by_product[product_id]
        neutral_count = max(0, mention_count - len(advantage | problem | mixed))
        confidence = confidence_score(mention_count)
        product_type = product_type_by_product.get(product_id, "")
        prior_counts = prior_counts_by_type_tag[(product_type, tag_name)]
        prior_mention_count = prior_counts["mention"]
        prior_advantage_rate = prior_counts["advantage"] / prior_mention_count if prior_mention_count else 0.0
        prior_problem_rate = prior_counts["problem"] / prior_mention_count if prior_mention_count else 0.0
        smoothed_advantage_rate = bayesian_smoothed_rate(
            len(advantage),
            mention_count,
            prior_advantage_rate,
            BAYES_PRIOR_SAMPLE_SIZE,
        )
        smoothed_problem_rate = bayesian_smoothed_rate(
            len(problem),
            mention_count,
            prior_problem_rate,
            BAYES_PRIOR_SAMPLE_SIZE,
        )
        problem_pressure = round(smoothed_problem_rate * confidence, 4)
        advantage_support = round(smoothed_advantage_rate * confidence, 4)

        rows.append(
            {
                "product_id": product_id,
                "product_type": product_type,
                "tag_name": tag_name,
                "product_review_count": product_review_count,
                "mention_count": mention_count,
                "advantage_count": len(advantage),
                "problem_count": len(problem),
                "mixed_count": len(mixed),
                "neutral_count": neutral_count,
                "mention_rate": round(mention_count / product_review_count, 4) if product_review_count else 0.0,
                "smoothed_advantage_rate": round(smoothed_advantage_rate, 4),
                "smoothed_problem_rate": round(smoothed_problem_rate, 4),
                "confidence": confidence,
                "advantage_support": advantage_support,
                "problem_pressure": problem_pressure,
                "source_method": "rule_sentence_keyword",
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    reviews = read_reviews(RAW_REVIEWS_PATH)
    evidence_rows: list[dict[str, object]] = []
    for review in reviews:
        evidence_rows.extend(extract_evidence_for_review(review))

    stats_rows = build_product_tag_stats(reviews, evidence_rows)

    write_csv(
        EVIDENCE_OUTPUT_PATH,
        evidence_rows,
        [
            "review_id",
            "product_id",
            "product_type",
            "sku_type",
            "tag_name",
            "evidence_type",
            "matched_keyword",
            "evidence_text",
            "evidence_quality",
            "source_method",
        ],
    )
    write_csv(
        STATS_OUTPUT_PATH,
        stats_rows,
        [
            "product_id",
            "product_type",
            "tag_name",
            "product_review_count",
            "mention_count",
            "advantage_count",
            "problem_count",
            "mixed_count",
            "neutral_count",
            "mention_rate",
            "smoothed_advantage_rate",
            "smoothed_problem_rate",
            "confidence",
            "advantage_support",
            "problem_pressure",
            "source_method",
        ],
    )

    print(f"reviews: {len(reviews)}")
    print(f"evidence_rows: {len(evidence_rows)} -> {EVIDENCE_OUTPUT_PATH}")
    print(f"product_tag_stats: {len(stats_rows)} -> {STATS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
