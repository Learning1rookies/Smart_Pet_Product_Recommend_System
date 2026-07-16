"""
意义： 明确输入、中间状态、明确输出

步骤： 定义数据结构——1

"""



from __future__ import annotations

from dataclasses import dataclass   # 快速创建数据结构


PRODUCT_TYPES = (
    "智能宠物喂食器",
    "智能宠物饮水机",
    "智能宠物猫砂盆",
    "智能宠物摄像头",
    "智能宠物项圈",
    "智能逗猫器",
)


@dataclass(frozen=True)   # 只读模式，防止数据被串改
# 产品
class Product:
    product_id: str           # 产品ID
    product_type: str         # 产品类型（共六款产品）

    title: str                # 产品标题
    price: float | None  # 产品价格
    shop_name: str            # 店铺名称
    sales: int | None         # 销售数量
    source: str               # 货源



@dataclass(frozen=True)
# 评论
class Review:
    product_id: str           # 产品ID
    product_type: str         # 产品类型

    purchase_date: str        # 已购时间
    sku_type: str             # 产品款式
    review_content: str       # 回复内容


@dataclass(frozen=True)
# 评论标签证据：从评论句子中抽取出来的优点/问题/提及证据
class ReviewTagEvidence:
    review_id: str            # 评论唯一ID，通常来自 row_md5
    product_id: str           # 产品ID
    product_type: str         # 产品类型
    sku_type: str             # 产品款式
    tag_name: str             # 标签名称，例如 静音、漏水/密封
    evidence_type: str        # 证据类型：advantage/problem/mention/mixed
    matched_keyword: str      # 命中的关键词
    evidence_text: str        # 证据原文句子
    evidence_quality: str     # 证据质量：sentence_match/context_window
    source_method: str        # 证据来源方法


@dataclass(frozen=True)
# 商品标签统计：某个商品在某个标签上的聚合评分
class ProductTagStats:
    product_id: str                   # 产品ID
    product_type: str                 # 产品类型
    tag_name: str                     # 标签名称
    product_review_count: int         # 商品总评论数
    mention_count: int                # 提到该标签的评论数
    advantage_count: int              # 优点证据评论数
    problem_count: int                # 问题证据评论数
    mixed_count: int                  # 混合证据数
    neutral_count: int                # 只提到但未明确好坏的数量
    mention_rate: float               # 标签提及率
    smoothed_advantage_rate: float    # 贝叶斯平滑后的优势率
    smoothed_problem_rate: float      # 贝叶斯平滑后的问题率
    source_method: str  # 统计来源方法


    # &产品核心推荐评价指标
    confidence: float                 # 样本可信度
    advantage_support: float          # 优势支持度
    problem_pressure: float           # 风险压力值




@dataclass(frozen=True)
# 推荐需求
class RecommendationRequest:
    user_id: str                                # 用户ID
    session_id: str                             # 会话ID
    product_type: str | None                    # 产品类型
    budget_min: float | None               # 预算最低值
    budget_max: float | None               # 预算最高值
    pet_type: str | None                        # 宠物类型
    priority_tags: list[str]                    # 优先关注标签项列表
    avoid_tags: list[str]                       # 避免标签项列表——用户明确不想要什么
    conversation_history: list[dict[str, str]]  # 交流历史——让AIagent记住以前的对话


@dataclass(frozen=True)
# 推荐响应
class RecommendationResponse:
    recommended_products: list[dict]      # 推荐产品列表
    comparison_table: list[dict]          # 竞争产品列表
    recommendation_reason: str            # 推荐理由
    review_evidence: list[dict]           # 评论证据列表
    risk_notes: list[str]                 # 风险警告——控制幻觉和过度承诺
    required_action: str | None           # 前端需要执行的结构化补充动作
    action_options: list                  # 前端弹出选择项
