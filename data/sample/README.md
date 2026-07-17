# 公开数据样本

此目录用于展示项目的数据契约和离线处理结果，不是运行系统所需的完整数据集。

- `products_sample.csv`：商品基础字段的脱敏样本。
- `reviews_sample.csv`：关联商品的少量评论短片段，商品 ID、规格、店铺和标题已匿名化。
- `review_tag_evidence_sample.csv`：从评论中抽取的标签证据，包含命中词、倾向与证据质量。
- `product_tag_statistics_sample.csv`：按商品和标签聚合的统计结果，用于说明评分输入字段。

样本来源于影刀 RPA 采集的淘宝公开页面历史快照。完整原始数据、完整处理结果、SQLite 和 ChromaDB 运行数据均不公开。商品价格、销量和评论不代表实时信息。

`product_tag_statistics_sample.csv` 的统计值来自对应示例商品的完整本地评论集合；本目录仅展示其中少量评论，因此不能用样本文件重算全部统计值。

可在本地完整数据存在时运行下列命令重新导出样本：

```powershell
python scripts/core/export_public_sample.py
```
