# 2026-09-23 评审修订更新报告

## 交付结论

已按 `paper/review-2026-09-23-012947.tex` 完成一轮可交付修订。主文与补充材料均已重新编译，数值一致性校验通过，更新后的 PDF、图、表和 bundle 已同步。

## 已完成的修订

- 将标题和结论从“尺度帮助 identity”的因果表述改为“选择性语义对应、候选可用性与位姿恢复分离”。ViT-L/ViT-G 差异现在明确是最终保留集语义准确率的描述性点估计；候选层没有同口径 L/G 审计，因此不再声称候选身份改善已被隔离证明。
- 补齐匹配协议：20/20 帧、按 manifest 顺序取帧、224 输入、RGB 双线性/深度最近邻缩放、米制深度、14 像素 patch 中心、无跨帧 3-D 去重、逐层与拼接后的归一化、query-to-reference 变换方向。
- 补齐空间共识定义：候选最近邻指派、三点抽样、重复参考点允许、SVD Procrustes 反射修正、退化样本的实际处理、平局规则、一次重拟合/重指派，以及旋转角截断到 $[-1,1]$ 的公式和联合 $5^\circ/0.25$ m 门限。
- 统一“retained semantic accuracy”术语，修正主表为 Coverage / Sem. acc. / Rot. err. / Trans. err. / Strict pose；FPFH 标为非等密度诊断，不再与 DINO/FCGF 做直接覆盖率排序。
- 明确 mIoU 口径：区分 pooled conditional mIoU（61.2%）与 pair-macro conditional mIoU（67.3%）；新增固定查询总体、拒识视为假阴性的语义/实例 mIoU 表，当前 source-fit gate 的语义值为 4.1%。实例 ID 明确使用 3RScan `globalId`，不是 side-specific `objectId`。
- 将 “calibrated” 改为 empirical source-fit gate / source-only gate，保留 LOO、Clopper--Pearson 和 family-wise 无保证的限制说明。
- 失败分类图改为旋转--平移联合散点，虚线直接对应严格位姿门限，颜色为互斥的语义准确率区间，避免把仅旋转误差误当完整失败分类。
- 删除当前评审日期之后的 post-freeze external validation 和未来 MASt3R 发布路径叙述；README 和 paper README 改为 review-date snapshot 口径，不再把未来日期产物写成已发布事实。
- 更新表格生成器、图生成器、Makefile、manifest 清单和 consistency verifier；新增 `generated_abstention_miou_table.tex`。

## 验证结果

已运行：

```text
cd paper && make submission
python scripts/verify_submission_consistency.py --root .
```

结果：`status: pass`。

关键锁定值未漂移：ViT-G source coverage `0.3038277512`、source retained semantic accuracy `0.8499562555`、FCGF coverage `0.1873177390`、negative-control semantic accuracy `0.4632911392`、target source-fit semantic mIoU `0.8951512129`、instance mIoU `0.9071209477`。

## Prism 交接文件

- 主文源：[paper/main.tex](../../paper/main.tex)
- 补充材料源：[paper/supplementary.tex](../../paper/supplementary.tex)
- 主文 PDF：[paper/main.pdf](../../paper/main.pdf)
- 补充材料 PDF：[paper/supplementary.pdf](../../paper/supplementary.pdf)
- 更新后的失败图：[paper/figures/failure_taxonomy.pdf](../../paper/figures/failure_taxonomy.pdf)
- 新增拒识感知 mIoU 表：[paper/generated_abstention_miou_table.tex](../../paper/generated_abstention_miou_table.tex)
- 一致性报告：[outputs/reports/20260923_review_update.md](20260923_review_update.md)

## 尚未声称已解决的证据边界

当前版本没有新增 ViT-L/ViT-G 的 RANSAC 前 Recall@$K$、MRR 或候选排名分布，因此尺度效应仍只作为最终保留集指标上的描述性差异报告；若 Prism 要求“尺度帮助候选身份”的强因果结论，仍需补跑同一查询/候选池上的候选层审计。
