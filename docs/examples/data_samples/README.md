# 数据教学的真实 SFT 样本

这五个文件来自 `ascendc-kernelgen-data@0607a29d231cde384c9dfdb8a39f54f91837c327`。原始文件已经通过 Git LFS 对象的 SHA-256 和字节数校验。摘出的 JSON 保留原字段值，只有空白排版改变。完整来源与校验信息在 [manifest.json](manifest.json)。

2026-09-28 已将这五个完整源文件放回原仓库的下表路径。这里保留的是单条教学摘录，原仓库中现在可以读取完整的 5,969、94、316、236 和 1,000 条记录。

| 文件 | 原文件 | 原记录索引，从 0 开始 |
|---|---|---:|
| [api_qa.json](api_qa.json) | `dataset/dataset/doc_data_9.1.0/ascendc_api_qa_910.json` | 0 |
| [vector_mse.json](vector_mse.json) | `dataset/dataset/vector_data/202605_0004.json` | 6 |
| [logit_generation.json](logit_generation.json) | `dataset/dataset/sft_distill_data/sft_gen.json` | 3 |
| [logit_debug.json](logit_debug.json) | `dataset/dataset/sft_distill_data/sft_debug.json` | 3 |
| [agent_groupedmatmul.json](agent_groupedmatmul.json) | `dataset/agent_dataset/train-00000.jsonl.zst`，解压后 | 10 |

Agent 原分片 `.jsonl.zst` 是 Zstandard 二进制压缩文件，文本编辑器显示乱码不代表编码损坏。要直接阅读完整的 1,000 条记录，请打开与原压缩文件同目录的 [train-00000.jsonl](../../../ascendc-kernelgen-data/dataset/agent_dataset/train-00000.jsonl)。解压文件约 333 MiB，先读单条可使用 [agent_groupedmatmul.json](agent_groupedmatmul.json)。当前已经解压完成，原 `.zst` 保留；仅在解压文件缺失时，才需按学习文档第 4.1 节重新解压。

每个摘出文件是单条 JSON 对象。给教学工具传入这些文件时使用 `--index 0`；给工具传入原始完整文件时，使用上表的原记录索引。

```bash
# 从 workspace 根目录执行，输出目录须不存在
python3 docs/examples/inspect_sft_data.py \
  --input docs/examples/data_samples/vector_mse.json \
  --index 0 --out /tmp/ascend-data-vector-example
```

生成 `records.raw.jsonl`、`messages.jsonl` 和转换 manifest。工具只读取数据；不会执行其中的 C++、Python、shell 或 Agent 动作。没有实现 tokenizer 或 token mask。

这些是教学材料，不是经过当前 t2a judge 复验的训练集。尤其是历史 Agent 轨迹包含 `profile.t2a.yaml` 禁用的 `Agent` 动作，不能直接作为该配置下工具协议的示范。样本中的成功、精度和硬件表述保留原文，不代表本次实测。

详细字段、数学语义、转换及评测步骤见 [SFT 与 RL 数据学习文档](../../ascendc-data-learning-guide.md)。来源许可见 [原仓库 LICENSE](../../../ascendc-kernelgen-data/LICENSE)。
