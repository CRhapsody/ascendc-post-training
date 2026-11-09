# 当前 t2a skill 文档装载与调用流程核对

核对日期：2026-10-12。此文只读分析，未启动训练、NPU、容器或付费模型，也未修改任何项目文件。

## 1. 版本、适用前提与复现缺口

- ProRL-Agent-Server：分支 `feat/ascendc-rl-t2a`，HEAD `a8e306509e0e246c5fc8ca9958dc98d222a385e0`；本轮 `git status --short` 为空。
- cannbot-skills：分支 `master`，HEAD `99fbf4655460ecd8a30c65f7ba7e53a0bf9bf346`；本轮 `git status --short` 为空。
- 下文“当前 RL 环境”仅指显式加载 `profile.t2a.yaml` 或等效渲染配置的情况。切分支本身不会选择 profile。启动脚本默认 `profile.yaml`，只有显式 `POLAR_PROFILE` 才换配置，见 [start_polar_nohup.sh:9](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/deploy/ascend_operator/start_polar_nohup.sh:9)。没有运行中的服务器证据，不能声称这就是某次实际训练已经加载的配置。
- t2a profile 指向 `operator_runtime_t2a`，workflow 是 `legacy`，backend 是 `ascendc`，并配置服务器上的任务 JSON 目录和 asc-devkit 目录，见 [profile.t2a.yaml:21](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/deploy/ascend_operator/profile.t2a.yaml:21)。
- 当前 canonical 实际有 **12 个** `skills/*/SKILL.md`。profile 开头“21 skills、TileLang 单路径”是历史注释，不符合当前目录和流程。
- 只读校验结果：`build_claude_md.py --canonical operator_runtime_t2a --check` 报“CLAUDE.md 与 build_claude_md.py 的产物不一致”；`gen_skill_reference_list.py --canonical operator_runtime_t2a --check` 报“skill reference list up to date”。本轮没有修复。完整 preflight 在前者会失败，见 [preflight.sh:119](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/deploy/ascend_operator/preflight.sh:119)。因此本文描述“当前 committed CLAUDE.md 被 prepare 复制后的规则”，不承诺配置原样通过启动前检查。

当前 CLAUDE.md 比生成器和 claude_override.md 多出“简单路径内嵌设计”和“骨架不是语义契约”的内容。源码证据在 [CLAUDE.md:432](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/operator_runtime_t2a/CLAUDE.md:432)、[CLAUDE.md:655](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/operator_runtime_t2a/CLAUDE.md:655)；prepare 实际复制 canonical 的 CLAUDE.md，而不现场调用生成器，见 [prepare_operator_workdir.py:1175](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/operator_runtime_t2a/runtime/prepare_operator_workdir.py:1175)。写教学流程时优先讲该 committed 文件，另列生成漂移为接线缺口。

## 2. 三层来源不能混成一套

1. **历史 10 条 SFT 轨迹**：真实 messages 记录当时看到的旧 `ascendc-*` 名称、Skill 工具、Read/Bash 访问和脚本结果。应逐条以轨迹的实际 tool call 为准。它们是训练数据中的历史 Agent 求解过程，不能当作当前 t2a 的新 rollout，也不能据“曾调用 Skill”推出现在仍允许 Skill。
2. **当前 cannbot-skills 上游 master**：主插件位于 `plugins-community/tilelang2ascendc-ops-generator`；自己的 6 个知识包在插件 skills 下，共享知识包来自根目录 `ops/`。其 Agent 仍是 standalone 双路径：简单算子 ops-direct-invoke，复杂算子 TileLang，并包含 case-simplifier、完整用例恢复、trace/知识演进等，见 [上游 Agent:48](/Users/chizm/Documents/ascend-project/cannbot-skills/plugins-community/tilelang2ascendc-ops-generator/agents/tilelang2ascendc-kernel-generator.md:48)。这不是 RL runtime 的最终指令。
3. **当前 RL canonical 副本**：`ProRL-Agent-Server/operator_runtime_t2a` 是经过裁剪和修改、纳入 ProRL 版本控制的运行时。不会在每条 rollout 里临时 git pull 最新 cannbot。主轨迹读的是这份副本。逐字比较当前 12 个 SKILL.md 与上述 master 对应文件，仅 `tilelang2ascend-precision-tuning/SKILL.md` 相同，其余 11 个不同。这只说明文件内容不同，不能把差异都解释成本地 patch，因为也存在上游版本推进。

迁移笔记的目标基点写的是官方 `f76269e`，旧基点是另一 fork 的 `6cf50e29`，见 [MIGRATION_T2A.md:5](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/deploy/ascend_operator/MIGRATION_T2A.md:5)。当前 translator 内嵌设计又明确引用上游 `13b2ae5652c75fe83a3e4114552a6477d3a01f3d` 的设计资料，见 [translator/SKILL.md:305](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/operator_runtime_t2a/skills/tilelang2ascend-translator/SKILL.md:305)。这足以说明运行时不是眼前 master 的原样安装。

## 3. 旧名到当前文档的映射

“对应”表示用途和流程位置的映射。除有历史记录支持的改名外，不声称两文件内容等价。

| 历史旧名 | 当前 cannbot master 的对应入口 | 当前 RL t2a 中的实际地位 |
|---|---|---|
| ascendc-case-simplifier | tilelang2ascend-case-simplifier | 上游仍有；当前 canonical 已没有，Phase 2/6 移出 RL 主流程 |
| ascendc-operator-project-init | tilelang2ascend-operator-project-init | 上游仍有；当前 canonical 已没有，由 prepare 生成工程骨架，Agent 不再初始化 |
| ascendc-design-doc-generator | 无同名新 skill；上游简单路径转为 ops-direct-invoke 的设计流程 | 不是简单改名前缀。当前 RL 用 translator 中“内嵌设计与审查”和 cannbot 设计模板，不启动 Architect 子代理 |
| ascendc-code-gen | 无同名新 skill；上游简单路径转为 ops-direct-invoke 的开发流程 | 不是 tilelang2ascend-code-gen。当前 RL 简单/复杂统一由 tilelang2ascend-translator 指导实现 |
| ascendc-tilelang-designer | tilelang2ascend-tilelang-designer | 当前存在；仅 fused/other 的复杂路径默认读取，最多 3 份 AST 候选 |
| ascendc-translator | tilelang2ascend-translator | 当前存在；两种路径均读取，角色已包括直接依据 reference 写 kernel |
| ascendc-performance-analyzer | ops-profiling | 迁移记录明确“换成 ops-profiling”，是性能工具和知识入口替换，不应写成机械改名前缀 |
| ascendc-precision-debug | ascendc-precision-debug | 当前存在，D 类数值不匹配必读；输出全零/NaN 等也可能由 A 类路由提示读取 |
| ascendc-precision-tuning | tilelang2ascend-precision-tuning | 当前存在，DumpTensor 专题补充，不再有“7 次 debug 后自动切到 5 次 tuning”的 RL 固定阶梯 |
| ascendc-trace-recorder | tilelang2ascend-trace-recorder | 上游仍有；当前 canonical 已没有，Trace 只用于离线复盘，不进入求解主流程 |
| npu-arch | npu-arch | 当前存在，所有路径实现 AscendC 前读取 |

删除/替换证据：[SESSION_AUDIT_20260728.md:79](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/deploy/ascend_operator/SESSION_AUDIT_20260728.md:79) 明写旧 design-doc-generator/code-gen 被删除、简单流程改成 ops-direct-invoke；[MIGRATION_T2A.md:19](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/deploy/ascend_operator/MIGRATION_T2A.md:19) 明写 performance-analyzer 换 ops-profiling；[build_claude_md.py:201](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/deploy/ascend_operator/build_claude_md.py:201) 说明本 RL 又移除了 ops-direct-invoke 子代理入口，两条路共用 translator。旧 runtime 仍在 `operator_runtime_ascendc/skills` 保存 11 个旧入口，可以用于读旧轨迹，但它不是 t2a 的装载源。

当前 12 个知识包完整清单：

- ascendc-api-best-practices、ascendc-crash-debug、ascendc-docs-search、ascendc-precision-debug、ascendc-runtime-debug、ascendc-tiling-design
- npu-arch、ops-precision-standard、ops-profiling
- tilelang2ascend-precision-tuning、tilelang2ascend-tilelang-designer、tilelang2ascend-translator

前三个 tilelang2ascend 包的上游位置是 `plugins-community/tilelang2ascendc-ops-generator/skills/<name>`，其余 9 个对应 `cannbot-skills/ops/<name>`。上游开发 skill bundle 的实际数组只有 6 个条目，见 [marketplace.json:675](/Users/chizm/Documents/ascend-project/cannbot-skills/.claude-plugin/marketplace.json:675)；不要把 description 里“7 个”“9 个”等历史数字当目录统计。

## 4. 文档如何铺进每条 rollout 的工作目录

上游 `init.sh` 是供独立工具安装使用的另一条路线：从 AGENTS.md 和 agents/*.md frontmatter 收集 skill 名，扫描共享 ops/ 和本插件 skills/，为每个 skill 建发现用软链，并配置 workflows/agents/hooks，再尝试 clone/pull asc-devkit。见 [init.sh:116](/Users/chizm/Documents/ascend-project/cannbot-skills/plugins-community/tilelang2ascendc-ops-generator/init.sh:116)、[init.sh:449](/Users/chizm/Documents/ascend-project/cannbot-skills/plugins-community/tilelang2ascendc-ops-generator/init.sh:449)、[init.sh:553](/Users/chizm/Documents/ascend-project/cannbot-skills/plugins-community/tilelang2ascendc-ops-generator/init.sh:553)。本轮没有执行。当前 RL 不在每条任务里跑它。

当前 RL 的准备链：

1. profile 将 canonical 挂载到容器 `/opt/canonical:ro`，将 tools 挂到 `/opt/workspace/agent_workdir/tools:ro`；asc-devkit 单独只读挂载到 `/opt/asc-devkit`。见 [load_polar_profile.py:49](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/deploy/ascend_operator/tools/load_polar_profile.py:49)。
2. 渲染 prepare actions：上传 task_source 到 `input/{op_name}.py`；若任务目录存在 JSON 用例，则追加同名 JSON 上传；执行 `prepare_operator_workdir.py --backend ascendc --no-stub --readonly-tools --only-project-skills --require-claude`。judge 的 prepare 不带 require-claude。见 [load_polar_profile.py:86](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/deploy/ascend_operator/tools/load_polar_profile.py:86)。
3. prepare 创建 input/、judge_out/，检查 reference 与需要的同名 JSON，复制 canonical/skills 到 `.claude/skills`、workflows 到 `.claude/workflows`、CLAUDE.md 到工作区根目录。**不复制 agents/**，因为那会注册额外子代理且其拿不到当前覆盖区。见 [prepare_operator_workdir.py:1132](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/operator_runtime_t2a/runtime/prepare_operator_workdir.py:1132)。
4. 只在 Agent 侧，根据 reference 签名生成顶层 `{op_name}/` 的 wrapper、注册、host/device 等骨架，并从模板铺 CMake/setup/utils。签名解析失败会回退模板，已有文件不覆盖，因此骨架不等于实现了 reference 的语义。见 [prepare_operator_workdir.py:875](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/operator_runtime_t2a/runtime/prepare_operator_workdir.py:875)、[CLAUDE.md:682](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/operator_runtime_t2a/CLAUDE.md:682)。
5. prepare 写 .claude/settings.json，关闭已知 CLI bundled skills，并装只读 Stop 完成门禁。profile 的 `skills_path: ""` 让 harness 不再把第二份 skills 复制到会话级目录，见 [profile.t2a.yaml:119](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/deploy/ascend_operator/profile.t2a.yaml:119)。harness 对非空 skills_path 才复制，见 [claude_code.py:49](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/src/polar/agent/presets/claude_code.py:49)。
6. Claude Code 非交互启动，传入 profile 的 allowed/disallowed tools。**允许 Bash/Read/Edit/Write/Grep/Glob，明确禁用 Skill 和 Agent 等工具**，见 [profile.t2a.yaml:134](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/deploy/ascend_operator/profile.t2a.yaml:134)、[claude_code.py:64](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/src/polar/agent/presets/claude_code.py:64)。

“加载文档”至少分三件事：文件被复制到可读取位置；CLAUDE.md 提供流程与路径索引；Agent 的 Read 调用才把选中文件内容带入当前轨迹。不能把“目录里有 12 个 SKILL.md”写成“每条 rollout 预先读完 12 份全文”。

`gen_skill_reference_list.py` 从真实目录生成可解析路径索引，包含 SKILL.md，默认每个 skill 最多列 24 项，折叠模板目录，不执行这些知识包；`--check` 检查漂移。见 [gen_skill_reference_list.py:24](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/deploy/ascend_operator/gen_skill_reference_list.py:24)、[gen_skill_reference_list.py:65](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/deploy/ascend_operator/gen_skill_reference_list.py:65)。

`build_claude_md.py` 读取 ProRL 内保存的 upstream agent 文本，应用 D1–D13 等声明式改动，再拼接 claude_override.md，之后由索引生成器填 reference block；它不是直接实时读取本机 cannbot master。见 [build_claude_md.py:24](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/deploy/ascend_operator/build_claude_md.py:24)、[build_claude_md.py:620](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/deploy/ascend_operator/build_claude_md.py:620)。

## 5. 当前一条 RL rollout 应如何读文档并推进

这里是当前规则约束的预期流程，不是声称已观察到 10 条历史轨迹执行了该流程。

### 5.1 先 reference，再结构分类

先读原始 input/{op}.py 的 __init__、完整 forward、辅助逻辑、实际存在的输入生成入口和用例。当前规则禁止按名字、编号、历史白名单分 simple/complex。Agent 记录：

- `algorithm_classification="single_op"`：一个主导计算原语，或至多两个不需要分层定位的简单子步骤。
- `"fused"`：至少三个相互依赖的阶段，或者需要明确中间张量/阶段边界的多阶段数据流。
- `"other"`：源码缺失、随机/不可分解，或无法可靠建立计算结构。

再派生 `op_type`：single_op 为 simple，fused/other 为 complex。规则在 [CLAUDE.md:50](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/operator_runtime_t2a/CLAUDE.md:50)，记录要求在 [CLAUDE.md:322](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/operator_runtime_t2a/CLAUDE.md:322)。

**没有发现名为 task_kind 的现成输入字段或 Python 自动分类实现。** 搜索当前 ProRL/vime 的 Python、YAML、shell 和相关文档，algorithm_classification 只命中 CLAUDE.md 与它的生成器。这是提示词要求模型进行的结构判断，不是确定性编排器从样本字段读取的路由。不能从历史轨迹“走了旧 simple 路径”推出它在当前规则下一定 single_op，更不能看到算子名就定类。对那 10 个算子，若没有按当前 reference 重新分类并记录证据，教程应画带条件的两条路径。

### 5.2 共同知识与简单路径

全部路径在实现 AscendC 前 Read npu-arch/SKILL.md，按当前 SOC 取硬件参数；不要照旧文档自行 npu-smi 抢卡。[CLAUDE.md:493](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/operator_runtime_t2a/CLAUDE.md:493)

simple 只跳过 TileLang，不跳过设计。Read translator/SKILL.md 的“内嵌设计与审查”，继续按其指引读 .claude/workflows/templates/design-template.md 的相关章节、tiling/API 索引与命中的 references；主轨迹简短核对核/tile、归约/扫描轴、非连续输入、尾块、累加 dtype、输出回写六项。不新增 Architect/Reviewer 子代理，也不要求 DESIGN/PLAN/WALKTHROUGH/REVIEW 四份文件。[translator/SKILL.md:305](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/operator_runtime_t2a/skills/tilelang2ascend-translator/SKILL.md:305)

写 kernel 前按 ascendc-docs-search 和 translator 指引核对本实现实际使用的 API 文档、示例。当前文档根为 `$ASC_DEVKIT_DIR`；首次查实际使用 API，修复只查当前错误/新增/准备修改的 API，不要求每轮重复读全库。[translator/SKILL.md:83](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/operator_runtime_t2a/skills/tilelang2ascend-translator/SKILL.md:83)

### 5.3 复杂路径的 TileLang 是中间设计

complex 先 Read tilelang2ascend-tilelang-designer/SKILL.md，形成 block_level、tile_level、model_new_tilelang.py。每次修改后执行本地不占卡的 validate_tilelang_impl.py，最多 3 份候选。通过就把设计交给 translator；第 3 次仍失败，丢弃退化 wrapper，回到 reference 与原骨架继续实现，不能让中间表示失败直接结束 session。[CLAUDE.md:446](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/operator_runtime_t2a/CLAUDE.md:446)

当前不自动运行 TileLang 上板正确性和性能循环。verification_tilelang.py 仅在后续明确需要 DSL 诊断时经 lease 手工调用，不能作为最终 correctness/performance gate。[CLAUDE.md:715](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/operator_runtime_t2a/CLAUDE.md:715)

### 5.4 translator 读知识，当前会话写代码

两条路都 Read tilelang2ascend-translator/SKILL.md，由当前主会话补原位 AscendC 数学、tiling 和必要接线。不是“调用一个专门模型自动转译”。对于有 attention 或 MIX/CrossCore/Workspace 同步特征的代码，translator 内另有条件门禁和命中文档要求；没有触发则不必把对应专题全部读完。[translator/SKILL.md:108](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/operator_runtime_t2a/skills/tilelang2ascend-translator/SKILL.md:108)、[translator/SKILL.md:303](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/operator_runtime_t2a/skills/tilelang2ascend-translator/SKILL.md:303)

### 5.5 固定评测器运行脚本，不等 Agent 重做安装/评测编排

有效源码修改后，Agent 用 Bash 前台调用固定入口：
```bash
bash /opt/workspace/agent_workdir/tools/ascendc_eval_pipeline.sh --op_name {op_name} \
     --impl output/submission/{op_name}_impl.tar.gz --out_dir judge_out
```

它自动打包候选，在隔离 work 下重新注入数据集原 reference/JSON，铺 translator、ops-profiling 两个评测脚本目录，再执行 AscendC AST、编译、注册/加载冒烟、NPU lease 下的精度验证、状态化检测与性能采集，写 metrics 并保留 best。这里“用到了某 skill 目录下的脚本”不等于 Agent 又发生一次 Skill 或 Read 工具调用。

- 自动恢复 reference 与 JSON：[ascendc_eval_pipeline.sh:741](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/operator_runtime_t2a/tools/ascendc_eval_pipeline.sh:741)，对拍前再次恢复 :822。
- 从 canonical 铺评测脚本、AST、build：[ascendc_eval_pipeline.sh:752](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/operator_runtime_t2a/tools/ascendc_eval_pipeline.sh:752)。
- 注册冒烟与 lease 对拍：[ascendc_eval_pipeline.sh:827](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/operator_runtime_t2a/tools/ascendc_eval_pipeline.sh:827)。
- 性能使用 msprof_perf_summary.py --quick --quick-engine grouped，当前 warmup 默认 30、repeats 1：[ascendc_eval_pipeline.sh:913](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/operator_runtime_t2a/tools/ascendc_eval_pipeline.sh:913)。

RL 覆盖区明确禁止 Agent 直接执行 skill 内 AscendC evaluate/build/verify/profiling 脚本或自写测试替代固定入口；上游 skill_script_hook 没启用，Bash 不会自动被它拦截修正。只有 Stop hook 读完成状态，不代跑评测。[CLAUDE.md:694](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/operator_runtime_t2a/CLAUDE.md:694)、[CLAUDE.md:730](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/operator_runtime_t2a/CLAUDE.md:730)

### 5.6 根据真实错误读对应资料再修

当前脚本不只吐错误，还在分类行写下一步文档路径。[ascendc_eval_pipeline.sh:266](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/operator_runtime_t2a/tools/ascendc_eval_pipeline.sh:266)

| 实际分类/现象 | 当前 Read 路由 |
|---|---|
| submission_missing | 检查工程/包布局，脚本明确“不需要读取调试 Skill” |
| AST、编译/链接 | translator；符号/API 错再 docs-search 和实际 API 页 |
| 注册/加载 | runtime-debug + references/kernel_binary_debug.md |
| 崩溃/超时/启动 | crash-debug + crash_workflow；ACL 码再 runtime-debug/error_codes |
| shape/dtype/输出数量 | translator；不可冒称 D 类容差问题 |
| NaN/Inf/全零 | 脚本提示 precision-debug；仍按实际错误分类处理 |
| 正常运行、契约正确、存在数值差异的 D 类 | **先 ops-precision-standard，再 ascendc-precision-debug** |
| benchmark 失败 | ops-profiling；若其实 kernel/ACL 崩溃改 runtime 路线 |
| B 类 INFRA | 不改 kernel，不读取修复文档，停止并保留提交物 |

先读对应文件和实际日志，做一项有证据的代码修改，预算允许才再次评测。不能用“又读了一次 skill”代替修复。CLAUDE 中概括的 A 类大路由不能覆盖脚本对 submission_missing 更具体的提示。

### 5.7 precision-tuning 何时必读，何时只是可选

当前 **D 类硬要求是 precision-standard + precision-debug**。生成器已删除旧的“debug 7 次后 tuning 5 次”的独立计数器，见 [build_claude_md.py:532](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/deploy/ascend_operator/build_claude_md.py:532)、[build_claude_md.py:599](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/deploy/ascend_operator/build_claude_md.py:599)。当前固定入口的 Read 路由也没有自动切换 precision-tuning。

tilelang2ascend-precision-tuning/SKILL.md 仍安装，适合确实需要 DumpTensor 查看中间数据、逐段定位 AscendC 数值问题时补充读取。**不能把“已安装”“出现在 frontmatter”写成所有轨迹必读。** 一旦选择使用这份专题，其中要求先读 api-reference.md，再按 simple Vector 或 Cube+Vector 类型读对应 workflow，做最小抽样并回溯异常点输入；它明确不适用于 TileLang、编译错误、运行错误、卡死，定位后移除 dump。[precision-tuning/SKILL.md:15](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/operator_runtime_t2a/skills/tilelang2ascend-precision-tuning/SKILL.md:15)、[precision-tuning/SKILL.md:85](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/operator_runtime_t2a/skills/tilelang2ascend-precision-tuning/SKILL.md:85)

专题里的“缩小 case”“simple_task/full_task”以及 precision-debug 中的独立清理命令仍可能与 RL 布局冲突。有效顺序必须服从 CLAUDE 覆盖区：不可修改原版用例，不可删工作区 input/output，不可绕开固定入口自己跑对拍。知识方法可以吸收，旧 standalone 操作不能原样照抄。

### 5.8 正确之后的性能与结束

正确性通过只产生 operator_valid，不保证 task_complete。若 geomean speedup 低于目标且仍有优化预算，必须 Read ops-profiling/SKILL.md，再读真实逐 case 指标/perf.log 与源码，实施通用性能修改，再跑同一入口。目标为 1.1x，预算耗尽交历史 best；不能反复测同一源码冒充优化。[CLAUDE.md:585](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/operator_runtime_t2a/CLAUDE.md:585)、[CLAUDE.md:775](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/operator_runtime_t2a/CLAUDE.md:775)

profile 当前 generation_max=15、optimization_max=4，但 Agent 应服从每次输出的 attempt/limit/remaining/next_step，而不是自己数旧 skill 文档里的固定轮数。[profile.t2a.yaml:41](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/deploy/ascend_operator/profile.t2a.yaml:41)

最终 evaluator 配置优先拿 .best.tar.gz，再拿末版 tar.gz，使用同一 canonical judge_command。[profile.t2a.yaml:159](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/deploy/ascend_operator/profile.t2a.yaml:159) 正常的非 task_request profile 渲染明确设置 `refresh_runtime=True`，见 [load_polar_profile.py:392](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/deploy/ascend_operator/tools/load_polar_profile.py:392)，因此该默认路径有独立 judge 刷新的源码依据。手工组装或其它请求路径仍应核对该标记；operator_judge 在启用时要求 fresh_eval_runtime，缺失会报错。[operator_judge.py:156](/Users/chizm/Documents/ascend-project/ProRL-Agent-Server/src/polar/trajectory/evaluator/operator_judge.py:156)

## 6. 讲解 10 条历史轨迹时建议的证据标记

每个算子保留两列：**历史消息实际做了什么**（M 索引、工具、旧路径、返回内容），以及 **当前 t2a 对应知识/流程**（现名、源码链接、何时读取）。不要把后一列补进历史 DAG，当作历史观测。

区分四类箭头最清楚：

- `Read SKILL.md/reference`：知识进入 Agent 上下文。
- `Write/Edit kernel`：Agent 生成或修复代码。
- `Bash fixed pipeline`：Agent 请求环境执行编译/验证/测速。
- `pipeline → bundled script`：确定性环境内部调用，不是模型又选择一次 skill。

如果历史一次 Skill 工具返回了整段说明，证据可以标“历史 Skill 展开”；当前只有 Read，应标“当前本地文档读取”。如果只有 assistant 说“按某 skill 执行”却没有工具证据，应标“assistant 声明”，不能计成真实调用。

尤其不能沿用这几条旧说法：全部算子必走 TileLang；21 个 skill 全部注入全文；每题必精简用例再全量恢复；debug 7 次后 tuning 5 次；每题必须 trace-recorder；Agent 必须 project-init；上游 Hook 会自动拦截脚本；更新本机 cannbot master 会自动更新 RL runtime。它们与本次查到的当前代码不符。

