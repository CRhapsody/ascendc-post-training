# 当前t2a知识读取与固定评测流程

依据当前committed canonical及显式t2a配置的条件流程。本轮启动前一致性检查失败，因此图明确标出需先对齐的分支；后续节点并不表示已启动训练。当前 Skill/Agent 工具禁用，文档由 Read 进入上下文，评测脚本由固定入口在环境内部执行。复杂路径 AST 通过时保留 TileLang 设计，连续 3 份候选失败才回退 reference。末节点为提交和判分，并不表示正确性或性能已经达标；若没有 best，则尝试末版提交物。

```mermaid
%%{init: {"theme":"base","themeVariables":{"fontFamily":"PingFang SC, Arial, sans-serif","fontSize":"18px"},"flowchart":{"curve":"linear","wrappingWidth":340,"nodeSpacing":30,"rankSpacing":35}}}%%
flowchart TB
    profile["显式选择 profile.t2a.yaml<br/>当前代码及 canonical 版本"]
    preflight{"启动前一致性检查通过？"}
    drift["本轮只读检查发现不一致<br/>CLAUDE 与生成器需先对齐"]
    prepare["prepare 铺任务与工程骨架<br/>12 个知识包进入 .claude/skills"]
    classification["Read reference 与病例<br/>记录 single_op / fused / other"]
    route{"根据实际计算结构分路"}
    simple["single_op / simple<br/>translator 内嵌设计与审查"]
    complex["fused 或 other / complex<br/>Read designer，生成 TileLang<br/>最多 3 份 AST 候选"]
    knowledge["Read translator 与所需专题<br/>实现前读 npu-arch 和实际 API"]
    code["主会话 Write/Edit 候选代码<br/>不调用 Skill 或 Agent 工具"]
    pipeline["Bash 请求固定评测入口<br/>自动打包并恢复受保护参考"]
    internals["环境内部调用评测脚本<br/>AST、编译、精度、状态、msprof"]
    metrics["读取 metrics 与日志<br/>按真实结果和剩余预算推进"]
    repair["按反馈 Read 对应知识<br/>数值：standard + precision-debug<br/>性能：ops-profiling"]
    finishSubmission["优先提交 best，缺失则末版<br/>交独立 judge 做最终判分"]
    infra["固定入口判为 B 类 INFRA<br/>停止，保留现有提交物"]
    profile --> preflight
    preflight -->|否| drift
    preflight -->|是，后续为条件流程| prepare
    prepare --> classification
    classification --> route
    route -->|single_op| simple
    route -->|fused / other| complex
    simple --> knowledge
    complex -->|"通过：采用设计<br/>3 次失败：回退 reference"| knowledge
    knowledge --> code
    code --> pipeline
    pipeline --> internals
    internals --> metrics
    metrics -->|"需修复或优化<br/>且仍有预算"| repair
    repair --> code
    metrics -->|允许完成或预算已耗尽| finishSubmission
    metrics -->|环境失败| infra
    classDef context fill:#f1f5f9,stroke:#64748b,color:#0f172a;
    classDef action fill:#dbeafe,stroke:#2563eb,color:#172554;
    classDef limited fill:#fef3c7,stroke:#d97706,color:#78350f;
    classDef fail fill:#fee2e2,stroke:#dc2626,color:#7f1d1d;
    classDef passed fill:#dcfce7,stroke:#16a34a,color:#14532d;
    class profile,preflight,classification,route context;
    class prepare,simple,complex,knowledge,code,pipeline,internals,repair action;
    class metrics,finishSubmission limited;
    class drift,infra fail;
```

[PNG图](agent-t2a-skill-flow.png) · [Mermaid源码](agent-t2a-skill-flow.mmd) · [完整技能调用指南](../docs/agent-skill-call-guide.md)
