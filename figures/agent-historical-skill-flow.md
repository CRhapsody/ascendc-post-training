# 历史十条轨迹的技能知识与脚本调用流程

基于10份SFT样本中的真实消息。M1的正文预置不计为Skill工具调用；后续文档读取、Skill展开和脚本执行分别表示不同动作。图内循环是调用流程概括，每条历史执行DAG中的重试仍按不同消息节点展开。

```mermaid
%%{init: {"theme":"base","themeVariables":{"fontFamily":"Arial, sans-serif","fontSize":"18px"},"flowchart":{"curve":"linear","wrappingWidth":340,"nodeSpacing":30,"rankSpacing":35}}}%%
flowchart TB
    system["M0 开发者工作流<br/>历史 Phase 0–7 说明"]
    context["M1 已内嵌 10 份技能正文<br/>这 10 条样本逐块相同"]
    choice["读取 reference 与病例<br/>按历史规则选择设计路径"]
    simple["AddCustom、CosMath<br/>design-doc 与 code-gen 资料"]
    complex["其余 8 条历史轨迹<br/>TileLang 与 translator 资料"]
    repeat["按需再次 Skill 展开<br/>共 21 次；并非每阶段都有"]
    readrefs["Read/Grep reference、模板或脚本<br/>没有直接 Read SKILL.md"]
    implementation["当前开发者写设计与 AscendC<br/>Skill 正文提供流程和约束"]
    run["Bash 调用 AST / evaluate 脚本<br/>部分命令被历史 hook 接管"]
    feedback["读取真实报错并修改<br/>编译、接口、数值分别处理"]
    performance["performance.py 或自写计时<br/>具体做法按每条轨迹核实"]
    finish["恢复或复用病例验证<br/>写 trace 与最终总结"]
    system --> context
    context --> choice
    choice --> simple
    choice --> complex
    simple --> readrefs
    complex --> readrefs
    context -.按需新增加载.-> repeat
    repeat -.补充上下文.-> readrefs
    readrefs --> implementation
    implementation --> run
    run --> feedback
    feedback -->|仍需修复| implementation
    feedback -->|通过当前精度检查| performance
    performance --> finish
    classDef context fill:#f1f5f9,stroke:#64748b,color:#0f172a;
    classDef action fill:#dbeafe,stroke:#2563eb,color:#172554;
    classDef limited fill:#fef3c7,stroke:#d97706,color:#78350f;
    class system,context,choice context;
    class simple,complex,repeat,readrefs,implementation,run,feedback action;
    class performance,finish limited;
```

[PNG图](agent-historical-skill-flow.png) · [Mermaid源码](agent-historical-skill-flow.mmd) · [完整技能调用指南](../docs/agent-skill-call-guide.md)
