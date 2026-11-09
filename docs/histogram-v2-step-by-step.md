# HistogramV2：从 TileLang 设计到 Ascend C 验证，逐步看完整轨迹

HistogramV2 的完整过程包括工程准备、TileLang 设计、Ascend C 落地、精度修复和测速。这里按原记录展开，并说明这些步骤怎样用于 GiGPO 和 GraphGPO。

来源是 [histogram-v2.json](examples/data_samples/agent_trajectories/histogram-v2.json)。共 168 条消息，71 条 assistant 消息，其中 70 条调用工具，共 85 次工具调用。M 编号是 messages 数组的索引，从 0 开始。文末逐条列出全部 assistant 动作。

本文分析历史数据，没有执行其中的命令。历史流程与当前 RL 环境的区别放在后面说明。新增的 Python 语法检查会单独标明，不当作历史运行结果。

## 1. 先看整条流程

轨迹沿用 Phase 0、1、2、3-C、4、5、6、7 的编号。Phase 0 是前置确认，后面才是开发步骤。

| 阶段 | 消息 | 做了什么 | 当时能确认什么 |
|---|---|---|---|
| 0：确认任务 | M2 至 M10 | 读参考算子和 14 例参数 | 知道输入、输出和测试范围 |
| 1：准备工程 | M11 至 M28 | 建目录、复制文件、找模板、读旧 Histc | 工程材料齐了，尚未生成本任务 kernel |
| 2：精简用例 | M29 至 M36 | 备份 14 例，筛成 10 例 | 后续早期评测使用 10 例 |
| 3-C：TileLang 设计 | M37 至 M71 | 写 block、tile、wrapper，修导入检查 | wrapper AST 通过，TileLang 功能未验证 |
| 4：Ascend C 落地和对拍 | M72 至 M133 | 手写 C++ 工程、构建、修接口、修计数 | 构建成功；修复后评测 exit 0，预览没有逐例明细 |
| 5：测速 | M134 至 M156 | 修命令调用和初始化接口，再测速 | 10 例测完，全部慢于 reference |
| 6：全量对拍 | M157 至 M162 | 恢复 14 例，重新验证 | 明确看到 14/14 matched |
| 7：写报告 | M163 至 M167 | 读旧性能文件，写 trace 和总结 | 没有新优化，也没有 14 例性能重测 |

已有的[全阶段 DAG](../figures/agent-histogram-v2-dag.md)对应这条时间线。那是按时间展开的流程图；训练时合并相同状态形成的图是另一件事。

## 2. Phase 0：先确定 HistogramV2 到底算什么

M2 读取参考文件，M3 返回的核心计算是：

~~~python
return torch.histc(
    input.to(torch.float32),
    bins=bins,
    min=min[0].item(),
    max=max[0].item(),
).to(torch.int32)
~~~

输入包含 input 张量、各含一个数的 min/max 张量和整数 bins。输出是长度为 bins 的 int32 计数。input 先转 float32，再统计。

这里最容易漏掉的是范围规则：低于 min 或高于 max 的元素不计数，等于 max 的元素计入最后一个 bin。后面的算法错误正出在这条规则上。

JSON 存的是 shape、dtype、bins 等生成参数，不是已经固定好的输入数组。输入准备函数会生成数据和 min/max，并保证 min 小于 max。M6 查看文件行数和开头，M9 读完整内容，确认有 14 例，覆盖 7 种 dtype、2D/3D shape、bins 2 至 32。M11 将任务路由到需要 TileLang 设计表达的复杂算子路径。

这一步的状态包含参考语义、用例和任务要求。只是读完文件，还没有可运行候选。

M1 已把 10 份 skill 正文放进模型上下文。这条轨迹没有 Skill 工具调用；后面主要是 Read 参考文档和 Bash 调用脚本，不能把它描述成逐个调用了 10 次 Skill。

## 3. Phase 1：建工程，同时参考已有 Histc

这部分并非从零只看 PyTorch 自动生成。

M11 创建 design/block_level、design/tile_level、kernel/op_host、kernel/op_kernel、kernel/utils 和 test 目录。M13 把参考复制为 model.py，把测试参数复制为 HistogramV2.json。

M15 尝试复制 torch_kernel_helper.h，路径不对，M16 报文件不存在。M17 在原目录找不到，M19 扩大搜索范围找到正确位置，M21 重新复制成功。这次失败发生在找工程模板时，不能归为编译失败。

M21 还发现了旧的 6_Histc 工程。M24 读它的 host、device kernel、ops.h 和 Ascend C wrapper。需要注意两点：

- 旧 Histc 的 C++ kernel 已经先跳过范围外元素，再处理 bin 索引；正确边界规则其实已经出现在 Agent 读过的材料中。
- 旧接口和本任务不同：旧实现接收标量 min/max，输出类型随输入；本任务接收 min/max 张量，输出固定 int32，不能直接复制接口。

这一阶段增加了模板和参考知识，还没有本任务的编译或精度证据。读过哪些资料也是状态信息，后续不能只用目录和源码判断两个状态相同。

## 4. Phase 2：把 14 例暂时缩成 10 例

M29 先备份原 JSON。M31 尝试写精简集，却被工具拒绝：输出目录中的文件还没读过。M33 补读这个副本，M35 再写，M36 才确认成功。之前读过 benchmark 原路径，不等于读过复制后的文件。

保留的原 case_id 是：

~~~text
0, 1, 2, 3, 4, 5, 6, 8, 10, 12
~~~

删去的是 7、9、11、13。10 例仍覆盖 7 种 dtype 和原有维度类别，但不再覆盖所有原始组合。

从此以后，代码没变也不能把"14 例待验"和"10 例待验"看成同一状态。用例版本改变了，评测结果的覆盖范围也改变了。后面性能报告的 index 是子集索引，例如 index 7 对应原 case_id 8。

这是历史流程。当前 RL 固定评测不允许 Agent 自行缩减正式用例，不能把精简病例当作 GraphGPO 的合法捷径。

## 5. Phase 3-C：TileLang 设计，分三个文件看

### 5.1 先读设计材料

M37、M41 读旧 Histc 的 block-level、tile-level 和 wrapper；M44 读 BlockLevelDesign.md 与 TileLangAscendProgrammingGuide.md。

旧 TileLang 方案直接 clamp 范围外索引，与刚才读过的旧 C++ 过滤规则不同。新设计延续了 TileLang 这套错误处理。这个细节说明：参考材料之间可能不一致，不能把"参考过已有实现"当成算法正确。

当前仓库中的[BlockLevelDesign.md](../cannbot-skills/ops/tilelang-op-design/references/BlockLevelDesign.md)可帮助理解逻辑 block 和实际 core。历史文档名称和路径以 M45、M46 的返回内容为准。

### 5.2 M47：block-level 决定任务怎么分

Agent 写出 design/block_level/histogram_v2.py。关键参数是：

~~~python
num_physical_cores = 20
block_size = 2048
num_blocks = (num_elements + block_size - 1) // block_size
used_core_num = min(num_physical_cores, num_blocks)
tasks_per_core = (num_blocks + used_core_num - 1) // used_core_num
~~~

计划把输入每 2048 个元素分成一个逻辑 block，最多安排 20 个 core。每个 core 可以处理多个 block。20 是代码里写死的设计参数，这段没有硬件查询证明它来自实际 core 数。

每个 block 在核内缓冲 UB 中统计 local_hist[bins]，写入全局内存 GM 的 workspace[num_blocks, bins]。最后由 block 0 把各块计数加起来，生成 out[bins]。

block-level 主要表达任务分配和数据流，计算位置还是 TODO。它甚至有一处 if 下只有注释、没有语句。本次只用标准库 ast.parse 检查 M47 写出的文本，就得到 IndentationError；这不是历史运行报错。后续 wrapper 加载的是 tile-level，AST 通过也没有验证这个 block 文件。

### 5.3 M49：tile-level 决定块内怎么计算

Agent 写出 design/tile_level/histogram_v2.py，沿用 2048 元素 block，并设：

~~~python
elem_tile = 256
bin_tile = 64
~~~

具体步骤如下：

1. 给当前 block 分配 local_hist[bins]，清零。
2. 每次把最多 256 个输入元素搬到 UB。
3. 算 shifted = x - min，再乘 scale = bins / (max - min)。
4. 用 CAST_FLOOR 得到 int32 bin 索引。
5. 逐元素把计数加到 local_hist。
6. 把这个 block 的直方图写入 workspace。
7. block 0 按最多 64 个 bin 一组，读取所有 block 的结果，累加并写到输出。

索引计算使用 TileLang 的 tile 操作，但计数仍是逐元素循环。M49 的实际逻辑是：

~~~python
for i in T.serial(tile_len):
    idx = bin_indices[i]
    if idx < 0:
        idx = 0
    elif idx >= bins:
        idx = bins - 1
    local_hist[idx] = local_hist[idx] + 1
~~~

范围外元素被夹到首尾 bin，已经偏离参考语义。这个错误起于设计阶段，后来被带进了 C++。

另一个未验证的问题是归约顺序：bx == 0 写完自己的第一个 block，就读取全部 workspace；代码没有展示所有 block 完成后的独立归约阶段。若一个 core 分到多个任务，连 core 0 自己后续的 block 都还没算。不能仅凭自动同步配置就认为这套多核设计已经正确。

本次对 M49 做 Python AST 解析成功，只说明语法可解析，没有运行 TileLang 编译器或 NPU，不能证明 API、同步和结果正确。

### 5.4 M51：wrapper 把 Python 输入接到 TileLang kernel

model_new_tilelang.py 的 forward 做这些事：

- 从 min/max 张量取标量。
- 把 input 转 float32，展平成连续的一维张量。
- 用元素数、bins、min/max 调 builder。
- 分配 out[bins] 和 workspace[num_blocks, bins]，调用 kernel 后返回 out。

因此，7 种输入 dtype 在这条 wrapper 路径中都转为 float32。它不是把整个 input 搬去 CPU 计算；代码中的 float() 是类型转换。

初版通过 importlib 动态加载设计文件。它写了 builder 和 kernel 调用，问题是检查器识别不了这种导入方式。

### 5.5 M53 至 M71：三次 AST 调用分别发生了什么

| 调用 | 结果 | 改了什么 |
|---|---|---|
| M53，反馈 M55 | validator 路径不存在，exit 2 | M56、M58 搜索正确脚本位置 |
| M60，反馈 M62 | 没识别到 design.tile_level 的导入，exit 1 | M63 读检查器，M65 读 wrapper，M67 改导入 |
| M69，反馈 M71 | wrapper AST 通过，exit 0 | 没再修改算法 |

M67 改成标准导入：

~~~python
from design.tile_level.histogram_v2 import histogram_v2 as _histogram_v2_builder
~~~

检查器在 wrapper 中识别到 builder 和 kernel 调用，并确认 forward 中没有它禁止的 PyTorch 操作或逐元素 Python for。它没有递归编译 tile-level，所以 M49 的逐元素 T.serial 计数仍然存在。

M62 虽然打印四项 FAIL，源码是在导入检查失败后提前返回，不能理解成独立发现四类算法问题。

工具返回的 hook-noop 也不是最终结果。真正 exit_code/stdout/stderr 在随后 user-role 的 system-reminder 中，例如 M55、M62、M71；分析轨迹时要把它们关联到前一次 Bash。

### 5.6 M72：明确跳过 TileLang 功能验证

Agent 按当时指南把 TileLang 当设计表达，跳过功能验证，开始读 Ascend C 转译资料。

到这里，准确状态是：设计文件已经写出，wrapper AST 通过，TileLang 数学语义和多核执行没有运行验证。不能把它缩写成"TileLang 正确性通过"。

## 6. Phase 4：转成 Ascend C 工程，再分层验证

### 6.1 这里的"转译"实际是怎么做的

M72 读 API 映射和 dsl2Ascendc；M75、M78 读旧 Histc 的 setup.py 与 CMake；M87 查 TBuf/InitBuffer/AllocTensor，M90 读 GELU kernel。

随后 Agent 用 Write 生成 C++ 和工程文件。轨迹没有调用一个编译器，把 TileLang 文件自动导出成这些 C++。后续 CANN 编译生成 launch header，是 Ascend C 工程的编译结果。

当前可读的[API 映射](../cannbot-skills/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/TileLang-AscendC-API-Mapping.md)和[dsl2Ascendc](../cannbot-skills/plugins-community/tilelang2ascendc-ops-generator/skills/tilelang2ascend-translator/references/dsl2Ascendc.md)有助于理解这一步。

而且 Agent 改了设计：

| 项目 | TileLang M47/M49 | Ascend C M92/M94 |
|---|---|---|
| core 数 | 最多 20 | 固定 1 |
| 输入划分 | 2048 元素 block，块内 256 元素 tile | 单核遍历全部输入，1024 元素 tile |
| 局部计数 | 每个 block 一个 histogram | 一份 histogram 持续累积 |
| workspace | 保存各 block 的计数 | 没有这条 workspace 路径 |
| 最后归约 | block 0 合并各块 | 没有跨块归约 |
| 范围处理 | clamp 越界索引 | 初版也 clamp，M129 才修正 |

因此，这一步同时发生了代码生成和算法结构调整，不能当成保持并行策略的逐行翻译。

### 6.2 M80、M92：device kernel 在 NPU 上做什么

M80 先写一个初稿，随后查 API 和 GELU 示例，M92 完整重写。中间没有初稿编译失败的日志，所以应写"读资料后主动重写"，不是"编译失败后修复"。

M92 的处理顺序是：

1. 从 tiling 中读输入长度、bins、min/max。
2. 只有 blockIdx == 0 工作。
3. 创建输入队列与 shifted、scaled、binIndices、localHist 的 UB 缓冲。
4. 清零 localHist，按最多 1024 元素循环。
5. CopyIn 搬输入；Compute 用 Adds、Muls、Cast 算 bin 索引。
6. 在 UB 中逐个 GetValue/SetValue 更新计数。
7. 全部输入处理完，再把 localHist 写回输出。

向量部分可摘为：

~~~cpp
Adds(shifted, xLocal, -tiling_.minVal, curTileLength);
Muls(scaled, shifted, scale, curTileLength);
Cast(binIndices, scaled, RoundMode::CAST_FLOOR, curTileLength);
~~~

前面能向量计算索引，不代表后面的 histogram 累加也向量化了。队列配置 BUFFER_NUM = 2，也不能直接证明有有效双缓冲重叠，源码按 CopyIn、Compute 顺序调用。

这份代码的尾块搬运把长度向上对齐到 8 个 float，DataCopyPad 的 isPad=false，validLen 只限制计数循环。这里没有额外边界证据，不能把它称作已经验证的完整尾块处理。

### 6.3 M94：host 准备输入、参数并启动 kernel

host 不是 kernel 本体。它从 Python 调用接收张量，准备 NPU kernel 需要的数据：

- 检查 bins 大于 0、min 小于 max。
- 把 min/max 单元素张量取到 CPU 得到标量。
- 把 input 转 float32、连续化、展平。
- 分配 int32 输出，并设 usedCoreNum = 1。
- 打包 tiling，复制到设备，调用 EXEC_KERNEL_CMD。

这例没有单独生成 tiling.h。host 用一个 10 元素 int64 张量存参数：slot 0 是元素数，slot 1 是 bins，slot 2/3 存 min/max 的 float 位内容。device 的 CopyTiling 按字段解码，不是直接复制一个紧凑 C++ struct。

所以真正的调用还包括类型转换、清零、参数和数据准备。wrapper AST 通过，不能证明整条调用只有一个纯 kernel，也不能说明测速已覆盖所有 host 开销。

### 6.4 M96 至 M102：把 device、host 和 Python 接起来

| 文件 | 写入消息 | 作用 |
|---|---:|---|
| kernel/op_kernel/histogram_v2.cpp | M80、M92 | device 计算 |
| kernel/op_host/histogram_v2.cpp | M94 | 参数准备、launch、Torch 注册和扩展入口 |
| kernel/ops.h | M96 | 声明 host 接口 |
| kernel/CMakeLists.txt | M98 | 编译 device 静态库、host 和 Python 扩展 |
| kernel/setup.py | M100 | 构建/打包扩展 |
| model_new_ascendc.py | M102 | 加载扩展，提供 ModelNew.forward |

wrapper 的 forward 实际调用：

~~~python
return torch.ops.npu.histogram_v2(input, min_t, max_t, bins)
~~~

这个 Torch 注册项指向 host 函数，host 再启动 device kernel。CANN 编译生成的 aclrtlaunch_histogram_v2.h 负责 launch 接线。写完这些文件，只说明源码齐了。

### 6.5 M104 至 M111：AST 通过，工程构建也成功，却没开始对拍

M104 调 Ascend C wrapper AST 检查，M106 通过。随后 M107 调 evaluate_ascendc.sh，依次清理构建、CMake/make、打 wheel、安装，再调用 verifier。

M110 读完整日志，M111 显示：

~~~text
[100%] Built target pybind11_lib
Successfully installed histogram-v2-0.1.0
TypeError: Model.__init__() takes 1 positional argument but 11 were given
Result: fail
~~~

因此这次不是编译失败，也不是计数不匹配，问题出在 Python 模型构造。

参考 get_init_inputs() 返回按 10 个用例组织的列表：

~~~python
[[], [], [], [], [], [], [], [], [], []]
~~~

但 verifier 把整个列表当一组构造参数，用 Model(*init_inputs)。于是传入 10 个参数，加 self 后就是报错里的 11。本例 Model 原本无需构造参数。

### 6.6 M112 至 M126：先修构造接口，再暴露计数错误

M112、M114 读评测入口和 verifier，M116 读 candidate wrapper。verifier 优先取 candidate 的初始化参数，再用这份参数同时构造 reference 和 candidate。

M118 在 candidate 增加：

~~~python
def get_init_inputs():
    return []
~~~

这让双方恢复无参构造。它不是 kernel 初始化修复，也没有修改直方图公式。

M120 重测。M123 读日志的 offset 太大，只得到"文件没有那么多行"；M125 改 offset，M126 才看到 10 例都不匹配，每例有两个 bin 偏大。

现在错误层才从接口变成了数学语义。反复修改构建参数不会解决这个问题。

### 6.7 M127 至 M133：修范围规则，重测返回成功

M127 读 kernel。原循环把负索引改 0，把过大索引改 bins-1，导致范围外输入也被统计。

M129 的实际修复是：

~~~cpp
int32_t idx = binIndices.GetValue(i);
if (idx < 0) {
    continue;
}
if (idx >= static_cast<int32_t>(bins)) {
    float val = xLocal.GetValue(i);
    if (val > tiling_.maxVal) {
        continue;
    }
    idx = static_cast<int32_t>(bins) - 1;
}
int32_t curVal = localHist.GetValue(idx);
localHist.SetValue(idx, curVal + 1);
~~~

负索引跳过；索引超过末 bin 时，再看原值，大于 max 跳过，等于 max 保留到末 bin。这是边界语义修复，不是调整误差容忍度。

用构造输入 [-1, 0, 0.5, 1, 2]、范围 [0, 1]、2 个 bin 就能看出差别：正确计数是 [1, 2]，旧 clamp 得到 [2, 3]。这不是日志里的真实随机数组。

M131 重测，M133 给出 exit 0，但展示的是截断预览，没有逐例 10/10 明细。M134 的"所有 case 匹配"是 Agent 自述。后面 M162 的 14 条 matched 才是完整的逐例证据。

还有一个设计一致性问题：M129 只修了 Ascend C，TileLang M49 仍保留错误 clamp。最终 C++ 已不完全对应原 DSL。

## 7. Phase 5：测速之前，又修了两层问题

| 动作与结果 | 发生了什么 | 状态变化 |
|---|---|---|
| M134，M136 返回 | hook 把 ASCEND_RT_VISIBLE_DEVICES=11 当成命令名，启动失败 | 没有性能证据 |
| M137，M139 返回 | 加另一个环境赋值前缀，仍被当成命令名 | 仍未进入测速 |
| M140，M141 返回 | 这次进入 Python，但两端模型构造都报 11 个参数 | 出现的是错误报告，不是有效测速 |
| M142、M144、M146 | 读 performance.py 和 reference helper | 知道测速参数取自 reference |
| M148 | 把 model.py 的 get_init_inputs() 改为 [] | reference 辅助接口变了，kernel 没变 |
| M150，M152 返回 | export 被 hook 当命令名，启动又失败 | helper 已修，但仍无性能结果 |
| M153、M155，M156 返回 | env 前缀运行，再读完整结果尾部 | 真正得到 10 例测速 |

这里有两点不能混在一起。

第一，M118 修 candidate helper 能解决旧 verifier 的构造，但旧 performance.py 直接取 reference helper，所以测速仍报同类错误。M148 改的是 reference 的初始化辅助函数，不是一次 kernel 优化。没看到它修改 reference forward，也没看到后面恢复这个 helper。

第二，hook 启动失败与已进入 Python 的构造失败是不同观察。不能因为都包含"failed"，就把它们放进同一个状态。

M153 完成的测速设置是 NPU、warmup 5、repeats 50、seed 0。M164 保存的数字如下：

| 子集 index | 原 case_id | reference ms | Ascend C ms | 加速比 |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0.0210 | 0.0256 | 0.8203 |
| 1 | 1 | 0.0230 | 0.0869 | 0.2647 |
| 2 | 2 | 0.0207 | 0.1060 | 0.1953 |
| 3 | 3 | 0.7593 | 47.0286 | 0.0161 |
| 4 | 4 | 0.1501 | 6.5082 | 0.0231 |
| 5 | 5 | 0.0559 | 1.5863 | 0.0352 |
| 6 | 6 | 0.0210 | 0.0276 | 0.7609 |
| 7 | 8 | 0.0211 | 0.0285 | 0.7404 |
| 8 | 10 | 0.0212 | 0.0258 | 0.8217 |
| 9 | 12 | 0.0211 | 0.0266 | 0.7932 |

每例都慢于 reference。最大的保留 shape [64, 128, 256]、float32、bins 8，对应 index 3，候选用了 47.0286 ms。

报告的 overall 0.4471 是"先求每例 reference_ms / candidate_ms，再取算术平均"，不是几何平均，也不是平均延迟之比。按 M164 复算，两端平均延迟是 0.11144 ms 和 5.54501 ms，均值之比约 0.0201。M165 摘要写的 reference 平均 0.113 ms 不准确。

日志还有 profiler 解析警告，不能据此断言结果全无效，也不能声称已覆盖所有 CPU、host 和传输开销。这里只沿用历史报告口径。

从源码可提出单核、标量计数和 host 准备等性能问题，但需要进一步实验定位。M157/M167 将缓慢说成直方图的"固有特性"，超出了证据；当前实现慢，不等于这个算子必然慢。

## 8. Phase 6、7：恢复全量、验证、写报告

M157 把备份恢复为 14 例，M158 确认。恢复的是 JSON，不是 M148 改过的 reference helper。

M159 重新调用完整评测，M161 从日志提取结果。M162 明确显示 case[0] 至 case[13] 全部 matched，以及 Result: pass。这是在当时工作 reference 下的全量精度证据。

随后 M163 读 preformance.json，M164 返回旧的 10 例性能报告。文件读取发生在恢复 14 例之后，不会让旧结果自动变成 14 例测速。这条轨迹没有全量性能重测。

M165 写 trace.md，M166 确认写入，M167 声明流程完成。这些报告动作没有增加新的精度或性能证据，也没有出现测速后的 kernel 优化。

最终状态应写为：历史流程完成，14 例精度通过，10 例性能全部慢于参考，没有证明达到我们研究中"正确且快"的目标。

## 9. 这些步骤怎样用于 GiGPO 和 GraphGPO

### 9.1 状态要具体到文件、证据和已知信息

阶段标签可以帮助找到候选状态，但不能直接用作合并键。例如，同是 Phase 4，下面四个状态完全不同：

| 检查点 | 当时已知什么 | 后续要解决什么 |
|---|---|---|
| M106 后 | wrapper AST 通过，未构建运行 | 工程能否构建、模型能否运行 |
| M111 后 | 构建安装成功，构造参数不兼容 | 接口适配，不是改计数公式 |
| M126 后 | 模型可运行，10 例首尾计数偏大 | 范围过滤的数学语义 |
| M133 后 | 修复评测 exit 0，逐例预览不完整 | 读取证据，后续测速与全量验证 |

读日志也会改变状态：M120 刚发评测时、收到 M124 后仍没读到所需诊断时、收到 M126 后知道具体 mismatch 时，代码相同，Agent 掌握的信息却不同。发出 Read 不等于已经获得它的返回，不能把未来观察写进动作前状态。M148 改 reference、M157 改病例集，则连评测定义都变了。

建议每次 assistant 决策保存：代码与构建状态、reference/病例/评测器版本、最新结果绑定的代码版本、已知诊断和文档、剩余预算、策略版本。工具输出和 hook 的 system-reminder 是观察，不能当作模型生成动作参与损失。

这条数据的 71 条 assistant 消息可作为决策边界索引。一条 assistant 消息可以包含多个工具调用，例如 M2 包含 3 次，但这份记录不能证明它们实际并行执行。阶段可以聚合展示，训练仍要对齐真实生成 token。

### 9.2 哪些地方适合比较不同动作

下表是新实验的分支设计，不是历史中已经采样到的同状态组。每组要从相同快照和有效上下文重新采样。

| 固定起点 | 可以比较什么 | 必须保留的验证 |
|---|---|---|
| 已理解 reference，准备写 tile 设计 | 是否正确表达范围过滤；是否保留可验证的分块与归约 | 最终 Ascend C 精度与性能，不能只评注释 |
| 已有设计，准备写 wrapper | 标准导入与不被检查器识别的导入写法 | wrapper 检查；不能把它等同 kernel 正确性 |
| 已读 API 映射，准备生成 C++ | 单核计数与其它实际实现方案 | 构建、对拍、统一测速；多核只是待验证方案 |
| 已确认范围外计数错误 | 修过滤、改无关 tile、尝试无修改重测但被规则拒绝 | 绑定修改后的候选重新对拍 |
| 正确但慢，准备优化 | 改实际计算或调度，还是只调整命令/写总结 | 全量精度和合法的统一测速 |

当前 runtime 禁止源码未变化时重复评测。表中的无修改重测只作为被拒绝的无效动作记录，不能产生一份新的 NPU 评测证据。

接口与环境问题最好在采样前固定。不能让算法学成遇到 benchmark 兼容问题就修改 reference。失败层应分别记录：找文件、工具启动、构造接口、编译、精度、性能；也不能仅凭报错自动标为当前环境的基础设施故障。

### 9.3 为什么不能给这条历史轨迹逐步填出优势数字

它是一条 SFT 记录，没有当前策略的同任务 rollout group、逐 token 行为 logprob 和完整状态快照。我们可以标注每一步改变了什么，但不能据此声称算出了 on-policy GiGPO 优势。

若在新采样组中使用 GiGPO，每条真实动作比较的是从该步开始的实际折扣回报。AST 修复、接口修复、范围修复都可能因为后续性能失败而拿不到理想局部信用，不能按阶段手工各填一个正奖励再称为原方法。

GraphGPO 需要图中有可信成功终点。如果只要求正确性，M162 是一个精度通过状态；如果目标是全量正确且达到速度门槛，这条记录没有终点。10 例全慢，14 例性能未知，不能把 M167 的完成声明设成成功 G，再倒推出各步到 G 的距离。严格目标缺失时图局部项归零，现有终局质量和速度奖励仍可保留。

前面[三条短轨迹的优势计算](ascendc-gigpo-graphgpo-research.md)只演示范围修复这一段，并假设另一条合法轨迹已到达正确且快的 G。它不代表这里的 71 次动作都已有可计算的图距离。

### 9.4 想缩短链路，需要验证哪条边真能省

可以提出这些待验证假设：初次就写标准 import，减少 wrapper 修复；设计时就保留 reference 的范围过滤，减少后面的对拍返工；用已经验证的工程模板，减少文件路径搜索。它们都要在新 rollout 中检查，不能从历史总结补出未观察的边。

更不能直接把 M134 跳到 M153：中间 M148 修了测速初始化接口，单换 env 前缀是否成功没有证据。也不能省掉恢复全量，却继续声称 14 例通过；读旧性能报告同样不能替代新测速。

当前 t2a 使用固定评测入口，reference 和正式用例受保护，不允许照搬历史的用例精简、reference 修改、设备环境变量设置或直接调用旧脚本。历史 skill_script_hook 也不能当作当前的预期流程。[当前运行规则](../ProRL-Agent-Server/operator_runtime_t2a/CLAUDE.md)、[固定评测入口](../ProRL-Agent-Server/operator_runtime_t2a/tools/ascendc_eval_pipeline.sh)

当前 RL 以 geomean_speedup 为主指标，历史报告取逐例比值的算术平均，不能把两种协议的数字拼成一条训练性能曲线。

因此，新实验应先固定合法任务与工具链，完整保留真实采样轨迹，利用组内图距离改进优势。发现的短路径用于信用分配；要做 SFT 示例仍需重放验证，不能冒充旧策略采样记录。验证收益时同时看精度、速度、token 和 NPU 评测成本。

## 10. 全部 assistant 动作索引

下表覆盖 71 条 assistant 消息。多工具动作合在同一行；回包索引包含 hook 附加反馈。Write/Edit 成功只证明文件变了，Read 成功只证明获得内容，都不能自动标成验证通过。

| 动作 | 工具 | 做了什么 | 可见结果 |
|---|---|---|---|
| M2 | Read、Bash×2 | 读参考，检查输出目录和相关文件 | M3至M5：参考正文、目录与文件 |
| M6 | Bash×2 | 统计行数、预览用例 | M7/M8：开头与14例行数 |
| M9 | Read | 读完整用例 | M10：14例参数 |
| M11 | Bash | 设置当次设备变量、建目录 | M12：目录创建 |
| M13 | Bash | 复制 reference 和 JSON | M14：复制完成 |
| M15 | Bash | 复制 helper，路径不对 | M16：文件不存在 |
| M17 | Glob | 在原范围找 helper | M18：没找到 |
| M19 | Glob | 扩大范围查 helper | M20：找到正确位置 |
| M21 | Bash×2 | 复制 helper，查看旧 Histc | M22/M23：复制成功、工程文件列表 |
| M24 | Read×4 | 读 Histc host、kernel、ops.h、wrapper | M25至M28：返回源码 |
| M29 | Bash | 备份14例JSON | M30：备份成功 |
| M31 | Write | 写10例子集，尚未读副本 | M32：工具拒绝 |
| M33 | Read | 补读输出副本 | M34：14例 |
| M35 | Write | 再写10例子集 | M36：更新成功 |
| M37 | Read×2、Glob | 读旧block/model，列设计文件 | M38至M40：参考内容 |
| M41 | Read×2 | 读旧tile和TileLang wrapper | M42/M43：参考内容 |
| M44 | Read×2 | 读设计指南 | M45/M46：指南正文 |
| M47 | Write | 写block-level骨架 | M48：文件创建 |
| M49 | Write | 写tile-level设计 | M50：文件创建 |
| M51 | Write | 写动态导入wrapper | M52：文件创建 |
| M53 | Bash | 调TileLang检查，路径错误 | M54/M55：hook预览、exit 2 |
| M56 | Glob | 在原范围找检查器 | M57：没找到 |
| M58 | Glob | 改搜索范围 | M59：找到检查器 |
| M60 | Bash | 调正确检查器 | M61/M62：导入模式FAIL |
| M63 | Read | 读检查器源码 | M64：识别规则 |
| M65 | Read | 读wrapper | M66：动态导入内容 |
| M67 | Write | 改标准导入wrapper | M68：更新成功 |
| M69 | Bash | 重做TileLang AST检查 | M70/M71：wrapper PASS |
| M72 | Read×2 | 跳过TileLang功能验证，读转译资料 | M73/M74：API映射和转译说明 |
| M75 | Read、Bash | 读旧setup，列kernel目录 | M76/M77：构建参考 |
| M78 | Read | 读旧CMake | M79：构建参考 |
| M80 | Write | 写第一版device kernel | M81：文件创建，没有构建结果 |
| M82 | Glob×2 | 找TBuf、TPipe文档 | M83/M84：没找到 |
| M85 | Bash | 列转译reference目录 | M86：文件名列表 |
| M87 | Grep、Glob | 查buffer API片段和GELU示例 | M88/M89：片段与路径 |
| M90 | Read | 读GELU kernel | M91：队列和缓冲参考 |
| M92 | Write | 重写device kernel | M93：更新成功 |
| M94 | Write | 写host和注册代码 | M95：文件创建 |
| M96 | Write | 写ops.h | M97：文件创建 |
| M98 | Write | 写CMake | M99：文件创建 |
| M100 | Write | 写setup.py | M101：文件创建 |
| M102 | Write | 写Ascend C wrapper | M103：文件创建 |
| M104 | Bash | 做Ascend C wrapper AST检查 | M105/M106：PASS |
| M107 | Bash | 构建、安装、运行verifier | M108/M109：构建预览 |
| M110 | Read | 读第一次完整评测日志 | M111：构建成功，模型构造失败 |
| M112 | Read | 读evaluate入口 | M113：构建和验证顺序 |
| M114 | Read | 读verifier | M115：初始化参数取法 |
| M116 | Read | 读candidate wrapper | M117：待适配接口 |
| M118 | Edit | 增加candidate get_init_inputs | M119：更新成功 |
| M120 | Bash | 再构建对拍 | M121/M122：日志预览 |
| M123 | Read | 用过大offset读日志 | M124：没有所需结果 |
| M125 | Read | 换正确offset读日志 | M126：10例都计数不匹配 |
| M127 | Read | 读kernel确认clamp | M128：当前实现 |
| M129 | Edit | 修范围外计数 | M130：更新成功 |
| M131 | Bash | 修复后重测 | M132/M133：exit 0，明细未展开 |
| M134 | Bash | 调性能脚本，赋值前缀不被hook识别 | M135/M136：启动失败 |
| M137 | Bash | 加另一个赋值前缀重试 | M138/M139：启动失败 |
| M140 | Bash | export方式进入Python | M141：两端构造参数错误 |
| M142 | Read | 读performance.py构造部分 | M143：参数展开规则 |
| M144 | Read | 读performance.py参数来源 | M145：取reference初始化参数 |
| M146 | Read | 读reference helper | M147：按用例分组的初始化列表 |
| M148 | Edit | 将reference helper改成[] | M149：更新成功，kernel没变 |
| M150 | Bash | 再调性能脚本 | M151/M152：export被hook错误解析 |
| M153 | Bash | env前缀重跑 | M154：持久日志预览 |
| M155 | Bash | 读测速输出尾部 | M156：两端OK，10例性能 |
| M157 | Bash | 恢复原14例JSON | M158：恢复成功 |
| M159 | Bash | 做全量验证 | M160：构建和运行预览 |
| M161 | Bash | 提取完整逐例结果 | M162：14/14 matched、pass |
| M163 | Read | 读旧性能JSON | M164：仍是10例，overall 0.4471 |
| M165 | Write | 写trace.md | M166：文件写成功，无新测量 |
| M167 | 无工具 | 输出完成总结 | 没有新增验证证据 |
