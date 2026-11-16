# 病理切片名单划分与交付

FastAPI + SQLite 的病理研究切片名单工具：网页上传/追加 CSV，按全库精确划分训练集与盲测集，
并把训练名单交付给一个**独立**的接收服务，核实回执后区分「终态接纳 / 终态拒收 / 状态未明」。

## 快速开始（Docker Compose）

```bash
docker compose up --build
```

* 网页应用：<http://localhost:8000>
* 独立接收服务：<http://localhost:8001>

两个服务各有独立的数据卷（`app-data`、`receiver-data`）与各自的 SQLite 文件，
所以“已交付”是外部事实，而不是应用自己的一个标志位。

网页上的操作顺序：上传 CSV → 填测试集良性/恶性张数 → 求解并发布名单 →
下载测试/训练名单 → 交付训练名单 → 在交付表里「核实」或「重试」。
失败、超时或重启后都可以用页面上的「核实 / 重试」按钮继续，不会重复接纳。

## CSV 格式

```csv
slide_id,patient_id,repeat_group,malignancy
S001,P001,G1,benign
S002,P002,G1,malignant
S003,P001,,benign
```

* 表头可省略（无表头时接受 3 列 `slide,patient,malignancy` 或 4 列）。
* `repeat_group` 可空；空表示“没有重复图组”，不会与任何切片相连。
* `malignancy` 只接受 `benign` / `malignant`（也识别 良/恶）。
* 总库最多 **100** 条。**整批校验**：任一行非法则整批不写入；重复切片号会被跳过而不是覆盖。

## 规则

**隔离**：切片按「同患者号」或「同一非空重复图组」传递连通。一个连通组必须整个进同一集合，
因此组内任意切片被禁入测试，整组都只能进训练。

**精确划分**：给定测试集良性 `b` 张、恶性 `m` 张，其余全部进训练集。
求解器对完整连通组做 0/1 精确子集和（动态规划，状态空间 ≤ 101×101），
不做贪心或随机抽样。无解时返回具体原因（配额超过全库 / 超过禁入后可用数 / 连通组组合
无法精确匹配，并给出最接近的可行组合）。

**交付与禁入**

| 交付状态 | 对测试集的影响 |
| --- | --- |
| 待核实（pending） / 查无记录 / 超时 / 非 2xx | 相关切片**及其当前连通组**暂禁入测试 |
| 终态接纳（accepted） | 相关切片**永久**禁入测试 |
| 终态拒收（rejected） | 只解除**该笔**交付带来的禁入；切片重新变为“未明交付”，仍需显式交付才能进测试 |

禁入集合每次都从交付表重新推导，不做缓存，所以崩溃或重启不会丢失限制；
拒收也不会顺带清掉其它交付的限制。

**名单重判**：每次补传、交付状态变化（以及启动时）都会用当前库内容和当前交付状态重判所有旧名单。
未纳入名单的新切片若与某测试切片属于同一连通组，或名单内切片被禁入，该名单即
“隔离失效”，停止下载并显示原因；重新划分出合格名单后恢复可下载。

**交付协议 / 崩溃安全**

1. 发送前先把「待核实」记录和这份训练名单的冻结副本写入数据库（同一事务）。
2. 再向接收服务 POST。接收服务按交付号写入一次终态，回执号固定；
   同一交付号重复提交只返回原回执，不会二次接纳。
3. 超时 / 连接中断 / 5xx / 404 一律记为“状态未明”，切片继续暂禁入测试；
   之后可「核实」（GET 交付号，核对校验和）或「重试」（用同一交付号重发）。
4. 每次启动会重判所有名单并重新核实所有待核实交付，不会自动解除任何限制。
5. 求解在写事务之外进行；发布时在事务内比对库指纹与交付代次，期间有任何补传/交付变化
   则丢弃结果、不发布（“过期计算不得发布”）。名单与完整成员表在**同一事务**写入，
   写入失败不会留下半份名单。

## API（便于自查/脚本）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/library` | 总库与容量 |
| GET | `/api/restrictions` | accepted / uncertain / barred 及原因 |
| GET | `/api/plans` | 名单（含状态、原因、交付） |
| GET | `/plans/{id}/download/{test\|train}` | 下载 CSV；失效时 409 + 原因 |
| POST | `/plans` | 表单 `benign_count`,`malignant_count` |
| POST | `/plans/{id}/deliver` | 登记待核实并发送 |
| POST | `/deliveries/{id}/verify` / `retry` | 核实 / 重试 |
| GET | `/api/deliveries` | 交付列表 |

接收服务：`POST /api/deliveries`、`GET /api/deliveries/{id}`、`GET /api/deliveries`，
以及模拟控制 `POST /api/simulate`（`fault` ∈ `none|http500|http404|http_verify500|timeout|slow|drop_after_store`，
`refuse=1` 模拟拒收）。终端终态不可修改（`POST /api/deliveries/{id}/reject` 返回 409）。

## 本地开发与测试

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest tests -q

# 两个进程本地跑
RECEIVER_DB_PATH=/tmp/r.sqlite3 uvicorn receiver.main:app --port 8001 &
APP_DB_PATH=/tmp/a.sqlite3 RECEIVER_URL=http://127.0.0.1:8001 uvicorn app.main:app --port 8000
```

`examples/library.csv` 可直接用于网页上传。

测试覆盖：传递连通组、精确划分、无解原因、禁入与连通组牵连、待核实→超时→幂等重试、
终态接纳永久禁入、拒收只解除本笔、补传导致名单停供与恢复、容量上限整批回滚、
重复切片号不覆盖、过期计算不发布、CSV 整批校验、发送前落库、接收方已写入但本地崩溃后的恢复、
接收服务校验和校验与终态不可变。
