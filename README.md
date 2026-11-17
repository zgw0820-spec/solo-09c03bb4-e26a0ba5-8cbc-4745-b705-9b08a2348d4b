# 风机停机预警训练平台（可续跑 + 完成通知）

用 FastAPI + NumPy + SQLite 实现的风机停机预警逻辑回归训练平台。训练由两个工作器
容器执行，每个批次都把权重、动量、损失和进度作为**一个原子断点**提交；工作器退出后
另一个工作器自动接管，并从最近断点继续。最终权重与不中断训练逐位一致。

训练完成后，若提交时提供了运维接收地址，平台会把完成消息**可靠地**投递到接收端；
投递失败、接收端暂不可用或工作器重启都不会丢失消息，也不会改变已经完成的训练结果。

## 组成

| 组件 | 说明 |
| --- | --- |
| `api` | FastAPI，提交任务、查询进度/断点/权重、查询通知状态、Swagger 演示、内置演示接收端 |
| `worker-1` / `worker-2` | 同一镜像的 `python -m app.worker`，共享同一个 SQLite 文件；各带一个投递线程 |
| `training-data` | Docker 卷，挂载到三个容器的 `/data` |

两个工作器共享**同一个** SQLite 数据库（`/data/training.db`），因此"检查租约并写入
断点"和"写入断点"是同一个 `BEGIN IMMEDIATE` 事务，不需要跨库协调。完成通知同样落在
这个数据库里，使用事务性 outbox。

## 训练约定

- 输入 8～64 条记录，每条是风机振动/电流等特征，标签 `0`=正常、`1`=停机预警；
  各特征列按提交样本标准化后追加偏置列，权重长度为 `特征数 + 1`。
- 轮数 2～5；按输入顺序每 8 条一批（最后一批可不足 8 条）。
- 逻辑回归（sigmoid + 二元交叉熵），SGD，学习率 `0.1`，动量 `0.9`，float64。
- 每批提交的断点包含：`step`、`training_epoch`、`batch_in_epoch`、`weights`、
  `momentum_velocity`、`batch_loss`、`loss`（全量交叉熵）、`progress`、提交时间、
  `worker_id`、`lease_epoch`。

## 启动

```bash
docker compose up -d --build
# Swagger: http://localhost:8000/docs
```

`worker-1` / `worker-2` 故意不设 `restart` 策略，便于演示接管。被 kill 的工作器用
`docker compose start worker-1` 恢复（它会以空闲状态重新加入，不会抢占已完成任务）。

## API

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `POST` | `/jobs` | 提交 `{"features": [[...]], "labels": [...], "epochs": 3, "notification_url": "..."}`，返回 201 与任务 |
| `GET` | `/jobs` | 最近任务列表 |
| `GET` | `/jobs/{job_id}` | 已确认步数、进度、损失、当前权重；完成时含 `final_weights` 和 `notification` |
| `GET` | `/jobs/{job_id}/checkpoints` | 全部批次断点（可按 `after_step` 增量拉取） |
| `GET` | `/jobs/{job_id}/events` | 审计事件：创建、开始/从第几步续训、完成、通知入队、丢失租约 |
| `GET` | `/jobs/{job_id}/notifications` | 完成消息内容与投递状态（`pending`/`delivered`、尝试次数、上次错误） |
| `GET` | `/workers` | 工作器心跳与租约持有者 |
| `GET` | `/health` | 健康检查 |
| `POST` | `/demo/receiver` | 内置演示接收端：接收完成消息，按 `event_id` 去重 |
| `GET` | `/demo/receiver/inbox` | 演示接收端已收到的消息（含重复投递次数） |
| `DELETE` | `/demo/receiver/inbox` | 清空演示收件箱（不影响训练与 outbox） |

`Swagger`（`/docs`）里直接 `POST /jobs` 提交即可看到训练开始；`GET /jobs/{id}`、
`GET /jobs/{id}/checkpoints` 与 `GET /jobs/{id}/notifications` 查看进度、每个断点的
权重/动量/损失，以及完成消息的投递结果。

## 完成通知

`POST /jobs` 的 `notification_url` 可选，必须是绝对 `http(s)` URL。设置后：

1. **与完成状态同一事务入队。** 末批权重、`completed` 状态、`finished_at` 与 outbox
   中待发送的消息在同一个 `BEGIN IMMEDIATE` 事务提交；不存在"权重已完成但消息没入队"
   或"消息已入队但权重回滚"的中间态。旧工作器迟到唤醒时，其围栏校验失败，既不能重写
   完成状态，也不能重复入队。
2. **消息内容固定。** 入队时一次性生成：

   ```json
   {
     "event_id": "job.completed:<job_id>",
     "event": "job.completed",
     "job_id": "<job_id>",
     "final_weights": [ ... ],
     "loss": 0.3006965745459481,
     "completed_steps": 15,
     "total_steps": 15,
     "finished_at": 1763...
   }
   ```

   `event_id` 由任务号确定，永不改变；`payload_json` 入队后不再修改。任何重试都原样
   重发同样的字节，接收端按 `event_id` 去重即可。请求头同时带 `X-Event-Id`。
3. **投递在训练之外。** 每个工作器有一个投递线程，和训练循环并行；它在共享数据库中
   以带时限的认领（claim）原子地取一条到期消息，POST 后在收到 2xx 时标记 `delivered`。
   连接拒绝、超时、非 2xx 都算失败：消息保持 `pending`，按指数退避
   （`notification_retry_min` 起，封顶 `notification_retry_max`）重试。
4. **两个工作器并行、重启都安全。** claim 用 `BEGIN IMMEDIATE` 条件更新，两个工作器
   同一时刻只有一方发送；发送方崩溃留下 `pending`，认领到期后由另一个工作器（或重启后
   的同一工作器）重发。所有状态在共享 SQLite 中，服务重启不会丢消息。
5. **回执丢失可重送。** 接收端已保存但 201 没回到发送方时，消息仍是 `pending`，之后
   重发；因为内容与 `event_id` 不变，接收端只会记一次（演示接收端
   `/demo/receiver` 会返回 `duplicate: true` 并累加 `receive_count`；同一 `event_id`
   内容不同则 422）。
6. **故障不影响训练结果。** 投递线程只读写 `outbox`，不碰任务、断点或租约。接收端
   不可用时，`GET /jobs/{id}` 的 `status`、`final_weights`、`loss`、`finished_at`
   保持已提交的值，`GET /jobs/{id}/notifications` 显示 `pending`、尝试次数与错误。
7. **重复完成不会重复入队。** 已完成任务再被迟到工作器调用时，只返回完成状态；入队
   只在 `completed_steps` 真正从 `total_steps - 1` 变为 `total_steps` 的那一次事务里
   发生，且 `outbox.event_id` 是主键，即便重复执行也最多一条。

演示环境中"运维接收端"就是 API 自带的 `POST /demo/receiver`（与 Swagger 同端口，
便于直接用 Swagger 操作）：Compose 内工作器用 `http://api:8000/demo/receiver`，
本地进程用 `http://127.0.0.1:8000/demo/receiver`。接真实运维系统时只需把
`notification_url` 换成对方地址；对方需返回 2xx 表示已接收（超时/其他状态码会重试，
3xx 不跟随，按失败处理，避免 urllib 把 POST 改成无 body 的 GET）。

## 续跑与防回退机制

1. 工作器每批在**同一个事务**里写检查点、更新任务进度、刷新租约心跳。任何一项失败
   则整体回滚，不会出现"权重已变但进度没变"的中间态。
2. 租约保存在同一数据库的 `leases` 表。工作器每 `1.5s` 心跳；持有者心跳超过
   `6s`（`TRAINING_LEASE_SECONDS`）未更新，另一个工作器才可接管，接管时
   `epoch + 1`。
3. 每次提交前在事务内校验 `(worker_id, epoch)` 仍是租约持有者。被接管后仍存活（或
   迟到唤醒）的旧工作器校验失败，直接 `LeaseLost` 退出：既不能回退断点，也不能把
   别人的完成状态改写，更不能重复提交完成事件。
4. 完成状态由 `step == total_steps` 在提交事务内一次性置位；旧工作器无法再写入。
5. SQLite 使用 WAL + `synchronous=FULL`，已提交的断点在进程崩溃/容器重启后仍然存在。

## 演示与验证

```bash
# 完整演示（Docker）：提交 -> 观察 -> kill 当前持有者 -> 接管续跑 -> 对比不中断训练
#                   -> 确认完成消息已投递到 /demo/receiver
python3 scripts/demo.py

# 没有 Docker 时用本机进程演示同样流程
python3 scripts/demo.py --local

# 不依赖 Docker 的四项自检
python3 scripts/selftest.py

# 单元/集成测试（含真实 uvicorn + 两个工作器进程、SIGKILL 接管、迟到工作器围栏、
# 完成通知的首次投递/接收端不可用重试/回执丢失重送/双工作器只发一次/重启补发）
python3 -m pytest -q
```

演示脚本第 6 步会用同一份输入再训练一次（不中断），并断言两次的 `final_weights`、
`loss`、`momentum_velocity` 完全相等；第 7 步确认完成消息已送达演示接收端，
且 `payload` 与 API 中记录的最终权重/损失一致。`--no-notify` 可跳过通知。

## 环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `TRAINING_DATA_DIR` | `./data` | 数据库目录 |
| `TRAINING_DB_PATH` | `$TRAINING_DATA_DIR/training.db` | 共享数据库 |
| `TRAINING_LEASE_SECONDS` | `6` | 租约超时 |
| `TRAINING_HEARTBEAT_INTERVAL` | `1.0` | 心跳间隔（必须小于租约） |
| `TRAINING_POLL_INTERVAL` | `0.3` | 空闲轮询/批间间隔 |
| `TRAINING_BATCH_DELAY_SECONDS` | `0` | 模拟每批耗时，便于演示 |
| `TRAINING_NOTIFY_POLL_INTERVAL` | `0.5` | 投递线程扫描待发消息的间隔 |
| `TRAINING_NOTIFY_TIMEOUT` | `5` | 单次 HTTP 投递超时（秒） |
| `TRAINING_NOTIFY_RETRY_MIN` | `1` | 首次失败后的重试间隔（秒），随后指数增长 |
| `TRAINING_NOTIFY_RETRY_MAX` | `30` | 重试间隔上限（秒） |
| `TRAINING_NOTIFY_CLAIM_SECONDS` | `15` | 发送认领时限；超时后其他工作器可重发 |
| `TRAINING_DEMO_PAUSE_FILE` | 未设置 | 演示用：文件存在时在批间暂停（不心跳） |
| `TRAINING_DEMO_CRASH_AFTER` | 未设置 | 演示用：指定步数提交后 `os._exit` |

## 目录

```
app/main.py       API 与 Swagger、演示接收端
app/worker.py     工作器主循环、租约续训、投递线程
app/store.py      任务状态机、批次提交、断点/通知查询
app/outbox.py     完成消息的事务性入队、认领、投递/失败标记
app/notifier.py   HTTP 投递与指数退避重试
app/receiver.py   演示接收端（按 event_id 去重）
app/lease.py      租约获取/续约/围栏校验
app/model.py      逻辑回归、交叉熵、SGD + 动量
app/features.py   输入校验与标准化
app/db.py         SQLite 连接与表结构
tests/            单元、API、进程级集成、通知投递测试
scripts/demo.py   Swagger 同款接口的端到端演示
scripts/selftest.py  无 Docker 自检
```
