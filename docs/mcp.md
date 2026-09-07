# MCP integration

Install `pip install "llm-matgen[mcp]"`, then run `llm-matgen mcp --output-root output`.
The server exposes the same deterministic Tool Registry through `initialize`,
`tools/list`, `tools/call`, and safe `resources/read` requests. Artifact paths
are relative to the configured output root and must be present in a manifest;
path traversal and unregistered files are rejected.

## 生成与可视化（0.2.0）

默认 generate 工具已接通本地十类生成器，参数示例：

```json
{"generator":"vacancy","arguments":["--input","Si.cif","--target-element","Si","--count","1"]}
```

arguments 为对应 CLI 的参数数组，不经过 shell；输出目录由 MCP 服务设置。
工具返回 viewer 路径和 artifact:// 引用，resources/read 返回 text/html。
客户端应向用户展示查看链接或打开 HTML；MCP 不启动浏览器。search 和 download 在 0.2.1 已接通；其它查询工具仍需应用上下文配置。

## MP 优先流程（0.2.1）

对未提供本地结构的已有材料，先调用 search，再将选定的材料编号传给 download，最后将下载返回的 path 交给 generate。

```json
{"formula":"Si","limit":5}
```

以上为 search 的参数。download 参数为 `{"material_ids":["mp-149"]}`，编号必须来自搜索或用户明确指定。
有多个晶型时先确定目标，不能擅自把第一条结果当作所需晶型。MP_API_KEY 仅通过环境变量提供，禁止传入工具参数。
