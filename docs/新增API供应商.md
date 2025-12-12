# 新增 API 供应商（适配器开发指南）

本插件采用“供应商适配器”架构：不同 `api_type` 的差异在 `tl/api/*` 中隔离实现。新增一个 API 供应商时，建议按本文流程逐步落地，尽量避免改动主流程逻辑。

## 0. 先判断属于哪一类

通常分两类：

1. **OpenAI 兼容网关/反代**（最常见）
   - 请求路径/鉴权基本兼容 OpenAI
   - 差异集中在：参数字段名、返回结构、图片链接形态（相对路径/临时链接/多层嵌套）
   - 推荐：继承 `tl/api/openai_compat.py` 的 `OpenAICompatProvider`

2. **非 OpenAI 兼容协议**（例如 Google/Gemini 官方）
   - 请求/响应结构差异较大
   - 推荐：参考 `tl/api/google.py` 结构单独实现

## 1. 新建 Provider 文件

在 `tl/api/` 下新增文件，例如：`tl/api/my_provider.py`。

### 1.1 最小实现（必须）

Provider 需要实现两类职责：

- `build_request()`：产出 `ProviderRequest(url, headers, payload)`
- `parse_response()`：解析为 `(image_urls, image_paths, text_content, thought_signature)`

建议优先继承 `OpenAICompatProvider` 来复用参考图处理、图片落盘、文本提取等通用逻辑。

### 1.2 OpenAICompatProvider 的常用扩展点（推荐）

若你是 OpenAI 兼容网关，通常只需要覆盖这些“钩子”之一：

1. **请求参数差异**
   - 覆盖 `_prepare_payload()`：例如把 `image_config` 改名、增删字段、写入额外参数等

2. **返回图片链接差异（重点）**
   - 覆盖 `_handle_special_candidate_url()`：处理特殊 URL 并决定是否“吃掉”该候选（返回 `True` 表示已处理）
   - 典型场景：
     - 相对路径图片：`/images/xxx`
     - 临时缓存 URL：需要立即下载落盘，避免过期
     - 特殊协议：非 `http(s)`、非 `data:image/` 的下载入口

3. **从文本中额外提取图片链接**
   - 覆盖 `_find_additional_image_urls_in_text()`：例如从 Markdown 里提取相对路径 `![img](/images/xxx)`

可参考现成实现：`tl/api/grok2api.py`。

## 2. 注册到 provider registry

修改 `tl/api/registry.py`，把 `api_type` 映射到你的 Provider：

1. `from .my_provider import MyProvider`
2. 创建单例（推荐）：`_MY: Final[MyProvider] = MyProvider()`
3. 在 `get_api_provider()` 中添加分支返回 `_MY`

建议做一点归一化兼容：

- 允许 `my-provider`、`my_provider`、`myprovider` 等别名（根据实际需要）

## 3. 确认 api_base 透传（相对路径必需）

若你的返回里可能出现相对路径（如 `/images/xxx`），需要依赖 `api_base` 拼接为完整 URL。

当前主流程在 `tl/tl_api.py` 会将 `config.api_base` 透传到：

`provider.parse_response(..., api_base=config.api_base)`

因此，你的 Provider 可以在 `parse_response(..., api_base=...)` 或 `_handle_special_candidate_url(..., api_base=...)` 中使用它来构造完整 URL。

## 4. 更新配置与文档

为保证用户可配置/可选：

1. 修改 `_conf_schema.json`
   - 在 `api_settings.api_type.options` 里加入你的 `api_type`
   - 在 `hint` 中补充说明该类型用途（必要时）

2. 修改 `README.md`
   - 在 `api_settings.api_type` 说明中补充新类型

3.（可选）修改 `main.py`
   - 同步更新 `api_type` 相关的提示文案，便于排障

## 5. 自检建议

开发完成后建议至少跑：

- `python3 -m compileall -q .`
- `RUFF_CACHE_DIR=.ruff_cache ruff check .`
- `RUFF_CACHE_DIR=.ruff_cache ruff format .`

## 6. 常见坑位清单

1. **把 URL 当成本地路径**
   - 切图/压缩等流程通常要求本地文件路径；若上游返回 URL，需要先下载落盘

2. **图片链接重复添加**
   - 同一图片可能同时出现在 `images` 字段与文本里；建议在 Provider 内做去重或“已处理 URL 跳过”

3. **临时缓存链接过期**
   - 需要强制下载并只保留本地路径，避免用户端展示失败

