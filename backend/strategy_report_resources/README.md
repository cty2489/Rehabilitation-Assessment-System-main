# 训练策略报告模板资源

`gesture_library.json` 和 `gesture_images/` 提供手势代码、分期映射与图片。
病例档案、患者数据、生成报告及原始参考文档不随源码发布。

管理员从“训练策略报告”上传医院端 ZIP，服务端建立异步任务，页面轮询进度并提供 PDF、JSON、Markdown 和 ZIP 下载。

部署使用现有后端推理与报告模型环境。以下环境变量可配置资源及输出目录：

- `STRATEGY_REPORT_RESOURCE_DIR`：默认本目录。
- `STRATEGY_REPORT_EXPORT_ROOT`：报告任务及输出目录。
- `STRATEGY_REPORT_MAX_UPLOAD_BYTES`：单文件上传上限。

需要沿用服务器中文字体时，将经过授权的字体部署到本目录的
`fonts/NotoSansSC-VF.ttf`，并一同保留字体许可证。未部署时渲染器按现有顺序选择系统字体或 ReportLab 中文字体。

运行报告专项单元测试：

```bash
PYTHONPATH=backend:. python -m pytest backend/test_strategy_report_production.py -q
```
