"""电池售后运维 Agent（MLflow ResponsesAgent，"models from code" 方式）。

这是一个普通 Python 文件，不是 notebook：
- 07b notebook 用 `from agent import AGENT` 在本地测试；
- `mlflow.pyfunc.log_model(python_model="agent.py")` 把它原样打包部署。

工作流程（手写的 tool-calling 循环，便于理解 Agent 的本质）：
    用户消息 → LLM 决定调用哪个工具 → 执行工具 → 结果回填给 LLM → … → 最终回答
"""

import json
from typing import Any, Generator
from uuid import uuid4

import mlflow
from databricks.sdk import WorkspaceClient
from mlflow.entities import SpanType
from mlflow.models import ModelConfig
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from unitycatalog.ai.core.databricks import DatabricksFunctionClient

mlflow.openai.autolog()  # 每次 LLM 调用自动生成 Trace span

# ---------------------------------------------------------------------------
# 配置：开发时读取同目录的 agent_config.yaml；部署时由 log_model(model_config=...) 注入
# ---------------------------------------------------------------------------
config = ModelConfig(development_config="agent_config.yaml")
LLM_MODEL = config.get("llm_model")  # Unity Gateway 模型服务，如 system.ai.qwen3-next-80b-a3b-instruct
RISK_ENDPOINT = config.get("risk_endpoint")
FN_PROFILE = config.get("fn_profile")
FN_FEATURES = config.get("fn_features")
FN_ACTIONS = config.get("fn_actions")
FN_POLICY = config.get("fn_policy")
MAX_STEPS = int(config.get("max_steps"))

SYSTEM_PROMPT = """你是一家电动车公司的动力电池售后运维助手，服务对象是公司内部的售后服务顾问和车队运维人员。

工作流程：
1. 用户提到车辆 ID 时，先调用 get_vehicle_profile 查询车辆档案与电池状态；
2. 如果 safety_redline 为 true，这是最高优先级：必须按《维保规程》第 3 节给出 P0 紧急处置，不管模型风险多低；
3. 需要判断故障风险时，调用 predict_battery_risk；
4. 给处置建议前，必须调用 get_policy_doc('battery_maintenance_sop') 阅读规程，并调用 get_maintenance_actions 查询标准动作；
5. 涉及维修/更换费用或是否免费时，调用 get_policy_doc('battery_warranty_policy')，并以档案中的 in_warranty、soh_below_warranty_threshold 字段为准；
6. 车主关于充电、续航的一般性问题，调用 get_policy_doc('charging_faq')。

必须遵守的规则：
- 整包更换一律需要区域技术经理审批，要明确指出；
- 对车主的话术中不得透露模型故障概率数值，不得使用“马上起火”“报废”等恐慌性措辞；但触发安全红线时，安全注意事项必须清楚说明；
- 工具查不到信息时如实说明，绝不编造车辆数据。

输出格式（简体中文）：
【车辆概况】【风险判断】（含是否触发安全红线）【质保情况】【处置建议】（含优先级与时限）【对车主/车队的沟通话术】"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_vehicle_profile",
            "description": "根据车辆ID查询车辆档案与最新电池状态，包括是否在质保期内(in_warranty)、SOH是否低于质保门槛、是否触发安全红线(safety_redline)。",
            "parameters": {
                "type": "object",
                "properties": {"vehicle_id": {"type": "string", "description": "车辆ID，例如 V00042"}},
                "required": ["vehicle_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_battery_risk",
            "description": "调用机器学习模型预测车辆未来90天电池故障概率，并返回风险等级（low/medium/high）。",
            "parameters": {
                "type": "object",
                "properties": {"vehicle_id": {"type": "string", "description": "车辆ID，例如 V00042"}},
                "required": ["vehicle_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_maintenance_actions",
            "description": "根据车型和风险等级查询标准处置动作（优先级、动作、预计工时）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "vehicle_model": {"type": "string", "enum": ["city_ev", "suv_ev", "commercial_van"]},
                    "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
                },
                "required": ["vehicle_model", "risk_level"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_policy_doc",
            "description": "返回公司规程文档全文。",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_name": {
                        "type": "string",
                        "enum": ["battery_maintenance_sop", "battery_warranty_policy", "charging_faq"],
                        "description": "battery_maintenance_sop=维保规程（风险等级、安全红线、沟通要求）；"
                                       "battery_warranty_policy=质保政策；charging_faq=充电与续航FAQ",
                    }
                },
                "required": ["doc_name"],
            },
        },
    },
]


def as_text(content: Any) -> str:
    """把模型返回的 content 统一成纯文本。
    普通模型返回字符串；推理类模型（gpt-oss、qwen35 等）可能返回 [{"type": "reasoning", ...}, {"type": "text", "text": ...}]，
    这里丢弃推理过程，只保留最终文字。"""
    if content is None or isinstance(content, str):
        return content or ""
    parts = []
    for c in content:
        c = c if isinstance(c, dict) else getattr(c, "model_dump", lambda: {})()
        if c.get("type") in ("text", "output_text"):
            parts.append(c.get("text", ""))
    return "".join(parts)


def risk_level(p: float) -> str:
    """与 battery_maintenance_sop.md 中的定义保持一致。"""
    return "high" if p >= 0.6 else "medium" if p >= 0.3 else "low"


class BatteryServiceAgent(ResponsesAgent):
    def __init__(self):
        self.w = WorkspaceClient()
        self.uc = DatabricksFunctionClient()

    def _llm(self):
        """创建指向 Unity Gateway 的客户端。每次请求都新建：部署后凭证会轮换，缓存的 token 可能过期。"""
        from openai import OpenAI
        token = self.w.config.authenticate()["Authorization"].split(" ", 1)[1]
        return OpenAI(api_key=token, base_url=f"{self.w.config.host.rstrip('/')}/ai-gateway/mlflow/v1")

    # ------------------------------ 工具实现 ------------------------------
    def _uc(self, fn: str, params: dict[str, Any]) -> str | None:
        res = self.uc.execute_function(fn, params)
        if res.error:
            raise RuntimeError(res.error)
        return res.value

    @mlflow.trace(span_type=SpanType.TOOL)
    def call_tool(self, name: str, args: dict[str, Any]) -> str:
        try:
            if name == "get_vehicle_profile":
                return self._uc(FN_PROFILE, {"vid": args["vehicle_id"]}) or "未找到该车辆"
            if name == "get_maintenance_actions":
                return self._uc(FN_ACTIONS, {"model_name": args["vehicle_model"], "risk": args["risk_level"]}) or "[]"
            if name == "get_policy_doc":
                return self._uc(FN_POLICY, {"doc": args["doc_name"]}) or "未找到该文档"
            if name == "predict_battery_risk":
                raw = self._uc(FN_FEATURES, {"vid": args["vehicle_id"]})
                if not raw:
                    return "未找到该车辆的特征数据"
                resp = self.w.serving_endpoints.query(name=RISK_ENDPOINT, dataframe_records=[json.loads(raw)])
                p = float(resp.predictions[0][1])
                return json.dumps({"fault_probability_90d": round(p, 3), "risk_level": risk_level(p)})
            return f"未知工具: {name}"
        except Exception as e:  # 把错误告诉 LLM，让它自己决定如何应对
            return f"工具调用失败: {e}"

    # ------------------------------ 消息格式转换 ------------------------------
    @staticmethod
    def _to_chat_messages(request: ResponsesAgentRequest) -> list[dict]:
        """把 Responses API 格式的输入转成 Chat Completions 格式（只保留文本对话）。"""
        messages = []
        for item in request.input:
            d = item.model_dump() if hasattr(item, "model_dump") else dict(item)
            role, content = d.get("role"), d.get("content")
            if role not in ("user", "assistant", "system"):
                continue
            if isinstance(content, list):  # [{"type": "input_text", "text": "..."}]
                content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
            messages.append({"role": role, "content": content or ""})
        return messages

    # ------------------------------ Agent 主循环 ------------------------------
    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *self._to_chat_messages(request)]

        llm = self._llm()
        for _ in range(MAX_STEPS):
            resp = llm.chat.completions.create(
                model=LLM_MODEL, messages=messages, tools=TOOLS, temperature=0.1)
            msg = resp.choices[0].message

            if not msg.tool_calls:  # 模型不再调用工具 → 最终回答
                return ResponsesAgentResponse(
                    output=[self.create_text_output_item(text=as_text(msg.content), id=str(uuid4()))])

            messages.append({
                "role": "assistant",
                "content": as_text(msg.content) or None,
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ],
            })
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": self.call_tool(tc.function.name, args)})

        return ResponsesAgentResponse(output=[self.create_text_output_item(
            text="抱歉，这个问题需要的步骤太多，请把问题拆小一点再试。", id=str(uuid4()))])

    def predict_stream(self, request: ResponsesAgentRequest) -> Generator[ResponsesAgentStreamEvent, None, None]:
        # 简化实现：不做逐 token 流式，直接把最终结果作为一个完成事件返回
        for item in self.predict(request).output:
            yield ResponsesAgentStreamEvent(type="response.output_item.done", item=item)


AGENT = BatteryServiceAgent()
mlflow.models.set_model(AGENT)
