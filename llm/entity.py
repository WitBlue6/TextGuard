from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import json
import uuid
import logging
logger = logging.getLogger(__name__)


# 实体定义
class UIEntity(BaseModel):
    entity_id: str = Field(description="全局唯一ID，由系统生成，而非文本内容提供")
    name: str = Field(description="实体名称，例如一个物体、概念、人物、地点、组织等")
    type: str = Field(description="实体类型，由LLM根据文本自由归纳。例如：设备、人类、概念、化学物质、组织、事件、法规…")
    
    # 通用属性
    attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="实体的各种属性，不限制字段名。例如：{'产地':'泰国', '功率':'5kW'}"
    )
    
    # 如果文本中永远没有事件，可以为空
    events: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="与实体有关的事件列表，可选。例如：{'时间':'2021','动作':'发布','对象':'新型号'}"
    )
    
    # 通用关系
    relations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="实体间关系，例如：{'关系':'隶属于', '目标实体':'某集团'}"
    )

# 实体储存与检索
class EntityStore:
    def __init__(self):
        self.entities = {}  # entity_id -> UIEntity
        self.name_index = {}  # name -> entity_id

    def add_entity(self, entity: UIEntity):
        if entity.name not in self.name_index:
            # 新实体
            self.entities[entity.entity_id] = entity
            self.name_index[entity.name] = entity.entity_id
        else:
            # 已存在，需要合并
            old_id = self.name_index[entity.name]
            old_ent = self.entities[old_id]
            merged = self.merge(old_ent, entity)
            self.entities[old_id] = merged

    def merge(self, old: UIEntity, new: UIEntity) -> UIEntity:
        # 更新属性
        for k, v in new.attributes.items():
            old.attributes[k] = v

        # 合并事件
        for ev in new.events:
            if ev not in old.events:
                old.events.append(ev)

        # 合并关系
        for rel in new.relations:
            if rel not in old.relations:
                old.relations.append(rel)

        return old

    def all_entities(self):
        return list(self.entities.values())


def extract_entities(chain, text: str) -> List[UIEntity]:
    """
    从文本中提取实体。
    :param chain: 实体提取链
    :param text: 输入的文本
    :return: 提取到的实体列表
    """
    result = chain.invoke({"new_message": text},
                          config={"session_id": "lzh"}
                          ).content
    try:
        raw_entities = json.loads(result)
        #logger.info(f"原始实体提取结果: {raw_entities}")

        # 转化为 UIEntity 列表
        entities = [UIEntity(entity_id=str(uuid.uuid4()), **entity) for entity in raw_entities]
    except Exception as e:
        logger.error(f"实体提取失败: {e}")
        logger.debug("llm_response:", result)
        entities = []
    return entities 

def check_entity_consistency(chain, entity: UIEntity) -> Dict[str, Any]:
    """
    检查实体的一致性。
    :param chain: 实体一致性检查链
    :param entity: 待检查的实体
    :return: 检查结果
    """
    try:
        input = entity.model_dump_json()
    except Exception as e:
        logger.error(f"实体序列化失败: {e}")
        logger.debug("entity类型:", type(entity))
        logger.debug("entity:", entity)
        return {}
    result = chain.invoke({"new_message": input}).content
    return json.loads(result)

def summarize_entity_memory(chain, chunk: str) -> str:
    """
    对输入chunking后的实体列表进行总结。
    :param chain: 实体内存总结链
    :param chunk: 输入的文本chunk
    :return: 总结文本
    """
    result = chain.invoke({"new_message": chunk}).content
    return result