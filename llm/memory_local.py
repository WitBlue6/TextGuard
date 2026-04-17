"""
本地内存管理模块
支持LangChain的消息历史管理
"""
from langchain_core.chat_history import BaseChatMessageHistory
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class LocalSimpleMemory(BaseChatMessageHistory):
    """
    简单的内存实现（本地版本）
    使用字典存储会话历史
    """

    def __init__(self, max_messages: int = 5):
        self.messages: List[Dict[str, Any]] = []
        self.max_messages = max_messages
        logger.debug(f"初始化LocalSimpleMemory，最大消息数: {max_messages}")

    def add_message(self, message: Dict[str, Any]) -> None:
        """
        添加消息到历史

        :param message: 消息字典，包含role和content
        """
        self.messages.append(message)

        # 超过限制时触发 memory 压缩
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
            logger.debug(f"消息数量达到上限 {self.max_messages}，保留最后 {self.max_messages} 条消息")

    def clear(self) -> None:
        """清空消息历史"""
        self.messages = []
        logger.debug("LocalSimpleMemory已清空")

    def messages_to_langchain_messages(self) -> List[Dict[str, Any]]:
        """
        将本地消息格式转换为LangChain消息格式

        :return: LangChain消息列表
        """
        langchain_messages = []
        for msg in self.messages:
            langchain_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
        return langchain_messages

    def __len__(self) -> int:
        """返回消息数量"""
        return len(self.messages)

    def __repr__(self) -> str:
        """字符串表示"""
        return f"LocalSimpleMemory(messages={len(self.messages)}, max_messages={self.max_messages})"


class ChatMemoryManager:
    """
    会话记忆管理器
    用于管理多个会话的内存
    """

    def __init__(self):
        self.memory_store: Dict[str, LocalSimpleMemory] = {}
        logger.debug("初始化ChatMemoryManager")

    def get_memory(self, session_id: str) -> LocalSimpleMemory:
        """
        获取指定会话的内存

        :param session_id: 会话ID
        :return: LocalSimpleMemory实例
        """
        if session_id not in self.memory_store:
            self.memory_store[session_id] = LocalSimpleMemory()
            logger.debug(f"为会话 {session_id} 创建新的内存实例")
        return self.memory_store[session_id]

    def clear_memory(self, session_id: str) -> None:
        """
        清除指定会话的内存

        :param session_id: 会话ID
        """
        if session_id in self.memory_store:
            self.memory_store[session_id].clear()
            logger.debug(f"已清除会话 {session_id} 的内存")
        else:
            logger.warning(f"会话 {session_id} 不存在，无法清除")

    def clear_all(self) -> None:
        """清除所有会话的内存"""
        for session_id in list(self.memory_store.keys()):
            self.clear_memory(session_id)
        logger.info("已清除所有会话的内存")

    def get_memory_count(self) -> int:
        """返回当前活跃的会话数量"""
        return len(self.memory_store)


# 全局会话记忆管理器
_global_memory_manager = None


def get_memory_manager() -> ChatMemoryManager:
    """
    获取全局记忆管理器实例

    :return: ChatMemoryManager实例
    """
    global _global_memory_manager
    if _global_memory_manager is None:
        _global_memory_manager = ChatMemoryManager()
        logger.debug("初始化全局记忆管理器")
    return _global_memory_manager


def cleanup_all_memory():
    """
    清理所有会话记忆
    在关闭应用时调用
    """
    manager = get_memory_manager()
    manager.clear_all()
    logger.info("所有会话记忆已清理")
