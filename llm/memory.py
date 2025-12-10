from langchain_core.chat_history import BaseChatMessageHistory
#from langchain_core.messages import AIMessage, HumanMessage

# 最简单的内存实现（内存版）
class SimpleMemory(BaseChatMessageHistory):
    def __init__(self, max_messages=5):
        self.messages = []
        self.max_messages = max_messages

    def add_message(self, message):
        self.messages.append(message)
        # 超过限制时触发 memory 压缩
        self.messages = self.messages[-self.max_messages:]

    def clear(self):
        self.messages = []
