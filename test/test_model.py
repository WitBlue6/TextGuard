# model_test.py
import sys
import os

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.model import get_grammar_check_chain_with_memory, get_grammar_check_chain, get_memory

def main():
    print("Test PE Model with Memory")
    grammar_check_with_memory = get_grammar_check_chain_with_memory(
        model_name="gpt-5-mini",
        base_url="https://api.chatanywhere.tech/v1",
    )
    ans1 = grammar_check_with_memory.invoke(
            {"new_message": "通过这个项目，使我受益匪浅"}, 
            config={"session_id": "lzh"}
        ).content
    print("案例1:通过这个项目，使我受益匪浅\n", ans1)
    ans2 = grammar_check_with_memory.invoke(
            {"new_message": "他昨天去学校，但是今天会生病。"}, 
            config={"session_id": "lzh"}
        ).content
    print("案例2:他昨天去学校，但是今天会生病。\n", ans2)
    ans3 = grammar_check_with_memory.invoke(
            {"new_message": "我经过长时间的练习，他最终能熟练弹钢琴，手指不灵活"}, 
            config={"session_id": "lzh"}
        ).content
    print("案例3:我经过长时间的练习，他最终能熟练弹钢琴，手指不灵活\n", ans3)
    ans4 = grammar_check_with_memory.invoke(
            {"new_message": "？？？"}, 
            config={"session_id": "lzh"}
        ).content
    print("案例4:？？？\n", ans4)
    memory = get_memory("lzh")
    print("\n\n历史信息:", memory)

if __name__ == "__main__":
    main()
