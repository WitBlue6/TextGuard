import argparse
from llm.model import get_grammar_check_chain_with_memory, get_grammar_check_chain

def parse_args():
    parser = argparse.ArgumentParser(description="Chinese_Grammar_Error_Correction")
    parser.add_argument("--model_name", type=str, default="gpt-5-mini", help="Model name")
    parser.add_argument("--base_url", type=str, default="https://api.chatanywhere.tech/v1", help="Base URL")
    args = parser.parse_args()
    return args

def main(args):
    print("Hello from cgec!")
    grammar_check_with_memory = get_grammar_check_chain_with_memory(args.model_name, args.base_url)
    ans1 = grammar_check_with_memory.invoke(
            {"new_message": "通过这个项目，使我受益匪浅"}, 
            config={"session_id": "lzh"}
        ).content
    print(ans1)

if __name__ == "__main__":
    args = parse_args()
    main(args)
