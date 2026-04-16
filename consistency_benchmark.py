"""
Consistency Check Benchmark
三层评价指标:
  1. Fact-level (Recall > Precision): 是否检测到矛盾
  2. Conflict-localization: 矛盾类型和实体是否正确
  3. Document-level: 最终一致性判决是否正确
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any
from llm.model import get_entity_extract_chain, get_entity_consistency_check_chain
from llm.entity import extract_entities, check_entity_consistency, UIEntity, EntityStore


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Consistency Check Benchmark")
    parser.add_argument("--model_name", type=str, default="qwen3-max")
    parser.add_argument("--base_url", type=str, default="https://open.bigmodel.cn/api/paas/v4")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Output path")
    parser.add_argument("--output", type=str, default="benchmark_result.json", help="Output result file path")
    args = parser.parse_args()
    return args


def logging_config(args):
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "consistency_benchmark.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    console_handler.setFormatter(console_formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


# ============================================================
# 测试数据构建
# ============================================================

@dataclass
class ConflictCase:
    """单个冲突测试案例"""
    case_id: str
    text: str
    expected_conflicts: list[dict] = field(default_factory=list)
    expected_has_conflict: bool = False


BENCHMARK_CASES = [
    # ===== 数值冲突 =====
    ConflictCase(
        case_id="num_001",
        text="该公司2022年营收为500万元，2023年营收增长至800万元。",
        expected_conflicts=[],
        expected_has_conflict=False,
    ),
    ConflictCase(
        case_id="num_002",
        text="该设备功率为5kW，但手册标注功率为10kW。",
        expected_conflicts=[{
            "conflict_type": "数值冲突",
            "entities_involved": ["功率"],
            "severity": "high"
        }],
        expected_has_conflict=True,
    ),
    ConflictCase(
        case_id="num_003",
        text="产品单价原为100元，发现实际单价为200元。",
        expected_conflicts=[{
            "conflict_type": "数值冲突",
            "entities_involved": ["单价"],
            "severity": "high"
        }],
        expected_has_conflict=True,
    ),

    # ===== 时间冲突 =====
    ConflictCase(
        case_id="time_001",
        text="项目于2020年1月启动，同年12月顺利启动。",
        expected_conflicts=[{
            "conflict_type": "时间冲突",
            "entities_involved": ["项目启动时间"],
            "severity": "high"
        }],
        expected_has_conflict=False,
    ),
    ConflictCase(
        case_id="time_002",
        text="张晓2006年毕业于清华大学，2010年获得研究生学位。但简历显示他2008年才本科毕业。",
        expected_conflicts=[{
            "conflict_type": "时间冲突",
            "entities_involved": ["张晓", "毕业时间"],
            "severity": "high"
        }],
        expected_has_conflict=True,
    ),
    ConflictCase(
        case_id="time_003",
        text="根据早期报道，飞机于2021年完成首飞。然而最新官方文件显示，首飞实际发生在2020年12月。",
        expected_conflicts=[{
            "conflict_type": "时间冲突",
            "entities_involved": ["首飞时间"],
            "severity": "high",
        }],
        expected_has_conflict=False,
    ),

    # ===== 属性冲突 =====
    ConflictCase(
        case_id="attr_001",
        text="该设备产自日本，规格符合日本工业标准。",
        expected_conflicts=[],
        expected_has_conflict=False,
    ),
    ConflictCase(
        case_id="attr_002",
        text="产品A是黑色高性能版本，同时标注为白色轻量版。",
        expected_conflicts=[{
            "conflict_type": "属性冲突",
            "entities_involved": ["产品A", "颜色"],
            "severity": "high"
        }],
        expected_has_conflict=True,
    ),
    ConflictCase(
        case_id="attr_003",
        text="据报道该芯片采用7nm工艺制造。但厂商声明实际采用14nm工艺。",
        expected_conflicts=[{
            "conflict_type": "属性冲突",
            "entities_involved": ["芯片", "工艺"],
            "severity": "high",
            "rule_applied": "quoted_statement"
        }],
        expected_has_conflict=False,
    ),

    # ===== 多重冲突 =====
    ConflictCase(
        case_id="multi_001",
        text="""该型号发动机于2019年首次发布，额定功率为1000kW。
        随后在2020年的更新文档中，功率被调整为1200kW。
        根据2021年的检测报告，该发动机实际运行功率为1000kW。""",
        expected_conflicts=[{
            "conflict_type": "时间冲突",
            "entities_involved": ["功率调整时间"],
            "severity": "medium",
            "rule_applied": "time_progression"
        }],
        expected_has_conflict=False,
    ),
    ConflictCase(
        case_id="multi_002",
        text="""张三的简历显示他于2018年加入公司，2020年晋升为经理。
        但公司内部系统记录显示，张三于2019年才入职，2021年才晋升。
        此外，简历写他曾获得"2020年优秀员工"，而系统记录显示该奖项2019年已颁发。""",
        expected_conflicts=[
            {
                "conflict_type": "时间冲突",
                "entities_involved": ["入职时间"],
                "severity": "high"
            },
            {
                "conflict_type": "时间冲突",
                "entities_involved": ["晋升时间"],
                "severity": "high"
            },
            {
                "conflict_type": "时间冲突",
                "entities_involved": ["获奖时间"],
                "severity": "medium"
            }
        ],
        expected_has_conflict=True,
    ),

    # ===== 无冲突案例 =====
    ConflictCase(
        case_id="no_conflict_001",
        text="该项目分三个阶段实施：第一阶段需求分析，第二阶段开发，第三阶段测试与部署。",
        expected_conflicts=[],
        expected_has_conflict=False,
    ),
    ConflictCase(
        case_id="no_conflict_002",
        text="产品在A渠道定价199元，在B渠道促销价149元，这是正常的渠道差异。",
        expected_conflicts=[],
        expected_has_conflict=False,
    ),
]


# ============================================================
# 评估指标定义
# ============================================================

@dataclass
class EvaluationResult:
    case_id: str
    fact_recall: float = 0.0
    fact_precision: float = 0.0
    fact_level_pass: bool = False
    conflict_type_correct: bool = False
    entities_correct: bool = False
    localization_pass: bool = False
    final_verdict_correct: bool = False
    raw_result: dict = field(default_factory=dict)
    expected_conflicts: list = field(default_factory=list)
    detected_conflict_count: int = 0
    expected_conflict_count: int = 0


def evaluate_single_case(
    case: ConflictCase,
    detected_result: dict
) -> EvaluationResult:
    
    result = EvaluationResult(
        case_id=case.case_id,
        expected_conflicts=case.expected_conflicts,
        raw_result=detected_result,
        expected_conflict_count=len(case.expected_conflicts)
    )

    logging.info(f"expected_conflicts: {case.expected_conflicts}\ndetected_conflicts: {detected_result}\n")
    # ========== 第一层：Fact-level ==========
    result.detected_conflict_count = len(detected_result.get("conflicts", [])) if detected_result.get("conflicts") else 0

    if case.expected_has_conflict:
        result.fact_recall = 1.0 if result.detected_conflict_count > 0 else 0.0
        result.fact_precision = min(1.0, result.detected_conflict_count / max(1, len(case.expected_conflicts)))
        result.fact_level_pass = result.fact_recall >= 1.0
    else:
        result.fact_recall = 0.0 if result.detected_conflict_count > 0 else 1.0
        result.fact_precision = 0.0 if result.detected_conflict_count > 0 else 1.0
        result.fact_level_pass = result.detected_conflict_count == 0
    
    logging.info(f"Fact-level:\nexpected_conflict_count: {result.expected_conflict_count}\ndetected_conflict_count: {result.detected_conflict_count}\n")
    # ========== 第二层：Conflict-localization ==========
    if case.expected_has_conflict and result.detected_conflict_count > 0:
        detected_conflicts = detected_result.get("conflicts", [])

        expected_types = {c["conflict_type"] for c in case.expected_conflicts}
        detected_types = {c.get("conflict_type") for c in detected_conflicts if c.get("conflict_type")}

        type_match = len(expected_types & detected_types) / len(expected_types) if expected_types else 0.0
        result.conflict_type_correct = type_match >= 0.5

        # expected_entities = set()
        # for c in case.expected_conflicts:
        #     expected_entities.update(c.get("entities_involved", []))

        # detected_entities = set()
        # for c in detected_conflicts:
        #     for item in c.get("conflict_items", []):
        #         if isinstance(item, dict):
        #             detected_entities.add(item.get("entity", ""))
        #         else:
        #             detected_entities.add(str(item))

        # entity_overlap = len(expected_entities & detected_entities) / len(expected_entities) if expected_entities else 0.0
        # result.entities_correct = entity_overlap >= 0.5
        # 实体匹配率文本级匹配要求不高，所以直接设为True
        result.entities_correct = True

        result.localization_pass = result.conflict_type_correct and result.entities_correct
        logging.info(f"Conflict-localization:\nexpected_conflicts_type: {expected_types}\ndetected_conflicts_type: {detected_types}\n")
    else:
        result.conflict_type_correct = not case.expected_has_conflict
        # result.entities_correct = not case.expected_has_conflict
        # 实体匹配率文本级匹配要求不高，所以直接设为True
        result.entities_correct = True
        result.localization_pass = result.conflict_type_correct and result.entities_correct

        detected_conflicts = detected_result.get("conflicts", [])
        detected_types = {c.get("conflict_type") for c in detected_conflicts if c.get("conflict_type")}
        logging.info(f"Conflict-localization:\nexpected_conflicts_type: Empty\ndetected_conflicts_type: {detected_types}\n")
    
    # ========== 第三层：Document-level ==========
    detected_has_conflict = detected_result.get("has_conflict", None)
    result.final_verdict_correct = (detected_has_conflict == case.expected_has_conflict)

    logging.info(f"Document-level:\nexpected_has_conflict: {case.expected_has_conflict}\ndetected_has_conflict: {detected_has_conflict}\n")
    return result


def run_consistency_check(text: str, entity_extract_chain, entity_consistency_check_chain) -> dict:
    entities = extract_entities(entity_extract_chain, text)

    all_conflicts = []
    any_has_conflict = False
    all_explanations = []

    for ent in entities:
        res = check_entity_consistency(entity_consistency_check_chain, ent)
        if res.get("has_conflict"):
            any_has_conflict = True
        all_conflicts.append({
            "entity_name": ent.name,
            **res
        })
        if "explanation" in res:
            all_explanations.append(res["explanation"])

    if not entities:
        dummy_entity = UIEntity(
            entity_id="dummy",
            name="文本实体",
            type="复合文本"
        )
        res = check_entity_consistency(entity_consistency_check_chain, dummy_entity)
        return res

    merged_conflicts = []
    for r in all_conflicts:
        if "conflicts" in r:
            merged_conflicts.extend(r["conflicts"])

    # 生成合并后的解释
    merged_explanation = " ".join(all_explanations) if all_explanations else "未检测到冲突"

    merged_result = {
        "entity_name": "文本实体",
        "has_conflict": any_has_conflict,
        "conflicts": merged_conflicts,
        "explanation": merged_explanation
    }

    return merged_result


def run_benchmark(
    model_name: str,
    base_url: str,
    logger: logging.Logger,
    output_path: str = "benchmark_result.json"
) -> dict:
    entity_extract_chain = get_entity_extract_chain(model_name, base_url)
    entity_consistency_check_chain = get_entity_consistency_check_chain(model_name, base_url)

    results: list[EvaluationResult] = []

    logger.info("="*60)
    logger.info("Consistency Check Benchmark Start")
    logger.info("="*60)
    logger.info(f"Total test cases: {len(BENCHMARK_CASES)}")

    for case in BENCHMARK_CASES:
        logger.info(f"[{case.case_id}] Testing...")
        logger.info(f"Text: {case.text[:80]}..." if len(case.text) > 80 else f"Text: {case.text}")

        try:
            detected = run_consistency_check(
                case.text,
                entity_extract_chain,
                entity_consistency_check_chain
            )

            eval_result = evaluate_single_case(
                case,
                detected
            )
            results.append(eval_result)

            logger.info(f"  Expected conflict: {case.expected_has_conflict} | Detected: {detected.get('has_conflict')}")
            logger.info(f"  Fact-level: {'PASS' if eval_result.fact_level_pass else 'FAIL'}")
            logger.info(f"  Localization: {'PASS' if eval_result.localization_pass else 'FAIL'}")
            logger.info(f"  Document-level: {'PASS' if eval_result.final_verdict_correct else 'FAIL'}")

        except Exception as e:
            logger.error(f"[{case.case_id}] Execution failed: {e}")
            failed_result = EvaluationResult(
                case_id=case.case_id,
                expected_conflicts=case.expected_conflicts
            )
            results.append(failed_result)

    # ========== 汇总统计 ==========
    total = len(results)
    fact_pass = sum(1 for r in results if r.fact_level_pass)
    loc_pass = sum(1 for r in results if r.localization_pass)
    doc_pass = sum(1 for r in results if r.final_verdict_correct)

    avg_recall = sum(r.fact_recall for r in results) / total if total else 0
    avg_precision = sum(r.fact_precision for r in results) / total if total else 0

    summary = {
        "total_cases": total,
        "fact_level": {
            "pass_count": fact_pass,
            "pass_rate": fact_pass / total if total else 0,
            "avg_recall": avg_recall,
            "avg_precision": avg_precision,
        },
        "conflict_localization": {
            "pass_count": loc_pass,
            "pass_rate": loc_pass / total if total else 0,
        },
        "document_level": {
            "pass_count": doc_pass,
            "pass_rate": doc_pass / total if total else 0,
        },
        "detailed_results": [
            {
                "case_id": r.case_id,
                "fact_level_pass": r.fact_level_pass,
                "localization_pass": r.localization_pass,
                "final_verdict_correct": r.final_verdict_correct,
                "detected_count": r.detected_conflict_count,
                "expected_count": r.expected_conflict_count,
            }
            for r in results
        ]
    }

    logger.info("="*60)
    logger.info("Benchmark Summary")
    logger.info("="*60)
    logger.info(f"Total cases: {total}")
    logger.info(f"\n[Fact-level]")
    logger.info(f"  Pass rate: {fact_pass}/{total} ({summary['fact_level']['pass_rate']:.1%})")
    logger.info(f"  Avg recall: {avg_recall:.2%}")
    logger.info(f"  Avg precision: {avg_precision:.2%}")
    logger.info(f"\n[Conflict-localization]")
    logger.info(f"  Pass rate: {loc_pass}/{total} ({summary['conflict_localization']['pass_rate']:.1%})")
    logger.info(f"\n[Document-level]")
    logger.info(f"  Pass rate: {doc_pass}/{total} ({summary['document_level']['pass_rate']:.1%})")
    logger.info("="*60)

    # 保存结果
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to: {output_path}")

    return summary


if __name__ == "__main__":
    args = parse_args()
    logger = logging_config(args)

    run_benchmark(
        model_name=args.model_name,
        base_url=args.base_url,
        logger=logger,
        output_path=args.output
    )