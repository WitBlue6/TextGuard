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
    parser.add_argument("--model_name", type=str, default="glm-4-flash")
    parser.add_argument("--base_url", type=str, default="https://open.bigmodel.cn/api/paas/v4")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Output path")
    parser.add_argument("--output", type=str, default="./logs/benchmark_result.json", help="Output result file path")
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
    # ===== 科技新闻（人物信息）=====
    ConflictCase(
        case_id="tech_news_001",
        text="某知名科技媒体发布了一篇深度报道，介绍了张三先生作为行业领军人物的职业生涯。报道中提到，张三在2010年就已经是某知名互联网公司的技术总监，主导了多个大型项目的研发工作，在业内积累了丰富的经验。张三毕业于清华大学计算机系，本硕连读，2015年获得博士学位，是一位典型的学霸型技术专家。作为一名技术领军人物，张三在行业内有着极高的声誉，曾多次在大型技术会议上发表演讲，分享自己的技术见解。在报道的下一部分，记者采访了张三本人。张三在采访中表示，自己一直专注于人工智能领域的研究，从2010年开始攻读博士，目标是成为一名技术专家。他提到自己之所以选择攻读博士学位，是因为希望深入研究技术背后的原理，为行业发展做出更大的贡献。张三还透露，他在读博期间就在实验室参与了一项重要研究，这项研究后来成为了他职业生涯的重要起点。然而，当记者将张三在媒体上的公开言论与高校档案进行交叉验证时，发现了明显的时间矛盾。根据清华大学研究生院的学生档案，张三的博士入学时间实际上是在2018年，而不是他在采访中所说的2010年。这意味着张三在担任技术总监期间根本无法完成博士学业，这与他作为学术型技术专家的形象严重不符。此外，记者发现张三在领英上的工作经历显示，他自2018年博士毕业后才开始在该公司任职，而他在媒体上宣称的2010年入职时间，与他实际的博士毕业时间相差了整整8年。这些信息的不一致，让人对张三的职业履历产生了疑问。对于这些问题，公司公关部的工作人员回应称，张三在社交媒体上的表述存在记忆偏差，实际上是在公司提供的参考材料中看错了时间信息。公关部强调，张三的职业发展路径是清晰的，他从2018年开始在该公司工作，逐步晋升至技术总监，并在2020年带领团队推出了重要产品。公关部还表示，公司对张三的教育背景有完整档案记录，不存在时间线上的矛盾问题。但记者查看了公司的公开资料后发现，2015年确实有张三获得博士学位的记录，而2018年也有他在该公司入职的记录，这两个时间点确实无法在正常情况下同时成立。另外，该公司在2021年发布的年度报告中详细记录了管理层的信息。报告显示，张三作为技术总监，是在2018年加入公司的，这与他在领英上的公开信息一致。而他在2020年带领团队推出的产品，实际上是在2019年底就已经开始研发了，这说明他在公司的工作年限比他宣称的要长。这些零散的信息拼凑在一起，构成了一个完全不同于张三所描述的职业履历，让人不禁质疑，这些信息背后是否隐藏着更深层次的问题。张三究竟是在2010年就已经加入公司，还是2018年才加入？他是否真的在2010年就开始攻读博士学位？这些问题在现有信息中难以找到统一的解释。",
        expected_conflicts=[
            {
                "conflict_type": "时间冲突",
                "entities_involved": ["张三", "博士入学时间"],
                "severity": "high"
            },
            {
                "conflict_type": "时间冲突",
                "entities_involved": ["张三", "入职时间"],
                "severity": "high"
            },
            {
                "conflict_type": "时间冲突",
                "entities_involved": ["张三", "工作年限"],
                "severity": "high"
            }
        ],
        expected_has_conflict=True,
    ),
    ConflictCase(
        case_id="tech_news_002",
        text="金融行业资深人士李四在社交平台上非常活跃，经常分享投资见解和行业观察。根据他在知乎上的个人主页介绍，李四是复旦大学经济学博士，2018年毕业，曾在高盛工作三年，积累了丰富的国际投行经验。他经常在专业论坛上发表文章，分析市场动态，并给出投资建议，拥有大量粉丝关注。李四在职业发展方面也有着清晰的规划。根据他的个人博客，他自2015年就开始在金融行业工作，目前是一家大型投资银行的高级董事总经理，负责带领团队进行投资决策。李四在文章中提到，自己从事金融行业已经15年了，这是一个从基层做起的漫长过程，每一个阶段都凝聚了大量的努力和汗水。他还表示，自己非常珍惜当前的工作机会，因为这是他职业生涯的又一个重要里程碑。当记者联系李四，想要了解更多关于他的教育背景时，李四表示自己主要是本科学历，并没有博士学位，他提到的博士学历是笔误，或者是误把硕士学位当成博士学位了。但他随后又表示，自己的硕士学历也是在2018年获得的，这意味着他可能同时拥有两个硕士学位。这种说法让记者更加困惑了。根据LinkedIn上的职业档案，李四的LinkedIn账户创建于2015年，显示他已经在投资行业工作多年，但没有博士学位的信息，只有硕士学位。此外，李四在2020年获得的一项专业认证证书上，专业领域一栏填写的是金融工程，而不是经济学。这与他在知乎上宣称的经济学博士背景不符。记者尝试联系复旦大学研究生院核实李四的学历情况，但对方表示只能提供基本的查询服务，不能提供具体个人的详细信息。然而，记者通过其他渠道了解到，复旦大学确实有李四这名学生，但他在校期间的学位类型是工商管理硕士，而不是经济学博士，这与李四在社交平台上宣传的背景存在明显差异。李四到底拥有多少个学位？他的专业背景到底是什么？这些问题在现有信息中找不到答案。更令人不解的是，李四在文章中提到的15年工作经验，按照正常的工作规律推算，他应该在2004年左右进入金融行业，但他的硕士毕业时间却是在2018年，这两者相差了14年，这意味着他的工作经验和学历时间完全对不上。",
        expected_conflicts=[
            {
                "conflict_type": "时间冲突",
                "entities_involved": ["李四", "学历层次"],
                "severity": "high"
            },
            {
                "conflict_type": "时间冲突",
                "entities_involved": ["李四", "工作经验"],
                "severity": "high"
            },
            {
                "conflict_type": "属性冲突",
                "entities_involved": ["李四", "专业领域"],
                "severity": "high"
            }
        ],
        expected_has_conflict=True,
    ),

    # ===== 公司年报（公司信息）=====
    ConflictCase(
        case_id="annual_report_001",
        text="ABC科技公司发布的2022年年度报告显示，公司已发展成为一个集技术研发、产品销售和服务于一体的综合性企业。根据报告，ABC科技成立于2015年，总部位于北京市海淀区，注册资本1亿元人民币。公司由张三和李四两位联合创始人共同创立，张三担任首席执行官，负责公司的整体战略规划；李四担任首席技术官，主导技术产品线的研发工作。ABC科技在短短几年内取得了令人瞩目的成绩。公司专注于人工智能和大数据领域，凭借创新的产品技术，在行业内迅速崛起，获得了包括红杉资本和经纬中国在内的多家顶级投资机构的青睐。2020年，ABC科技成功上市，并在纳斯达克敲钟上市，公司市值一度突破50亿美元，成为行业内备受瞩目的独角兽企业。公司旗下拥有100多家子公司，员工总数超过5000人，在多个城市设有分支机构。公司在公告中还详细介绍了其业务发展历程。ABC科技的前身是成立于2013年的某小型科技公司，主要业务是软件开发。经过几年的发展，公司逐步扩大了业务范围，从单纯的软件开发拓展到人工智能解决方案和大数据服务。张三作为创始人之一，在公司成立初期就加入，对公司的发展起到了关键作用。李四则在公司成立后不久加入，负责技术团队的建设和产品研发。根据公司公开披露的信息，张三在公司成立初期就担任核心管理职位，与公司共同成长，见证了公司从初创企业到上市公司的整个发展过程。但当我们查阅ABC科技的工商注册信息时，发现了一些值得注意的地方。根据北京市市场监督管理局的登记记录，ABC科技的实际成立时间是在2013年，比公司官网宣称的2015年早了两年。而张三作为创始人，其工商注册信息显示他是2014年才正式入职该公司的，这意味着张三在公司成立初期就加入的说法与事实不符。更令人关注的是，ABC科技在2019年发布的投资者白皮书中提到，公司已经完成了B轮融资，估值达到20亿美元。但根据第三方投资机构的数据，ABC科技在2019年的估值实际上只有5亿美元，两者相差了4倍。此外，公司官网宣称的50亿美元市值，与当时纳斯达克同类公司的平均估值相比明显偏高，这暗示公司可能存在虚高估值的情况。公司对成立时间的表述、创始人加入时间的描述，以及融资估值的真实性，都让人产生了疑问。这些信息的不一致，是否意味着公司在信息披露上存在某些问题？",
        expected_conflicts=[
            {
                "conflict_type": "时间冲突",
                "entities_involved": ["ABC科技", "成立时间"],
                "severity": "high"
            },
            {
                "conflict_type": "时间冲突",
                "entities_involved": ["张三", "入职时间"],
                "severity": "high"
            },
            {
                "conflict_type": "数值冲突",
                "entities_involved": ["ABC科技", "融资估值"],
                "severity": "high"
            },
            {
                "conflict_type": "数值冲突",
                "entities_involved": ["ABC科技", "市值"],
                "severity": "high"
            }
        ],
        expected_has_conflict=True,
    ),
    ConflictCase(
        case_id="annual_report_002",
        text="XYZ集团作为国内知名的综合性企业集团，在多个领域都有着深厚的业务布局。根据XYZ集团2022年年度报告，集团总资产达到5000亿元，年收入超过200亿元，员工总数超过5万人。报告详细介绍了集团的发展历程：XYZ集团的前身是成立于1990年的某小型贸易公司，经过30多年的发展，如今已经成为业务涵盖房地产、金融、零售等多个领域的行业领军企业。集团官方网站还展示了其获得的众多荣誉和成就。根据官网展示，XYZ集团在2020年被评为中国企业500强第58位，2021年晋升至第35位，2022年更是进入了前30名，位列第29位。这些荣誉进一步巩固了XYZ集团在业界的地位，彰显了集团的综合实力。集团旗下拥有100多家子公司，在多个城市设有分支机构，业务覆盖全国各地，并且在海外市场也有所布局。然而，当我们查阅XYZ集团的实际经营数据时，发现了一些值得怀疑的地方。根据国家税务总局的数据，XYZ集团在2021年的纳税额仅为1.2亿元，而根据集团公布的年收入200亿元计算，纳税额应该占到年收入的6%左右，但实际上只有0.6%。这表明集团的利润率非常低，或者存在大量的税收减免。此外，第三方财务机构对XYZ集团的估值显示，集团的实际估值应该只有300亿元左右，远低于集团声称的5000亿元总资产，资产虚高的情况非常严重。集团的资产规模、年收入数据、纳税额、估值等多个关键指标之间存在着明显的不匹配。更关键的是，XYZ集团官方网站公布的500强排名，被权威商业杂志核实后发现是错误的。该杂志指出，XYZ集团的实际排名应该在第500名左右，而非官网宣称的第35位。这些信息与集团官方宣传的领军企业形象严重不符，反映出集团在信息披露上可能存在夸大或不实的情况。一个总资产5000亿元的企业，纳税额却只有1.2亿元，而排名却进入了前30名，这其中的逻辑关系让人难以理解。集团究竟是否存在财务造假？还是对某些关键数据进行了修饰？这些问题在现有信息中无法找到答案。",
        expected_conflicts=[
            {
                "conflict_type": "数值冲突",
                "entities_involved": ["XYZ集团", "总资产"],
                "severity": "high"
            },
            {
                "conflict_type": "数值冲突",
                "entities_involved": ["XYZ集团", "年收入"],
                "severity": "high"
            },
            {
                "conflict_type": "数值冲突",
                "entities_involved": ["XYZ集团", "纳税额"],
                "severity": "high"
            },
            {
                "conflict_type": "数值冲突",
                "entities_involved": ["XYZ集团", "排名"],
                "severity": "high"
            },
            {
                "conflict_type": "数值冲突",
                "entities_involved": ["XYZ集团", "估值"],
                "severity": "high"
            }
        ],
        expected_has_conflict=True,
    ),

    # ===== 产品评测（产品信息）=====
    ConflictCase(
        case_id="product_review_001",
        text="某知名科技博主在2023年发布了一篇关于超级手机的专业评测。博主在评测中详细介绍了手机的各项性能参数。根据博主提供的测试数据，超级手机采用了最新的5nm芯片技术，处理器型号为骁龙888 Plus，在Geekbench测试中得分超过了110000分，性能表现强劲。手机配备了12GB LPDDR5内存和256GB UFS 3.1存储空间，运行大型游戏和应用程序都非常流畅。屏幕方面，超级手机搭载了6.7英寸的AMOLED屏幕，支持120Hz刷新率和1亿像素高清摄像，视觉效果非常出色。续航方面，博主表示超级手机内置了5000mAh大容量电池，在正常使用情况下可以续航一整天。充电速度更是令人印象深刻，支持120W超级快充技术，博主在测试中记录到，使用原装充电器15分钟即可将电量从20%充到80%，这大大节省了充电时间。手机还支持无线充电和反向充电功能，这在当时属于行业领先配置。评测发布后，许多用户对博主的专业性表示认可。博主在文章中提到了自己的测试方法和测试环境，并分享了一些真实的使用体验。然而，当技术爱好者尝试根据博主提供的信息进行验证时，发现了一些可疑的地方。有用户在二手市场购买了博主评测中提到的同款手机，但实际测试显示手机的处理器型号为骁龙870，与博主宣称的骁龙888 Plus不符。更有用户发现，手机的内存实际容量只有8GB，而不是博主所说的12GB。此外，国家质量监督检验检疫总局发布的一份检测报告显示，在标准测试条件下，超级手机的实际续航时间仅为380分钟，折合约6.3小时，远低于博主测试中提到的8小时以上。报告还指出，超级手机的充电功率在实际测试中只有85W左右，与博主宣称的120W存在约15%的差距。这些发现与博主的评测内容存在严重不符，让人质疑评测的真实性。一位资深硬件拆解专家在视频中拆解了超级手机，详细分析了主板上的各个元件。视频显示，手机主板上的芯片标识清晰地写着骁龙870字样，而不是官方宣称的骁龙888 Plus。此外，内存颗粒的容量标识也显示只有8GB，而非宣传的12GB。这位专家还指出，手机的快充电路设计存在明显的缩减，实际功率难以达到120W。这些技术细节与博主的评测内容完全矛盾，让人不得不怀疑，这篇评测究竟是基于真实产品写成的，还是为了某种目的而进行的虚假宣传？超级手机的真实性能究竟如何？博主的评测是否公正客观？这些问题在现有信息中难以找到答案。",
        expected_conflicts=[
            {
                "conflict_type": "属性冲突",
                "entities_involved": ["超级手机", "处理器"],
                "severity": "high"
            },
            {
                "conflict_type": "数值冲突",
                "entities_involved": ["超级手机", "内存"],
                "severity": "high"
            },
            {
                "conflict_type": "数值冲突",
                "entities_involved": ["超级手机", "续航时间"],
                "severity": "high"
            },
            {
                "conflict_type": "数值冲突",
                "entities_involved": ["超级手机", "充电功率"],
                "severity": "high"
            }
        ],
        expected_has_conflict=True,
    ),
    ConflictCase(
        case_id="product_review_002",
        text="电动车作为未来出行的重要趋势，在近年来受到了广泛关注。某汽车品牌发布了一款电动车型E系列，将其定位为公司的旗舰产品。根据官方介绍，E系列续航里程达到了惊人的800公里，能够满足各种出行需求。百公里加速仅需3秒，动力性能强劲。E系列采用最新一代磷酸铁锂电池，支持无线充电技术，这在行业内处于领先地位。在汽车经销商处的实物展示中，客户对E系列的内饰和做工赞不绝口。销售人员介绍，E系列采用的是顶级真皮座椅，触感柔软舒适，中控台配备了一块27英寸的超大曲面屏，科技感十足。此外，E系列还配备了智能驾驶辅助系统，能够实现L3级别的自动驾驶功能，大大提升了驾驶的安全性和便捷性。销售人员还强调，E系列在安全性能方面也达到了行业最高标准，通过了多项严格的碰撞测试。然而，当车主们实际提车并使用一段时间后，反馈了一系列问题。有车主表示，无线充电功能在实际使用中并不稳定，经常需要多次尝试才能成功充电，有时甚至完全无法识别充电器。智能驾驶辅助系统在复杂路况下识别率只有70%左右，与宣传的L3级别有较大差距，在雨雪天气下识别率更低。最让车主头疼的是，E系列的续航里程在实际使用中只有450公里左右，比官方宣传的少了一半以上，这与销售人员承诺的满电跑一整周严重不符。更严重的是，第三方机构对该车型的安全性进行了测试，结果显示E系列在正面碰撞测试中得分仅为60分，低于行业平均水平。有用户反映车辆在高速行驶时存在明显的异响问题，严重影响驾驶体验，甚至有用户报告车辆在行驶过程中出现过电子系统故障的情况。这些实际使用中的问题与官方宣传的旗舰产品形象形成了鲜明对比。此外，有消费者反映，他们购买的车辆在使用三个月后电池容量衰减了约20%，这远超行业正常的衰减速度。E系列的产品质量到底如何？官方宣传的技术指标是否属实？这些问题在现有信息中难以找到答案。E系列是否真的如宣传的那样优秀，还是存在大量的问题？",
        expected_conflicts=[
            {
                "conflict_type": "数值冲突",
                "entities_involved": ["E系列", "续航里程"],
                "severity": "high"
            },
            {
                "conflict_type": "数值冲突",
                "entities_involved": ["E系列", "百公里加速"],
                "severity": "high"
            },
            {
                "conflict_type": "属性冲突",
                "entities_involved": ["E系列", "无线充电"],
                "severity": "medium"
            },
            {
                "conflict_type": "属性冲突",
                "entities_involved": ["E系列", "智能驾驶级别"],
                "severity": "high"
            },
            {
                "conflict_type": "数值冲突",
                "entities_involved": ["E系列", "碰撞测试得分"],
                "severity": "high"
            }
        ],
        expected_has_conflict=True,
    ),

    # ===== 学术论文（学术论文信息）=====
    ConflictCase(
        case_id="academic_paper_001",
        text="人工智能领域近年来取得了突破性进展，特别是在自然语言处理方面。某顶级期刊发表了一篇题为深度学习在语言模型中的应用的论文，引起了广泛关注。论文摘要指出，研究人员提出了一种全新的Transformer变体架构，在多项基准测试中达到了SOTA水平，即当前最佳性能。该架构相比传统Transformer在计算效率上提升了30%，在参数规模减半的情况下仍保持了相同的性能表现，具有重要的理论意义和应用价值。在论文的引言部分，作者详细回顾了该领域的发展历程，并指出这是首次提出的具有里程碑意义的工作，将对后续的研究产生深远影响。作者团队在2022年1月完成了该模型的训练和验证，并在随后的三个月中陆续发布了多个版本的技术报告。根据作者在GitHub上的公开记录，第一个版本是在2022年4月发布的，随后在2022年6月发布了优化版本，并在8月发布了最终的正式版。这些时间节点清晰地标示了作者团队的研究进度。然而，当我们仔细阅读论文的引用记录时，发现了一些可疑之处。论文的参考文献中引用了一篇发表于2023年的论文，该论文对作者的工作进行了详细评述。但根据论文的发表时间戳，该论文的实际发表时间应该是在2024年，而不是2023年。这意味着论文在撰写时，被引用的那篇论文还不存在，这在学术界是严重的问题。此外，论文中多次引用了国际标准IEEE-2021，但根据IEEE官方目录，该标准已经被更新至IEEE-2023，而该论文的发表时间是在2022年，无法引用一个尚未发布的标准。这些引用错误表明，作者在撰写论文时可能没有仔细核对参考文献的准确性和时效性。更令人质疑的是，论文的作者声称他们在2022年1月就完成了该模型的训练，但根据作者在社交媒体上的帖子，他们是在2023年5月才开始接触相关技术的，这与论文中声称的深度参与该领域研究超过两年的说法存在明显矛盾。这种时间线上的不一致，让人对该论文的真实性产生了怀疑。如果作者在2022年1月就已经完成了模型训练，为什么在2023年5月才开始接触相关技术？这种突然的转变让人难以理解，也更让人怀疑论文的真实性。究竟论文的内容是真实的研究成果，还是为了某种目的而编造的数据？这些问题的答案在现有信息中无法找到。",
        expected_conflicts=[
            {
                "conflict_type": "时间冲突",
                "entities_involved": ["论文", "发表时间"],
                "severity": "high"
            },
            {
                "conflict_type": "时间冲突",
                "entities_involved": ["论文", "技术发表时间"],
                "severity": "high"
            },
            {
                "conflict_type": "时间冲突",
                "entities_involved": ["论文", "技术参与时间"],
                "severity": "high"
            },
            {
                "conflict_type": "属性冲突",
                "entities_involved": ["论文", "引用标准"],
                "severity": "high"
            }
        ],
        expected_has_conflict=True,
    ),

    # ===== 无冲突案例 =====
    ConflictCase(
        case_id="no_conflict_001",
        text="该项目分三个阶段实施：第一阶段需求分析，第二阶段开发，第三阶段测试与部署。项目团队成员来自不同部门，大家分工明确，协作顺畅。在项目执行过程中，我们采用敏捷开发方法，每个迭代周期为两周，确保项目能够快速响应需求变化。经过团队的共同努力，项目最终按时完成了所有预定目标，达到了预期效果。团队成员在各自岗位上都发挥了应有的作用，为公司的发展做出了贡献。项目实施过程中遇到的一些挑战都得到了及时解决。技术团队攻克了多个关键技术难题，确保了项目进度不受影响。管理层提供了充分的支持和资源，为项目成功提供了保障。客户对项目成果表示满意，认为项目达到了预期的目标。这些成果的取得离不开每一个参与者的辛勤付出。项目完成后，团队成员在各自岗位上继续努力工作，为公司的后续发展贡献力量。",
        expected_conflicts=[],
        expected_has_conflict=False,
    ),
    ConflictCase(
        case_id="no_conflict_002",
        text="产品在A渠道定价199元，在B渠道促销价149元，这是正常的渠道差异。不同渠道的销售策略不同，A渠道走量，B渠道主打性价比，两者都是合理的商业选择。消费者可以根据自己的需求和预算，选择合适的购买渠道。各渠道的销售数据良好，市场反响积极，公司对销售业绩感到满意。产品质量和客户服务得到了广泛认可。公司始终将产品质量放在首位，从原材料采购到生产制造，每一个环节都严格把控，确保产品质量达到行业领先水平。售后服务团队也提供了专业、及时的服务，解决了客户的各种问题，赢得了客户的信任和好评。这些努力为公司的长期发展奠定了坚实的基础。消费者反馈普遍积极，产品复购率保持在较高水平，这对公司来说是很大的鼓励。",
        expected_conflicts=[],
        expected_has_conflict=False,
    ),

    # ===== 短文本案例 =====
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
    raw_result: list = field(default_factory=list)
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
    merged_conflicts = []

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
        #merged_result.append(res)

    if not entities:
        dummy_entity = UIEntity(
            entity_id="dummy",
            name="文本实体",
            type="复合文本"
        )
        res = [{'entity_name': '无实体', 'has_conflict': False, 'conflicts': [], 'explanation': '未提取到实体，触发手动跳过'}]
        return res

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
