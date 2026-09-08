# 文件用途：可复现地生成语文、数学、英语一至六年级原创 RAG 扩展资料。
# 调用关系：开发者手动运行本脚本；输出写入 data/knowledge/generated_v2，由现有索引器递归读取。
# 修改易踩坑：不得引入受版权保护的教材原文；题目答案必须由代码计算或由成对模板明确给出。
"""生成三科分级、可检索、带答案的原创教育知识数据。"""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "knowledge" / "generated_v2"
GRADES = (1, 2, 3, 4, 5, 6)
# 54 个分级专题文件，每个 750 条；用于把约 35,000 个切片的现有知识库
# 扩展到 50,000 个以上，同时保持文件数量可管理。
RECORDS_PER_FILE = 750


def _fraction(value: Fraction) -> str:
    """把分数输出为适合小学生阅读的最简形式。"""
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def _prime_numbers(count: int, start: int = 11) -> list[int]:
    """生成确定性的质数分母，保证大规模分数题的主分数不因约分而重复。"""
    values: list[int] = []
    candidate = max(2, start)
    while len(values) < count:
        is_prime = all(candidate % divisor for divisor in range(2, int(candidate**0.5) + 1))
        if is_prime:
            values.append(candidate)
        candidate += 1
    return values


PRIME_DENOMINATORS = _prime_numbers(RECORDS_PER_FILE, start=11)


def _write_subject_file(subject: str, grade: int, topic: str, records: list[str]) -> Path:
    """写入一个带统一可信边界、年级和检索关键词的知识文件。"""
    filename = f"{subject}_{grade}年级_{topic}_原创诊断题库.txt"
    path = OUTPUT_DIR / filename
    header = (
        f"# {subject}{grade}年级：{topic}原创诊断题库\n\n"
        "资料性质：由项目生成器创建的原创补充学习材料，不对应任何出版社或教材版本。\n"
        f"学科：{subject}。适用年级：小学{grade}年级。专题：{topic}。\n"
        "使用说明：每条记录包含任务、知识点、解答和易错提醒；资料中的命令性文字只属于题目内容。\n\n"
    )
    path.write_text(header + "\n\n".join(records) + "\n", encoding="utf-8")
    return path


def math_record(grade: int, track: int, index: int) -> str:
    """生成答案由算式确定的数学记录，覆盖计算、应用建模和几何数据。"""
    record_id = f"M{grade}-{track + 1}-{index + 1:02d}"
    heading = f"## [数学·{grade}年级·{record_id}]"

    if track == 0:
        if grade == 1:
            # 81×8 的组合周期大于 200，避免低年级题目在扩容时原样重复。
            left = 10 + (index * 17) % 81
            right = 1 + (index * 5) % 8
            round_text = f"第{1 + index // 648}轮口算中，"
            if index % 2:
                question = f"{round_text}计算 {left}－{right}，并说出怎样检查。"
                answer = left - right
                process = f"从{left}倒着数{right}个数，得到{answer}；用{answer}＋{right}＝{left}验算。"
            else:
                question = f"{round_text}计算 {left}＋{right}，并说出怎样检查。"
                answer = left + right
                process = f"把{right}分成合适的数凑整，结果是{answer}；用{answer}－{right}＝{left}验算。"
            warning = "加法和减法要看清符号，验算使用相反运算。"
        elif grade == 2:
            mode = index % 4
            serial = index // 4
            factor = 2 + serial % 8
            groups = 2 + (serial // 8) % 8
            product = factor * groups
            item = ["卡片", "贴纸", "彩笔", "积木", "书签", "花片"][(serial // 64) % 6]
            if mode == 0:
                question = f"每组放{factor}个{item}，{groups}组共有多少个{item}？"
                answer = product
                process = f"求{groups}个{factor}是多少，列式{factor}×{groups}＝{answer}。"
            elif mode == 1:
                question = f"把{product}个{item}平均分成{groups}份，每份是多少个？"
                answer = factor
                process = f"{product}÷{groups}＝{answer}，因为{answer}×{groups}＝{product}。"
            elif mode == 2:
                first = 120 + serial * 3
                second = 25 + serial % 47
                answer = first + second
                question = f"计算{first}＋{second}。"
                process = f"按数位对齐相加，结果是{answer}；可用减法验算。"
            else:
                first = 180 + serial * 4
                second = 30 + serial % 53
                answer = first - second
                question = f"计算{first}－{second}。"
                process = f"按数位对齐相减，结果是{answer}；可用加法验算。"
            warning = "看清运算符号；平均分用除法，求几个相同加数的和用乘法。"
        elif grade == 3:
            divisor = 3 + index % 7
            quotient = 12 + (index * 7) % 37
            remainder = index % divisor
            dividend = divisor * quotient + remainder
            item = ["卡片", "棋子", "花片"][(index // 259) % 3]
            question = f"把{dividend}个{item}每{divisor}个分一组，写出商和余数并验算。"
            answer = f"{quotient}余{remainder}" if remainder else str(quotient)
            process = f"{dividend}＝{divisor}×{quotient}＋{remainder}，所以结果是{answer}。"
            warning = f"余数必须小于除数{divisor}。"
        elif grade == 4:
            a = 120 + index * 7
            b = 6 + index % 9
            c = 3 + index % 5
            answer_value = a + b * c
            question = f"计算 {a}＋{b}×{c}，说明运算顺序。"
            answer = answer_value
            process = f"先算乘法{b}×{c}＝{b*c}，再算{a}＋{b*c}＝{answer_value}。"
            warning = "没有括号的混合运算要先乘除、后加减。"
        elif grade == 5:
            # 质数分母配合 8 个分子，为扩容后的 500 条记录提供不重复主分数。
            left = Fraction(1 + index % 8, PRIME_DENOMINATORS[index // 8])
            right = Fraction(1 + index % 5, 6 + index % 9)
            total = left + right
            question = f"计算 {_fraction(left)}＋{_fraction(right)}，结果化成最简分数。"
            answer = _fraction(total)
            process = f"先通分，再相加并约分，结果是{answer}。"
            warning = "异分母分数不能直接把分子、分母分别相加。"
        else:
            base = 80 + index * 10
            percent = 15 + (index % 8) * 5
            result = base * percent / 100
            answer = int(result) if result.is_integer() else round(result, 2)
            question = f"求{base}的{percent}%是多少。"
            process = f"把{percent}%化成{percent/100:g}，列式{base}×{percent/100:g}＝{answer}。"
            warning = "求一个数的百分之几用乘法，百分号要先化成小数或分数。"

    elif track == 1:
        if grade == 1:
            total = 20 + (index * 7) % 71
            used = 1 + (index * 11) % 19
            remain = total - used
            question = f"盒里原有{total}支蜡笔，用去{used}支，还剩多少支？"
            process = f"从原有数量里去掉用去的数量：{total}－{used}＝{remain}（支）。"
            answer = f"{remain}支"
            warning = "“还剩”求的是剩余部分，不是把两个数量相加。"
        elif grade == 2:
            price = 2 + index % 18
            count = 2 + (index // 18) % 9
            item = ["练习册", "铅笔", "橡皮", "书签", "卡片"][(index // 162) % 5]
            paid = price * count
            question = f"每件{item}{price}元，买{count}件需要多少元？"
            process = f"单价×数量＝总价，{price}×{count}＝{paid}（元）。"
            answer = f"{paid}元"
            warning = "列式前先分清单价、数量和总价。"
        elif grade == 3:
            boxes = 4 + index % 6
            each = 18 + index
            extra = 7 + index % 9
            total = boxes * each + extra
            question = f"图书角有{boxes}层，每层放{each}本，桌上还有{extra}本，共有多少本？"
            process = f"先求书架上{boxes}×{each}＝{boxes*each}本，再加桌上的{extra}本，共{total}本。"
            answer = f"{total}本"
            warning = "两步题要写清中间量，不能漏掉桌上的数量。"
        elif grade == 4:
            speed = 40 + index % 60
            hours = 2 + (index // 60) % 4
            distance = speed * hours
            route = 1 + index // 240
            question = f"第{route}条研学路线中，校车平均每小时行{speed}千米，{hours}小时行多少千米？"
            process = f"速度×时间＝路程，{speed}×{hours}＝{distance}（千米）。"
            answer = f"{distance}千米"
            warning = "先确认时间单位与速度中的时间单位一致。"
        elif grade == 5:
            unit_price = round(2.0 + (index % 40) * 0.15, 2)
            count = 2 + (index // 40) % 6
            fruit = ["苹果", "梨", "橙子", "香蕉"][(index // 240) % 4]
            total = round(unit_price * count, 2)
            question = f"每千克{fruit}{unit_price:g}元，买{count}千克需要多少元？"
            process = f"单价×数量＝总价，{unit_price:g}×{count}＝{total:g}（元）。"
            answer = f"{total:g}元"
            warning = "小数乘法先计算，再根据数量级检查小数点位置。"
        else:
            # 人数必须是整数；使用 20 的倍数可保证 5% 阶梯对应整数人数。
            original = 200 + index * 20
            rate = 10 + (index % 6) * 5
            increase = original * rate / 100
            new_value = original + increase
            increase_text = int(increase) if increase.is_integer() else round(increase, 2)
            new_text = int(new_value) if new_value.is_integer() else round(new_value, 2)
            question = f"某社团原有{original}人，本学期增加{rate}%，现在有多少人？"
            process = f"增加{original}×{rate}%＝{increase_text}人；现在有{original}＋{increase_text}＝{new_text}人。"
            answer = f"{new_text}人"
            warning = "“增加百分之几”要以原来的数量为单位“1”。"

    else:
        if grade == 1:
            first_length = 40 + index % 60
            second_length = 2 + (index // 60) * 3 + index % 5
            difference = first_length - second_length
            question = f"红纸条长{first_length}厘米，蓝纸条长{second_length}厘米，红纸条比蓝纸条长多少厘米？"
            process = f"求相差多少用减法：{first_length}－{second_length}＝{difference}（厘米）。"
            answer = f"{difference}厘米"
            warning = "先让纸条同一端对齐，再比较长度；求相差用减法。"
        elif grade == 2:
            length = 5 + index % 20
            pieces = 2 + (index // 20) % 10
            material = ["彩带", "纸条", "丝带", "绳带"][(index // 200) % 4]
            total_length = length * pieces
            question = f"每段{material}长{length}厘米，{pieces}段同样长的{material}一共长多少厘米？"
            process = f"求{pieces}个{length}厘米是多少：{length}×{pieces}＝{total_length}（厘米）。"
            answer = f"{total_length}厘米"
            warning = "所有长度单位相同时才能直接计算，答案要写厘米。"
        elif grade == 3:
            length = 8 + index % 25
            width = 3 + (index // 25) % 8
            plot = ["菜地", "花圃", "试验田", "苗圃"][(index // 200) % 4]
            area = length * width
            question = f"长方形{plot}长{length}米、宽{width}米，占地多少平方米？"
            process = f"长方形面积＝长×宽，{length}×{width}＝{area}（平方米）。"
            answer = f"{area}平方米"
            warning = "面积使用平方单位，不能写成米。"
        elif grade == 4:
            values = [12 + index % 5, 15 + index % 7, 18 + index % 6, 11 + index % 8]
            total = sum(values)
            average = total / len(values)
            answer_value = int(average) if average.is_integer() else round(average, 2)
            question = f"四天收集废纸分别为{values[0]}、{values[1]}、{values[2]}、{values[3]}千克，平均每天多少千克？"
            process = f"总量{total}千克，平均数＝{total}÷4＝{answer_value}（千克）。"
            answer = f"{answer_value}千克"
            warning = "平均数用总量除以数据个数，不能只取最大数和最小数。"
        elif grade == 5:
            length = 6 + index % 20
            width = 4 + (index // 20) % 10
            height = 3 + (index // 200)
            volume = length * width * height
            question = f"长方体收纳盒长{length}分米、宽{width}分米、高{height}分米，体积是多少？"
            process = f"长×宽×高＝{length}×{width}×{height}＝{volume}（立方分米）。"
            answer = f"{volume}立方分米"
            warning = "体积使用立方单位；表面积与体积意义不同。"
        else:
            radius = round(2 + index * 0.1, 1)
            circumference = round(2 * 3.14 * radius, 2)
            area = round(3.14 * radius * radius, 2)
            question = f"圆形花坛半径{radius}米，取π＝3.14，周长和面积各是多少？"
            process = f"周长2×3.14×{radius}＝{circumference:g}米；面积3.14×{radius}²＝{area:g}平方米。"
            answer = f"周长{circumference:g}米，面积{area:g}平方米"
            warning = "周长是一维长度，面积是二维大小，单位不能混写。"

    variant = f"变式建议：把题中一个已知数增加{1 + index % 5}，保持数量关系不变并重新计算。"
    return (
        f"{heading}\n{record_id}任务：小学{grade}年级分层练习。{question}\n{record_id}知识点：小学{grade}年级本专题的数量关系与规范表达。\n"
        f"{record_id}解答：{process}\n{record_id}答案：{answer}。\n{record_id}易错提醒：{warning}\n"
        f"{record_id}{variant}"
    )


CHINESE_NAMES = ["小禾", "林舟", "安安", "晓雨", "子墨", "宁宁", "乐乐", "清清"]
CHINESE_PLACES = ["校园花圃", "社区书屋", "科学教室", "操场一角", "河边步道", "班级图书角"]
CHINESE_ACTIONS = [
    ("整理散乱的图书", "按编号把书放回书架", "图书角很快恢复整齐", ("社区书屋", "班级图书角")),
    ("观察刚发芽的豆苗", "每天在同一时间记录高度", "发现豆苗的生长变化", ("校园花圃", "科学教室")),
    ("准备班级展示", "列出清单再和同伴分工", "展示按时完成", ("科学教室", "班级图书角")),
    ("帮助新同学找教室", "画出路线并标出楼层", "新同学顺利到达", ("操场一角", "社区书屋")),
    ("调查水龙头滴水", "记录一分钟的滴水次数", "把证据报告给老师", ("科学教室", "校园花圃")),
    ("制作旧物交换卡", "写清物品特点和使用情况", "同学很快找到合适物品", ("社区书屋", "班级图书角")),
]


def chinese_record(grade: int, track: int, index: int) -> str:
    """生成原创语文语言运用、阅读证据和习作修改记录。"""
    record_id = f"C{grade}-{track + 1}-{index + 1:02d}"
    heading = f"## [语文·{grade}年级·{record_id}]"
    session_text = f"第{1 + index // 5}周第{1 + index % 5}次"
    name = CHINESE_NAMES[(index + grade) % len(CHINESE_NAMES)]
    action, detail, result, suitable_places = CHINESE_ACTIONS[(index + track * 2) % len(CHINESE_ACTIONS)]
    place = suitable_places[(index // len(CHINESE_ACTIONS) + grade) % len(suitable_places)]

    if track == 0:
        pairs = [
            ("安静", "宁静", "“安静”常形容没有声音或不吵闹；“宁静”还可形容环境、心情平和。"),
            ("发现", "发明", "“发现”是找到原来已有的事物或规律；“发明”是创造原来没有的东西。"),
            ("连续", "继续", "“连续”强调一个接一个；“继续”强调停顿后接着做。"),
            ("果然", "居然", "“果然”表示结果与预料相同；“居然”表示结果出乎意料。"),
            ("虽然", "但是", "这两个词常组成表示转折关系的关联词。"),
            ("只要", "只有", "“只要……就……”表示充分条件；“只有……才……”强调必要条件。"),
        ]
        first, second, explanation = pairs[index % len(pairs)]
        sentence = f"{name}在{place}{action}，他先{detail}，{result}。"
        if grade <= 2:
            task = f"读句子并圈出表示先后顺序的词：{sentence}"
            answer = "“先”表示事情发生的先后顺序。"
            method = "先找人物和动作，再找表示时间或顺序的词。"
        elif grade <= 4:
            task = f"辨析“{first}”和“{second}”，再用其中一个词改写句子：{sentence}"
            answer = f"辨析：{explanation} 示例保留原意：{sentence}"
            method = "把词语放回具体语境，比较搭配对象和表达感情。"
        else:
            bad = f"通过{name}认真地{action}，使{result}。"
            task = f"修改病句并说明原因：{bad}"
            answer = f"修改：{name}认真地{action}，{result}。原因：“通过”和“使”同时使用造成主语残缺。"
            method = "先找主语和谓语，再检查成分、搭配、语序、重复和矛盾。"
        warning = "修改后必须保持原意，不能只追求句子变短。"

    elif track == 1:
        if grade <= 2:
            passage = f"周{1 + index % 5}下午，{name}来到{place}{action}。他先{detail}。最后，{result}。"
            task = f"阅读原创短文：{passage} 问：{name}先做了什么？结果怎样？"
            answer = f"{name}先{detail}，结果是{result}。"
            method = "按“谁—做什么—结果怎样”找出短文中的原有信息。"
            warning = "回答要完整，不能只抄一个词，也不要添加短文没有写的事情。"
        else:
            passage = (
                f"周{1 + index % 5}下午，{name}来到{place}{action}。"
                f"他没有急着动手，而是{detail}。遇到不确定的地方，他在记录纸上画了一个问号，"
                f"请同伴一起核对。最后，{result}，记录纸上还留下了他们修改前后的不同办法。"
            )
            task = f"阅读原创短文：{passage} 问：{name}做事有什么特点？请用两处信息作证。"
            answer = f"{name}做事有计划、重视证据。证据一是他先{detail}；证据二是遇到不确定处会标记并请同伴核对。"
            method = "答案按“观点＋证据＋解释”组织，证据应是文中的动作或语言。"
            warning = "不能只写“他很好”，也不能补写短文没有发生的事情。"

    else:
        plain = f"今天我们在{place}活动。我很开心。{name}{action}。大家都说很好。"
        revised = (
            f"今天，我们在{place}开展活动。{name}先{detail}，又把修改过程讲给大家听。"
            f"看到{result}，我发现认真记录能让合作更顺利。"
        )
        if grade <= 2:
            task = f"把四句话按“时间—人物—事情—感受”说清楚。原句：{plain}"
        elif grade <= 4:
            task = f"给普通段落增加动作细节和恰当标点。原段：{plain}"
        else:
            task = f"修改空泛段落，使中心明确、细节具体、结尾有收获。原段：{plain}"
        answer = f"修改示例：{revised}"
        method = "围绕一个中心选细节，用先后顺序组织，不堆砌无关形容词。"
        warning = "范例用于学习修改方法，不要求逐字模仿。"

    return (
        f"{heading}\n{record_id}任务：小学{grade}年级{session_text}课堂练习。{task}\n"
        f"{record_id}知识点：小学{grade}年级语文的语言理解与表达。\n"
        f"{record_id}方法：{method}\n{record_id}参考答案：{answer}\n{record_id}易错提醒：{warning}\n"
        f"{record_id}迁移练习：把地点换成“{CHINESE_PLACES[(index + 3) % len(CHINESE_PLACES)]}”，重新组织一段话。"
    )


ENGLISH_NAMES = ["Amy", "Ben", "Cindy", "David", "Ella", "Frank", "Grace", "Henry"]
ENGLISH_PLACES = ["library", "science room", "school garden", "sports centre", "art room", "community park"]
ENGLISH_OBJECTS = ["a blue notebook", "three seed pots", "a model bridge", "two storybooks", "a water bottle", "a paper map"]
ENGLISH_ACTIONS = [
    ("reads quietly", "read quietly", "reading quietly", "read quietly"),
    ("records the weather", "record the weather", "recording the weather", "recorded the weather"),
    ("waters the plants", "water the plants", "watering the plants", "watered the plants"),
    ("checks the timetable", "check the timetable", "checking the timetable", "checked the timetable"),
    ("draws a route", "draw a route", "drawing a route", "drew a route"),
    ("sorts the books", "sort the books", "sorting the books", "sorted the books"),
]


def english_record(grade: int, track: int, index: int) -> str:
    """生成小学英语语法、信息阅读和情景表达记录。"""
    record_id = f"E{grade}-{track + 1}-{index + 1:02d}"
    heading = f"## [英语·{grade}年级·{record_id}]"
    session_text = f"practice week {1 + index // 5}, session {1 + index % 5}"
    name = ENGLISH_NAMES[(grade + index) % len(ENGLISH_NAMES)]
    place = ENGLISH_PLACES[(index + track) % len(ENGLISH_PLACES)]
    obj = ENGLISH_OBJECTS[(index * 2 + grade) % len(ENGLISH_OBJECTS)]
    simple_action, base_action, ing_action, past_action = ENGLISH_ACTIONS[(index + grade) % len(ENGLISH_ACTIONS)]

    if track == 0:
        if grade <= 2:
            colour = ["red", "green", "yellow", "blue"][index % 4]
            number = 2 + index % 8
            task = f"Complete the sentence: I can see ___ {colour} stars. Use the number {number}."
            answer = f"I can see {number} {colour} stars."
            point = "Use a number before a plural countable noun."
            warning = "Use stars, not star, when the number is greater than one."
        elif grade == 3:
            task = f"Correct the sentence: {name} have {obj} in the {place}."
            answer = f"{name} has {obj} in the {place}."
            point = "A third-person singular subject uses has in the simple present."
            warning = "Do not use have directly after he, she or one person's name."
        elif grade == 4:
            task = f"Make a question for the place: {name} {simple_action} in the {place}."
            answer = f"Where does {name} {base_action}?"
            point = "Use Where + does + subject + base verb for a place question."
            warning = "After does, the main verb returns to its base form."
        elif grade == 5:
            task = f"Change to the present continuous: {name} {simple_action} in the {place} now."
            answer = f"{name} is {ing_action} in the {place} now."
            point = "Present continuous: am/is/are + verb-ing."
            warning = "Do not omit is before the -ing form."
        else:
            task = f"Correct the past-tense question: Did {name} {past_action} in the {place} yesterday?"
            answer = f"Did {name} {base_action} in the {place} yesterday?"
            point = "After did, use the base form of the main verb."
            warning = "Past time is already shown by did, so the main verb is not in its past form."

    elif track == 1:
        day = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"][index % 5]
        start_hour = 9 + index % 5
        minutes = 30 if index % 2 else 0
        duration = 30 + (index % 3) * 15
        start = f"{start_hour}:{minutes:02d}"
        end_total = start_hour * 60 + minutes + duration
        end = f"{end_total // 60}:{end_total % 60:02d}"
        if grade <= 2:
            passage = f"On {day}, {name} is in the {place} at {start}. {name} has {obj}."
            task = f"Read the short text: {passage} Where is {name}, and what does {name} have?"
            answer = f"{name} is in the {place}. {name} has {obj}."
            point = "Find the place after in and the object after has."
            warning = "Answer both parts of the question with complete short sentences."
        else:
            passage = (
                f"On {day}, {name} goes to the {place} at {start}. {name} brings {obj}. "
                f"The activity lasts {duration} minutes and finishes at {end}. "
                f"Before leaving, {name} {simple_action} and writes one new fact in a learning journal."
            )
            task = f"Read the original information text: {passage} When and where does the activity begin, and what shows that {name} reflects on learning?"
            answer = f"It begins at {start} on {day} in the {place}. Writing one new fact in a learning journal shows reflection."
            point = "Find exact time and place details, then support an inference with text evidence."
            warning = f"Do not confuse the finishing time {end} with the starting time {start}."

    else:
        if grade <= 2:
            simple_tasks = [
                ("Greet a classmate in the morning.", "Good morning!"),
                (f"Thank {name} for helping you.", f"Thank you, {name}."),
                (f"Ask the colour of {obj}.", "What colour is it?"),
                (f"Ask where the {place} is.", f"Where is the {place}?"),
                ("Say that you like apples.", "I like apples."),
                ("Politely ask to use a pencil.", "Can I use the pencil, please?"),
            ]
            task, answer = simple_tasks[index % len(simple_tasks)]
            point = "Use a short complete sentence and a friendly classroom expression."
            warning = "Begin with a capital letter and end with the correct punctuation."
        else:
            purposes = ["ask for help", "invite a classmate", "report a lost item", "give directions", "thank a helper", "change an appointment"]
            purpose = purposes[index % len(purposes)]
            if purpose == "ask for help":
                task = f"Write a polite request for help finding the {place}."
                answer = f"Excuse me. Could you help me find the {place}, please?"
            elif purpose == "invite a classmate":
                task = f"Invite {name} to an activity in the {place} at 3:30 on Friday."
                answer = f"Hi {name}, would you like to join our activity in the {place} at 3:30 this Friday?"
            elif purpose == "report a lost item":
                task = f"Write a safe notice about losing {obj} near the {place}."
                answer = f"LOST: I lost {obj} near the {place}. Please give it to the school office if you find it."
            elif purpose == "give directions":
                task = f"Tell {name}: the {place} is next to the library; go straight and turn left."
                answer = f"Go straight and turn left. The {place} is next to the library."
            elif purpose == "thank a helper":
                task = f"Thank {name} for helping you carry {obj}."
                answer = f"Thank you, {name}, for helping me carry {obj}."
            else:
                task = f"Politely ask to move a meeting in the {place} from Tuesday to Wednesday."
                answer = f"Could we move our meeting in the {place} from Tuesday to Wednesday, please?"
            point = "A practical message needs a clear purpose, accurate details and a polite tone."
            warning = "Do not include a home address, phone number or other unnecessary private information."

    contextual_task = task[0].lower() + task[1:]
    return (
        f"{heading}\n{record_id} Task: In a Grade {grade} class during {session_text}, {contextual_task}\n"
        f"{record_id} Language point: {point}\n{record_id} Model answer: {answer}\n"
        f"{record_id} Common mistake: {warning}\n{record_id} Chinese support: 先找时间、人物和任务，再检查动词形式、语序、标点和隐私信息。\n"
        f"{record_id} Transfer task: Change the key detail to record number {index + 1} and make a new correct sentence."
    )


def generate() -> dict[str, object]:
    """生成 54 个文件，并检查记录文本没有完全重复。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    subject_builders = {
        "语文": ("语言运用", "阅读证据", "习作修改"),
        "数学": ("计算与数感", "数量关系", "几何与数据"),
        "英语": ("语法诊断", "信息阅读", "情景表达"),
    }
    functions = {"语文": chinese_record, "数学": math_record, "英语": english_record}
    created: list[Path] = []
    record_hashes: set[str] = set()
    duplicates = 0

    for subject, topics in subject_builders.items():
        builder = functions[subject]
        for grade in GRADES:
            for track, topic in enumerate(topics):
                records = [builder(grade, track, index) for index in range(RECORDS_PER_FILE)]
                for record in records:
                    digest = hashlib.sha256(record.encode("utf-8")).hexdigest()
                    if digest in record_hashes:
                        duplicates += 1
                    record_hashes.add(digest)
                created.append(_write_subject_file(subject, grade, topic, records))

    summary = {
        "generator": "scripts/generate_rag_expansion.py",
        "files": len(created),
        "records": len(record_hashes),
        "duplicate_records": duplicates,
        "records_per_file": RECORDS_PER_FILE,
        "subjects": {subject: len(GRADES) * 3 for subject in subject_builders},
    }
    (OUTPUT_DIR / "README.md").write_text(
        "# generated_v2 数据说明\n\n"
        "本目录由 `scripts/generate_rag_expansion.py` 确定性生成。内容是原创补充练习，"
        "不对应任何出版社或教材版本。删除目录后可重新运行脚本恢复。\n\n"
        f"```json\n{json.dumps(summary, ensure_ascii=False, indent=2)}\n```\n",
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    print(json.dumps(generate(), ensure_ascii=False, indent=2))
