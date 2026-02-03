import streamlit as st
import random
import pyperclip
from typing import List, Tuple, Optional
from enum import Enum
import io
import re
from collections import defaultdict


# ==================== 配置与常量 ====================
class Charset(Enum):
    """字符集类型枚举"""
    CHAR = "字母 (A-Z)"
    NUMBER = "数字 (01-99)"
    CUSTOM = "自定义字符集 ⚡"
    SPECIAL = "特殊字符集 🔮"  # 新增特殊字符集模式


# 预定义字符集
CHAR_LETTER = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
ORIGINAL_LETTER = [f"{i:02d}" for i in range(1, 100)]

# 预定义字符集大小限制
MAX_LIMIT_CHAR = len(CHAR_LETTER)
MAX_LIMIT_NUMBER = len(ORIGINAL_LETTER)

# 算法参数
RANDOM_SEED_BASE = 42
MIN_LIMIT = 3
MIN_CUSTOM_CHARS = 3  # 自定义字符集最小字符数
MAX_CUSTOM_CHARS = 200  # 自定义字符集最大字符数


# ==================== Walecki 构造法实现 ====================
def walecki_construction(nodes: List[str]) -> List[str]:
    """
    使用Walecki构造法为奇数个节点生成哈密顿路径
    适用于完全图的哈密顿分解

    Args:
        nodes: 节点列表，长度为奇数

    Returns:
        哈密顿路径列表
    """
    n = len(nodes)
    if n == 0:
        return []
    if n == 1:
        return [nodes[0]]
    if n == 2:
        return [nodes[0], nodes[1]]

    # Walecki构造法 - 适用于奇数个节点
    # 对于偶数个节点，我们将其视为奇数+1的问题，其中额外节点是虚拟的
    if n % 2 == 0:
        # 偶数情况：先处理前n-1个节点，然后插入最后一个节点
        path = walecki_construction(nodes[:-1])
        # 将最后一个节点插入到合适位置
        path.insert(1, nodes[-1])
        return path
    else:
        # 奇数情况：直接使用Walecki构造法
        path = [nodes[0]]  # 从第一个节点开始

        # 构造路径：交替从两边取节点
        left_idx = 1
        right_idx = n - 1
        take_left = True

        while left_idx <= right_idx:
            if take_left:
                path.append(nodes[left_idx])
                left_idx += 1
            else:
                path.append(nodes[right_idx])
                right_idx -= 1
            take_left = not take_left

        return path


def generate_walecki_cycles(nodes):
    """生成 Walecki 基础环"""
    num = len(nodes)
    chars = random.sample(nodes, k=num)
    center_node = chars[0]
    base_cycles_count = (num - 1) // 2
    cycles = []

    for k in range(base_cycles_count):
        deltas = [0]
        s = 1
        while len(deltas) < num - 1:
            deltas.append(-s)
            if len(deltas) < num - 1:
                deltas.append(s)
            s += 1

        seq = []
        for d in deltas:
            circle_pos = (k + d) % (num - 1)
            real_index = circle_pos + 1
            seq.append(chars[real_index])

        full_cycle = [center_node] + seq + [center_node]
        cycles.append(full_cycle)
    return cycles


def get_all_directed_paths(walecki_cycles):
    all_paths = []
    for cycle in walecki_cycles:
        base_fwd = cycle[:-1]
        base_rev = cycle[::-1][:-1]

        for path in [base_fwd, base_rev]:
            n = len(path)
            if n <= 1: continue

            # 生成所有旋转状态
            for shift in range(n):
                rotated = path[shift:] + path[:shift]
                all_paths.append(rotated)
    random.shuffle(all_paths)
    return all_paths


def stitch_groups_iteratively(group_ids, all_paths_dict, used_edges):
    current_chain = []
    current_chain_edges = set()

    # 随机打乱组顺序
    shuffled_groups = list(group_ids)
    random.shuffle(shuffled_groups)

    for i, gid in enumerate(shuffled_groups):
        candidates = all_paths_dict[gid]
        found_segment = None

        indices = list(range(len(candidates)))
        random.shuffle(indices)

        for idx in indices:
            path = candidates[idx]

            # 1. 边有效性检查 (首字母冲突、边冲突)
            valid = True

            # 检查桥接边 (如果是第一段则跳过)
            if i > 0:
                prev_tail = current_chain[-1][-1]
                curr_head = path[0]

                if prev_tail[0] == curr_head[0]:  # 首字母冲突
                    valid = False
                else:
                    bridge_edge = f"{prev_tail}>{curr_head}"
                    if bridge_edge in used_edges or bridge_edge in current_chain_edges:
                        valid = False

            # 检查内部边冲突
            if valid:
                for k in range(len(path) - 1):
                    e = f"{path[k]}>{path[k + 1]}"
                    if e in used_edges or e in current_chain_edges:
                        valid = False
                        break

            if valid:
                found_segment = path
                break

        if found_segment is None:
            return None

        current_chain.append(found_segment)

        # 收集边的占用情况
        if i > 0:
            # 记录上一段到这一段的桥接边
            prev_tail = current_chain[-2][-1]
            curr_head = found_segment[0]
            current_chain_edges.add(f"{prev_tail}>{curr_head}")

        # 录入内部边
        for k in range(len(found_segment) - 1):
            e = f"{found_segment[k]}>{found_segment[k + 1]}"
            current_chain_edges.add(e)

    # --- 修正后的闭合环逻辑 ---
    head_node = current_chain[0][0]
    tail_node = current_chain[-1][-1]

    # 1. 首字母闭合冲突
    if tail_node[0] == head_node[0]:
        return None

    # 2. 闭合边冲突
    closing_edge = f"{tail_node}>{head_node}"
    if closing_edge in used_edges or closing_edge in current_chain_edges:
        return None

    # --- 修正后的结果构建 ---
    full_cycle_nodes = []
    for i, p in enumerate(current_chain):
        # 修正点：直接 extend 拼接，不要手动 append p[0]
        # p[0] 自然会接在上一段的尾部后面
        full_cycle_nodes.extend(p)

    full_cycle_nodes.append(head_node)  # 最后补上闭合点

    return full_cycle_nodes


def run_large_scale_construction(nodes):
    print(f"开始处理总节点数: {len(nodes)}")

    groups = {}
    for node in nodes:
        key = node[-1]
        if key not in groups: groups[key] = []
        groups[key].append(node)

    group_ids = sorted(groups.keys())
    print(f"识别到分组: {group_ids} (共 {len(group_ids)} 组)")

    all_paths_dict = {}
    for gid, g_nodes in groups.items():
        cycles = generate_walecki_cycles(g_nodes)
        paths = get_all_directed_paths(cycles)
        all_paths_dict[gid] = paths
        print(f"组 {gid} ({len(g_nodes)}节点): 预生成候选路径 {len(paths)} 条")

    used_edges_global = set()
    final_res = []
    max_loops = 20
    fail_count = 0
    max_fails = 500

    while len(final_res) < max_loops and fail_count < max_fails:
        cycle = stitch_groups_iteratively(group_ids, all_paths_dict, used_edges_global)

        if cycle:
            final_res.append(cycle)
            for i in range(len(cycle) - 1):
                used_edges_global.add(f"{cycle[i]}>{cycle[i + 1]}")
            fail_count = 0
        else:
            fail_count += 1

    print(f"\n生成结束。共生成 {len(final_res)} 个有效的哈密顿环。")
    return final_res



# ==================== 核心算法模块 ====================
class SequenceGenerator:
    """序列生成器 - 基于图论分解算法"""

    def __init__(self, charset: List[str], seed: int = None):
        self.charset = charset
        if seed is not None:
            random.seed(seed)

    @staticmethod
    def _shuffle(arr: List[str]) -> List[str]:
        """Fisher-Yates 洗牌算法"""
        shuffled = arr.copy()
        for i in range(len(shuffled) - 1, 0, -1):
            j = random.randint(0, i)
            shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
        return shuffled

    @staticmethod
    def _get_letter_from_node(node: str) -> str:
        """从节点提取字母部分"""
        match = re.match(r'^([A-Za-z]+)', node)
        return match.group(1) if match else ""

    @staticmethod
    def _has_same_letter_adjacent(sequence_list: List[str]) -> bool:
        """检查序列中是否存在相邻节点包含相同字母"""
        for i in range(len(sequence_list) - 1):
            letter1 = SequenceGenerator._get_letter_from_node(sequence_list[i])
            letter2 = SequenceGenerator._get_letter_from_node(sequence_list[i + 1])
            if letter1 == letter2:
                return True
        return False

    def _extract_number_from_node(self, node: str) -> str:
        """从节点提取数字部分"""
        match = re.search(r'\d+', node)
        return match.group() if match else ""

    def generate(self, limit: int, charset_type: Charset = Charset.CHAR) -> List[str]:
        """
        生成循环序列路径

        Args:
            limit: 元素数量（需 ≥3）
            charset_type: 字符集类型

        Returns:
            序列列表，格式如 "A>B>C>A"

        Raises:
            ValueError: 当 limit 无效时
        """
        if limit < MIN_LIMIT:
            raise ValueError(f"元素数量必须 ≥ {MIN_LIMIT}")
        if limit > len(self.charset):
            raise ValueError(f"元素数量超过字符集大小 ({len(self.charset)})")

        # 选取并洗牌字符
        selected_chars = self._shuffle(self.charset[:limit])

        if charset_type == Charset.SPECIAL:
            # 特殊字符集：按数字分组，使用Walecki构造法
            results = run_large_scale_construction(self.charset)
            final_res = [">".join(res) for res in results]
            return final_res
        else:
            # 其他字符集类型使用原始算法
            letters = selected_chars
            center_node = letters[0]
            peripheral = letters[1:]
            n = len(peripheral)

            cycles = []
            base_cycles = (limit - 1) // 2

            for k in range(base_cycles):
                # 生成zigzag偏移序列: 0, -1, 1, -2, 2, ...
                deltas = [0]
                step = 1
                while len(deltas) < n:
                    deltas.extend([-step, step])
                    step += 1
                deltas = deltas[:n]

                # 构建循环路径
                seq = [peripheral[(k + d) % n] for d in deltas]
                forward = [center_node] + seq + [center_node]
                backward = [center_node] + seq[::-1] + [center_node]

                cycles.append(">".join(forward))
                cycles.append(">".join(backward))

            return cycles

    def _adjust_group_for_connection(self, group_seq: List[str], prev_letter: str) -> List[str]:
        """调整组序列以避免与前一个节点字母相同"""
        if len(group_seq) <= 1:
            return group_seq

        # 查找第一个字母不同的节点
        for i, node in enumerate(group_seq):
            current_letter = self._get_letter_from_node(node)
            if current_letter != prev_letter:
                # 将该节点移到前面
                new_seq = [group_seq[i]] + group_seq[:i] + group_seq[i + 1:]
                return new_seq

        # 如果所有节点字母都相同，返回原序列（这种情况理论上不应该发生）
        return group_seq

    def _generate_valid_special_sequence(self, selected_chars: List[str]) -> List[str]:
        """生成符合规则的特殊字符集序列"""
        # 按数字分组
        groups = defaultdict(list)
        for node in selected_chars:
            number_part = self._extract_number_from_node(node)
            groups[number_part].append(node)

        # 按数字大小排序组
        sorted_groups = []
        for number in sorted(groups.keys(), key=lambda x: int(x)):
            shuffled_nodes = self._shuffle(groups[number])
            walecki_path = walecki_construction(shuffled_nodes)
            sorted_groups.append(walecki_path)

        # 尝试构建无冲突的序列
        if not sorted_groups:
            return []

        final_sequence = sorted_groups[0][:]

        for group in sorted_groups[1:]:
            # 找到合适的插入点
            inserted = False
            for i in range(len(final_sequence)):
                if i == 0:
                    # 检查是否可以放在开头
                    if self._get_letter_from_node(final_sequence[0]) != self._get_letter_from_node(group[0]):
                        final_sequence = group + final_sequence
                        inserted = True
                        break
                elif i == len(final_sequence) - 1:
                    # 检查是否可以放在末尾
                    if self._get_letter_from_node(final_sequence[-1]) != self._get_letter_from_node(group[0]):
                        final_sequence = final_sequence + group
                        inserted = True
                        break
                else:
                    # 检查是否可以在中间某处插入
                    prev_letter = self._get_letter_from_node(final_sequence[i - 1])
                    next_letter = self._get_letter_from_node(final_sequence[i])
                    first_letter = self._get_letter_from_node(group[0])

                    if prev_letter != first_letter:
                        # 尝试将整个组插入到位置i
                        temp_seq = final_sequence[:i] + group + final_sequence[i:]
                        if not self._has_same_letter_adjacent(temp_seq):
                            final_sequence = temp_seq
                            inserted = True
                            break

            if not inserted:
                # 如果无法直接插入，尝试重新排列当前组
                for j in range(len(group)):
                    test_group = [group[j]] + group[:j] + group[j + 1:]
                    prev_letter = self._get_letter_from_node(final_sequence[-1])
                    first_letter = self._get_letter_from_node(test_group[0])

                    if prev_letter != first_letter:
                        final_sequence.extend(test_group)
                        inserted = True
                        break

            if not inserted:
                # 如果仍然无法插入，尝试更复杂的策略
                # 将当前组追加到末尾，并调整顺序
                for j in range(len(group)):
                    test_group = [group[j]] + group[:j] + group[j + 1:]
                    prev_letter = self._get_letter_from_node(final_sequence[-1])
                    first_letter = self._get_letter_from_node(test_group[0])

                    if prev_letter != first_letter:
                        final_sequence.extend(test_group)
                        break

        # 最后形成环
        if len(final_sequence) >= 2:
            last_letter = self._get_letter_from_node(final_sequence[-1])
            first_letter = self._get_letter_from_node(final_sequence[0])

            if last_letter != first_letter:
                final_sequence.append(final_sequence[0])
            else:
                # 如果首尾字母相同，需要特殊处理
                # 尝试移动一些节点来打破连续性
                for i in range(1, len(final_sequence) - 1):
                    mid_letter = self._get_letter_from_node(final_sequence[i])
                    if mid_letter != first_letter and mid_letter != last_letter:
                        # 将该节点移到首尾之间
                        new_seq = [final_sequence[0]] + [final_sequence[i]] + final_sequence[1:i] + final_sequence[
                                                                                                    i + 1:] + [
                                      final_sequence[0]]
                        if not self._has_same_letter_adjacent(new_seq):
                            final_sequence = new_seq
                            break
                        else:
                            # 如果不行，尝试其他方案
                            final_sequence = [final_sequence[0]] + final_sequence[1:i] + final_sequence[i + 1:] + [
                                final_sequence[i]] + [final_sequence[0]]
                            break
                else:
                    # 如果找不到合适的中间节点，简单地形成环
                    final_sequence.append(final_sequence[0])

        return final_sequence


# ==================== 特殊字符集验证 ====================
def validate_special_charset(input_text: str) -> Tuple[Optional[List[str]], Optional[str]]:
    """
    验证特殊字符集格式

    Args:
        input_text: 用户输入的文本

    Returns:
        (字符列表, 错误信息) - 成功时错误信息为None
    """
    if not input_text or not input_text.strip():
        return None, "请输入特殊字符集"

    # 使用正则表达式分割（支持空格、逗号(英文)、换行、Tab等分隔符）
    chars = re.split(r'[\s,\t\n;|]+', input_text.strip())

    # 过滤空字符串
    chars = [c.strip() for c in chars if c.strip()]

    # 检查最小字符数
    if len(chars) < MIN_CUSTOM_CHARS:
        return None, f"特殊字符集至少需要 {MIN_CUSTOM_CHARS} 个字符，当前只有 {len(chars)} 个"

    # 检查最大字符数
    if len(chars) > MAX_CUSTOM_CHARS:
        return None, f"特殊字符集最多支持 {MAX_CUSTOM_CHARS} 个字符，当前有 {len(chars)} 个"

    # 检查每个字符格式：必须是字母+数字格式
    invalid_format = []
    valid_chars = []
    for char in chars:
        # 检查是否符合字母+数字格式 (如A1, AB23, etc.)
        if re.match(r'^[A-Za-z]+\d+$', char):
            valid_chars.append(char)
        else:
            invalid_format.append(char)

    if invalid_format:
        return None, f"以下字符格式不正确（应为字母+数字格式，如A1, B23）: {', '.join(invalid_format[:5])}{'...' if len(invalid_format) > 5 else ''}"

    # 检查重复字符
    unique_chars = list(dict.fromkeys(valid_chars))  # 保持顺序去重
    if len(unique_chars) < len(valid_chars):
        duplicates = len(valid_chars) - len(unique_chars)
        st.warning(f"⚠️ 发现 {duplicates} 个重复字符，已自动去重")
        valid_chars = unique_chars

    return valid_chars, None


def display_special_charset_sample(chars: List[str]):
    """显示特殊字符集预览"""
    if len(chars) > 20:
        preview = " ".join(chars[:20]) + f" ... (共{len(chars)}个)"
    else:
        preview = " ".join(chars)

    st.info(f"🔮 当前特殊字符集: {preview}")


# ==================== 自定义字符集处理 ====================
def parse_custom_charset(input_text: str) -> Tuple[Optional[List[str]], Optional[str]]:
    """
    解析用户输入的自定义字符集

    Args:
        input_text: 用户输入的文本

    Returns:
        (字符列表, 错误信息) - 成功时错误信息为None
    """
    if not input_text or not input_text.strip():
        return None, "请输入自定义字符集"

    # 使用正则表达式分割（支持空格、逗号(英文)、换行、Tab等分隔符）
    chars = re.split(r'[\s,\t\n;|]+', input_text.strip())

    # 过滤空字符串
    chars = [c.strip() for c in chars if c.strip()]

    # 检查最小字符数
    if len(chars) < MIN_CUSTOM_CHARS:
        return None, f"自定义字符集至少需要 {MIN_CUSTOM_CHARS} 个字符，当前只有 {len(chars)} 个"

    # 检查最大字符数
    if len(chars) > MAX_CUSTOM_CHARS:
        return None, f"自定义字符集最多支持 {MAX_CUSTOM_CHARS} 个字符，当前有 {len(chars)} 个"

    # 检查重复字符
    unique_chars = list(dict.fromkeys(chars))  # 保持顺序去重
    if len(unique_chars) < len(chars):
        duplicates = len(chars) - len(unique_chars)
        # 显示警告但继续（自动去重）
        st.warning(f"⚠️ 发现 {duplicates} 个重复字符，已自动去重")
        chars = unique_chars

    # 检查每个字符长度（建议使用短字符）
    long_chars = [c for c in chars if len(c) > 3]
    if long_chars:
        st.warning(f"⚠️ 检测到 {len(long_chars)} 个长字符（>3字符），可能影响显示效果")

    return chars, None


def display_custom_charset_sample(chars: List[str]):
    """显示自定义字符集预览"""
    if len(chars) > 20:
        preview = " ".join(chars[:20]) + f" ... (共{len(chars)}个)"
    else:
        preview = " ".join(chars)

    st.info(f"📝 当前字符集: {preview}")


# ==================== Streamlit 应用 ====================
def initialize_session_state():
    """初始化会话状态"""
    if "refresh_counter" not in st.session_state:
        st.session_state.refresh_counter = 0
    if "last_result" not in st.session_state:
        st.session_state.last_result = []
    if "custom_charset_input" not in st.session_state:
        st.session_state.custom_charset_input = ""
    if "parsed_custom_charset" not in st.session_state:
        st.session_state.parsed_custom_charset = None
    if "custom_charset_error" not in st.session_state:
        st.session_state.custom_charset_error = None
    if "special_charset_input" not in st.session_state:
        st.session_state.special_charset_input = ""
    if "parsed_special_charset" not in st.session_state:
        st.session_state.parsed_special_charset = None
    if "special_charset_error" not in st.session_state:
        st.session_state.special_charset_error = None


def get_charset_config(charset_type: Charset, custom_chars: Optional[List[str]] = None,
                       special_chars: Optional[List[str]] = None) -> Tuple[List[str], int, int]:
    """获取字符集配置"""
    if charset_type == Charset.CHAR:
        return CHAR_LETTER, MIN_LIMIT, MAX_LIMIT_CHAR
    elif charset_type == Charset.NUMBER:
        return ORIGINAL_LETTER, MIN_LIMIT, MAX_LIMIT_NUMBER
    elif charset_type == Charset.CUSTOM:
        if custom_chars is None:
            raise ValueError("自定义字符集未提供")
        return custom_chars, MIN_LIMIT, len(custom_chars)
    elif charset_type == Charset.SPECIAL:
        if special_chars is None:
            raise ValueError("特殊字符集未提供")
        return special_chars, MIN_LIMIT, len(special_chars)
    else:
        raise ValueError(f"未知的字符集类型: {charset_type}")


def safe_copy_to_clipboard(text: str) -> Tuple[bool, str]:
    """
    安全复制到剪贴板

    Returns:
        (成功, 消息)
    """
    try:
        pyperclip.copy(text)
        return True, "✅ 已复制到剪贴板！"
    except Exception as e:
        error_msg = str(e).lower()
        if "cannot access clipboard" in error_msg or "pyperclip" in error_msg:
            return False, "⚠️ 复制功能受限（云环境限制），请手动复制文本框内容"
        return False, f"⚠️ 复制失败: {str(e)}"


def main():
    # 页面配置 - 必须在第一个 st 命令之前调用
    st.set_page_config(
        page_title="🔄 序列生成器",
        page_icon="🔄",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    # 初始化状态
    initialize_session_state()

    st.title("🔄 序列生成器")
    st.markdown("生成自定义循环路径序列")

    # ========== 侧边栏配置 ==========
    with st.sidebar:
        st.header("⚙️ 参数配置")

        # 字符集选择
        charset_selection = st.radio(
            "字符集类型",
            options=[c.value for c in Charset],
            index=0,
            key="charset_radio",
            help="选择预定义字符集或自定义"
        )
        charset_type = Charset(charset_selection)

        # 自定义字符集输入（仅在选择自定义时显示）
        custom_chars = None
        special_chars = None

        if charset_type == Charset.CUSTOM:
            st.markdown("---")
            st.markdown("### 📝 自定义字符集")

            # 预设示例
            preset_examples = st.selectbox(
                "快速选择示例",
                options=["自定义输入", "中文城市", "颜色名称"],
                index=0,
                help="选择示例可快速填充，也可手动输入"
            )

            # 示例映射
            preset_map = {
                "自定义输入": "",
                "中文城市": "北京 上海 广州 深圳 杭州 南京 武汉 成都 西安 重庆",
                "颜色名称": "red blue green yellow purple orange pink brown gray",
            }

            # 自动填充示例
            if preset_examples != "自定义输入" and not st.session_state.custom_charset_input:
                st.session_state.custom_charset_input = preset_map[preset_examples]

            # 文本输入区域
            custom_input = st.text_area(
                "输入自定义字符",
                value=st.session_state.custom_charset_input,
                height=150,
                placeholder="输入字符，用空格、逗号(英文)或换行分隔\n例如：A B C D E",
                help="支持空格、逗号(英文)、换行、Tab 等分隔符"
            )

            # 更新会话状态
            st.session_state.custom_charset_input = custom_input

            # 实时解析按钮
            if st.button("🔍 解析字符集", use_container_width=True, type="secondary"):
                parsed_chars, error_msg = parse_custom_charset(custom_input)
                if error_msg:
                    st.session_state.custom_charset_error = error_msg
                    st.session_state.parsed_custom_charset = None
                    st.error(f"❌ {error_msg}")
                else:
                    st.session_state.custom_charset_error = None
                    st.session_state.parsed_custom_charset = parsed_chars
                    st.success(f"✅ 成功解析 {len(parsed_chars)} 个字符")
                    display_custom_charset_sample(parsed_chars)

            # 显示解析结果
            if st.session_state.parsed_custom_charset:
                display_custom_charset_sample(st.session_state.parsed_custom_charset)
            elif st.session_state.custom_charset_error:
                st.error(f"⚠️ {st.session_state.custom_charset_error}")

            custom_chars = st.session_state.parsed_custom_charset

        elif charset_type == Charset.SPECIAL:
            st.markdown("---")
            st.markdown("### 🔮 特殊字符集")

            # 规则说明
            st.info("""
            **格式要求：**
            - 字母+数字格式 (如: A1, B2, X5)
            - 系统按数字分组：相同数字的节点在同一组，每组至少3个节点，否则无法生成
            - 相同字母前缀的节点不能相邻
            """)

            # 预设示例
            preset_examples = st.selectbox(
                "快速选择示例",
                options=["自定义输入", "示例1: A1 B1 A2 B2", "示例2: X1 Y1 Z1 A3 B3", "示例3: P1 Q1 R2 S2 T2"],
                index=0,
                help="选择示例可快速填充，也可手动输入"
            )

            # 示例映射
            preset_map = {
                "自定义输入": "",
                "示例1: A1 B1 A2 B2": "A1 B1 A2 B2",
                "示例2: X1 Y1 Z1 A3 B3": "X1 Y1 Z1 A3 B3",
                "示例3: P1 Q1 R2 S2 T2": "P1 Q1 R2 S2 T2",
            }

            # 自动填充示例
            if preset_examples != "自定义输入" and not st.session_state.special_charset_input:
                st.session_state.special_charset_input = preset_map[preset_examples]

            # 文本输入区域
            special_input = st.text_area(
                "输入特殊字符",
                value=st.session_state.special_charset_input,
                height=150,
                placeholder="输入字符，用空格、逗号(英文)或换行分隔\n例如：A1 B1 C1 A2 B2",
                help="格式：字母+数字 (如 A1, B2)，相同数字的节点会被分到同一组"
            )

            # 更新会话状态
            st.session_state.special_charset_input = special_input

            # 实时解析按钮
            if st.button("🔍 验证特殊字符集", use_container_width=True, type="secondary"):
                parsed_chars, error_msg = validate_special_charset(special_input)
                if error_msg:
                    st.session_state.special_charset_error = error_msg
                    st.session_state.parsed_special_charset = None
                    st.error(f"❌ {error_msg}")
                else:
                    st.session_state.special_charset_error = None
                    st.session_state.parsed_special_charset = parsed_chars
                    st.success(f"✅ 成功验证 {len(parsed_chars)} 个字符")

                    # 显示分组信息
                    groups = defaultdict(list)
                    for node in parsed_chars:
                        number_part = re.search(r'\d+', node)
                        if number_part:
                            groups[number_part.group()].append(node)

                    group_info = []
                    for number, nodes in sorted(groups.items(), key=lambda x: int(x[0])):
                        group_info.append(f"数字{number}: {', '.join(nodes)}")

                    st.info(f"**分组信息:**\n" + "\n".join(group_info))

            # 显示解析结果
            if st.session_state.parsed_special_charset:
                display_special_charset_sample(st.session_state.parsed_special_charset)

                # 显示分组详情
                groups = defaultdict(list)
                for node in st.session_state.parsed_special_charset:
                    number_part = re.search(r'\d+', node)
                    if number_part:
                        groups[number_part.group()].append(node)

                group_info = []
                for number, nodes in sorted(groups.items(), key=lambda x: int(x[0])):
                    group_info.append(f"数字{number}: {', '.join(nodes)}")

                if group_info:
                    st.info(f"**分组信息:**\n" + "\n".join(group_info))

            elif st.session_state.special_charset_error:
                st.error(f"⚠️ {st.session_state.special_charset_error}")

            special_chars = st.session_state.parsed_special_charset

        st.markdown("---")

        # 获取配置（仅在字符集有效时）
        try:
            if charset_type == Charset.CUSTOM:
                charset_data, min_val, max_val = get_charset_config(charset_type, custom_chars=custom_chars)
            elif charset_type == Charset.SPECIAL:
                charset_data, min_val, max_val = get_charset_config(charset_type, special_chars=special_chars)
            else:
                charset_data, min_val, max_val = get_charset_config(charset_type)
        except ValueError as e:
            st.error(f"❌ 字符集配置错误: {str(e)}")
            st.stop()

        # 数量选择
        limit = st.slider(
            "元素数量",
            min_value=min_val,
            max_value=max_val,
            value=min(10, max_val),
            step=1,
            help="序列中包含的元素数量（需 ≥3）"
        )

        # 刷新按钮
        if st.button("🔄 生成新序列", type="primary", use_container_width=True):
            st.session_state.refresh_counter += 1

        # 当前参数展示
        st.markdown("---")
        st.markdown("**当前配置**")
        st.markdown(f"- 字符集: {charset_type.value}")
        if charset_type == Charset.CUSTOM and custom_chars:
            st.markdown(f"- 可用字符: {len(custom_chars)} 个")
        elif charset_type == Charset.SPECIAL and special_chars:
            st.markdown(f"- 可用字符: {len(special_chars)} 个")
        st.markdown(f"- 元素数量: {limit}")
        if charset_type == Charset.SPECIAL:
            st.markdown(f"- 输出序列数: 1 (单个大环)")

    # ========== 主内容区 ==========
    # 设置随机种子（确保可重现性）
    seed = st.session_state.refresh_counter + RANDOM_SEED_BASE

    # 生成序列
    try:
        with st.spinner("生成序列中..."):
            generator = SequenceGenerator(charset_data, seed=seed)
            result = generator.generate(limit, charset_type)
            st.session_state.last_result = result
    except Exception as e:
        st.error(f"❌ 生成失败: {str(e)}")
        st.stop()

    # 结果展示
    st.header("📊 生成结果")
    st.markdown(f"共生成 **{len(result)}** 条循环序列")

    result_text = "\n".join(result)

    # 可复制文本区域
    st.text_area(
        "序列列表",
        value=result_text,
        height=350,
        key="result_display",
        label_visibility="collapsed"
    )

    # 操作按钮组
    col1, col2 = st.columns([1, 1])

    with col1:
        # 复制按钮
        if st.button("📋 复制所有序列", type="secondary", use_container_width=True):
            success, msg = safe_copy_to_clipboard(result_text)
            if success:
                st.toast(msg, icon="🎉")
            else:
                st.toast(msg, icon="⚠️")
                st.caption("💡 提示：您也可点击文本框右上角📋图标手动复制")

    with col2:
        # 下载按钮
        buffer = io.BytesIO()
        buffer.write(result_text.encode('utf-8'))
        buffer.seek(0)
        filename_prefix = "custom" if charset_type == Charset.CUSTOM else charset_type.name.lower()
        st.download_button(
            "📥 下载序列 (.txt)",
            data=buffer,
            file_name=f"sequences_{filename_prefix}_{limit}elements.txt",
            mime="text/plain",
            use_container_width=True
        )

    # 统计信息
    st.markdown("---")
    stats_col1, stats_col2, stats_col3 = st.columns(3)
    with stats_col1:
        st.metric("序列总数", len(result))
    with stats_col2:
        st.metric("元素数量", limit)
    with stats_col3:
        st.metric("字符集大小", len(charset_data))

    # 页脚说明
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.9em;'>
            <p>💡 每条序列以相同元素开始和结束，形成完整循环路径</p>
            <p>🔄 点击「生成新序列」可获得不同排列组合</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 特殊字符集使用提示
    if charset_type == Charset.SPECIAL:
        # 验证生成的序列是否符合规则
        if result:
            sequence_parts = result[0].split('>')
            has_violation = False
            violations = []
            for i in range(len(sequence_parts) - 1):
                letter1 = re.match(r'^([A-Za-z]+)', sequence_parts[i])
                letter2 = re.match(r'^([A-Za-z]+)', sequence_parts[i + 1])
                if letter1 and letter2 and letter1.group(1) == letter2.group(1):
                    has_violation = True
                    violations.append(f"'{sequence_parts[i]}' 和 '{sequence_parts[i + 1]}'")

            if has_violation:
                st.error(f"⚠️ 检测到规则违反: {'; '.join(violations[:3])}{'...' if len(violations) > 3 else ''}")
            else:
                st.success("✅ 所有相邻节点都符合规则（无相同字母前缀相邻）")

        st.info(
            """
            **🔮 特殊字符集算法说明**
            - 按数字部分分组：相同数字的节点分为一组 (如 A1, B1, C1 为一组)
            - 对每组使用Walecki构造法生成内部序列
            - 将各组序列首尾相连，形成一个大环
            - 确保相同字母前缀的节点不相邻
            """
        )

    # 自定义字符集使用提示
    if charset_type == Charset.CUSTOM:
        st.info(
            """
            **💡 自定义字符集提示**
            - 可以使用中文、英文、emoji等任意字符
            - 推荐使用短字符（1-3字符）以获得更好的显示效果
            - 重复字符会自动去重
            - 支持空格、逗号(英文)、换行等多种分隔符
            """
        )


if __name__ == "__main__":
    main()