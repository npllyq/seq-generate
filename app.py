import streamlit as st
import random
import pyperclip
from typing import List, Tuple, Optional
from enum import Enum
import io
import re


# ==================== 配置与常量 ====================
class Charset(Enum):
    """字符集类型枚举"""
    CHAR = "字母 (A-Z)"
    NUMBER = "数字 (01-99)"
    CUSTOM = "自定义字符集 ⚡"


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

        # 使用原始算法
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
    if "last_preset_example" not in st.session_state:
        st.session_state.last_preset_example = "自定义输入"


def get_charset_config(charset_type: Charset, custom_chars: Optional[List[str]] = None) -> Tuple[List[str], int, int]:
    """获取字符集配置"""
    if charset_type == Charset.CHAR:
        return CHAR_LETTER, MIN_LIMIT, MAX_LIMIT_CHAR
    elif charset_type == Charset.NUMBER:
        return ORIGINAL_LETTER, MIN_LIMIT, MAX_LIMIT_NUMBER
    elif charset_type == Charset.CUSTOM:
        if custom_chars is None:
            raise ValueError("自定义字符集未提供")
        return custom_chars, MIN_LIMIT, len(custom_chars)
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

        if charset_type == Charset.CUSTOM:
            st.markdown("---")
            st.markdown("### 📝 自定义字符集")

            # 预设示例
            preset_examples = st.selectbox(
                "快速选择示例",
                options=["自定义输入", "字母示例", "数字示例"],
                index=0,
                help="选择示例可快速填充，也可手动输入"
            )

            # 示例映射
            preset_map = {
                "自定义输入": "",
                "字母示例": "A B E F G H I K",
                "数字示例": "01 02 04 05 06",
            }

            # 自动填充示例（当用户切换示例时）
            if preset_examples != st.session_state.last_preset_example:
                st.session_state.last_preset_example = preset_examples
                if preset_examples != "自定义输入":
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

        st.markdown("---")

        # 获取配置（仅在字符集有效时）
        try:
            if charset_type == Charset.CUSTOM:
                charset_data, min_val, max_val = get_charset_config(charset_type, custom_chars=custom_chars)
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
        st.markdown(f"- 元素数量: {limit}")

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