import streamlit as st
import random
import pyperclip
from typing import List, Tuple
from enum import Enum
import io


# ==================== 配置与常量 ====================
class Charset(Enum):
    """字符集类型枚举"""
    CHAR = "字母 (A-Z)"
    NUMBER = "数字 (01-99)"


# 字符集定义
CHAR_LETTER = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
ORIGINAL_LETTER = [f"{i:02d}" for i in range(1, 100)]

# 算法参数
RANDOM_SEED_BASE = 42
MIN_LIMIT = 3
MAX_LIMIT_CHAR = 26
MAX_LIMIT_NUMBER = 99


# ==================== 核心算法模块 ====================
class SequenceGenerator:
    """序列生成器 - 基于图论分解算法（内部实现，前端不暴露术语）"""

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

    def generate(self, limit: int) -> List[str]:
        """
        生成循环序列路径

        Args:
            limit: 元素数量（需 ≥3）

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
        letters = self._shuffle(self.charset[:limit])
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


# ==================== Streamlit 应用 ====================
def initialize_session_state():
    """初始化会话状态"""
    if "refresh_counter" not in st.session_state:
        st.session_state.refresh_counter = 0
    if "last_result" not in st.session_state:
        st.session_state.last_result = []


def get_charset_config(charset_type: Charset) -> Tuple[List[str], int, int]:
    """获取字符集配置"""
    if charset_type == Charset.CHAR:
        return CHAR_LETTER, MIN_LIMIT, MAX_LIMIT_CHAR
    else:
        return ORIGINAL_LETTER, MIN_LIMIT, MAX_LIMIT_NUMBER


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
        # 检测常见环境限制
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
            key="charset_radio"
        )
        charset_type = Charset(charset_selection)

        # 获取配置
        charset_data, min_val, max_val = get_charset_config(charset_type)

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
        st.markdown(f"- 元素数量: {limit}")
        st.markdown(f"- 序列总数: {((limit - 1) // 2) * 2}")

    # ========== 主内容区 ==========
    # 设置随机种子（确保可重现性）
    seed = st.session_state.refresh_counter + RANDOM_SEED_BASE

    # 生成序列
    try:
        with st.spinner("生成序列中..."):
            generator = SequenceGenerator(charset_data, seed=seed)
            result = generator.generate(limit)
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
        # 复制按钮 - 恢复原设计思路
        if st.button("📋 复制所有序列", type="secondary", use_container_width=True):
            success, msg = safe_copy_to_clipboard(result_text)
            if success:
                st.toast(msg, icon="🎉")
            else:
                st.toast(msg, icon="⚠️")
                st.caption("💡 提示：您也可点击文本框右上角📋图标手动复制")

    with col2:
        # 下载按钮（可靠备用方案）
        buffer = io.BytesIO()
        buffer.write(result_text.encode('utf-8'))
        buffer.seek(0)
        st.download_button(
            "📥 下载序列 (.txt)",
            data=buffer,
            file_name=f"sequences_{limit}elements.txt",
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
        st.metric("字符集", charset_type.name.split('.')[-1])

    # 页脚说明（无专业术语）
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


if __name__ == "__main__":
    main()