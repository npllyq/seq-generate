import streamlit as st
import random
import pyperclip
from typing import List

# 数据定义
ORIGINAL_LETTER = [f'{i:02d}' for i in range(1, 100)]
CHAR_LETTER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def shuffle_array(arr: List[str]) -> List[str]:
    """随机打乱数组"""
    shuffled = arr.copy()
    for i in range(len(shuffled) - 1, 0, -1):
        j = random.randint(0, i)
        shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
    return shuffled

def generate_hamiltonian_decomposition(original_letters: List[str], limit: int) -> List[str]:
    """
    使用 Walecki 构造法生成哈密顿环分解
    """
    letters = original_letters[:limit]
    chars = shuffle_array(letters)
    
    center_index = 0
    center_node = chars[center_index]
    
    cycles = []
    base_cycles_count = (limit - 1) // 2
    
    for k in range(base_cycles_count):
        # 生成偏移量序列
        deltas = [0]
        s = 1
        while len(deltas) < limit - 1:
            deltas.append(-s)
            if len(deltas) < limit - 1:
                deltas.append(s)
            s += 1
        
        # 生成序列
        seq = []
        for d in deltas:
            circle_pos = ((k + d) % (limit - 1) + (limit - 1)) % (limit - 1)
            real_index = circle_pos + 1
            seq.append(chars[real_index])
        
        # 构建完整路径
        full_path = [center_node] + seq
        
        # 生成正向环
        directed_cycle_1 = full_path + [center_node]
        cycles.append(directed_cycle_1)
        
        # 生成反向环
        directed_cycle_2 = [center_node] + seq[::-1] + [center_node]
        cycles.append(directed_cycle_2)
    
    # 转换为字符串
    result = [">".join(cycle) for cycle in cycles]
    return result

# 初始化 session_state
if "refresh_counter" not in st.session_state:
    st.session_state.refresh_counter = 0

# Streamlit 应用配置
st.set_page_config(
    page_title="序列生成器",
    page_icon="🔄",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🔄 序列生成器")
st.markdown("生成自定义字符序列")

# 侧边栏 - 参数控制
st.sidebar.header("⚙️ 参数控制")

# 字符集类型选择
charset = st.sidebar.radio(
    "字符集类型",
    ["字母 (A-Z)", "数字 (01-99)"],
    index=0,
    key="charset"
)

# 根据字符集类型设置范围
if "字母" in charset:
    charset_value = "char"
    min_limit = 3
    max_limit = 26
else:
    charset_value = "number"
    min_limit = 3
    max_limit = 99

# 限制数量滑块
limit = st.sidebar.slider(
    "限制数量",
    min_value=min_limit,
    max_value=max_limit,
    value=10,
    step=1,
    key="limit"
)

# 生成按钮
if st.sidebar.button("🔄 刷新结果", type="primary", use_container_width=True, key="refresh_btn"):
    st.session_state.refresh_counter += 1

# 添加随机种子，确保每次刷新产生不同结果
random.seed(st.session_state.refresh_counter + 42)

# 显示当前参数
st.sidebar.markdown("---")
st.sidebar.markdown("**当前参数**:")
st.sidebar.markdown(f"- 字符集: {charset_value.upper()}")
st.sidebar.markdown(f"- 数量: {limit}")

# 生成序列
with st.spinner("正在生成序列..."):
    if charset_value == "char":
        result = generate_hamiltonian_decomposition(list(CHAR_LETTER), limit)
    else:
        result = generate_hamiltonian_decomposition(ORIGINAL_LETTER, limit)

# 结果展示
st.header("📊 生成结果")
st.markdown(f"共生成 **{len(result)}** 条序列")

# 显示结果文本框
result_text = "\n".join(result)

# 使用 text_area 显示并支持复制
st.text_area(
    "序列内容",
    value=result_text,
    height=300,
    key="result_text",
    help="点击复制按钮可复制所有序列",
    label_visibility="collapsed"
)

# 复制按钮
if st.button("📋 复制所有序列", type="secondary", use_container_width=True, key="copy_btn"):
    try:
        pyperclip.copy(result_text)
        st.toast("✅ 已复制到剪贴板！", icon="🎉")
    except Exception as e:
        st.toast(f"⚠️ 复制失败: {str(e)}", icon="⚠️")
        st.session_state.clipboard_text = result_text

# 统计信息
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.metric("总序列数", len(result))
with col2:
    st.metric("字符集类型", charset_value.upper())
with col3:
    st.metric("限制数量", limit)

# 页脚
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "💡 提示：点击文本框可手动选择复制，或点击「复制所有序列」按钮"
    "</div>",
    unsafe_allow_html=True
)
