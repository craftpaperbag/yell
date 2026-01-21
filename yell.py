import sys
import os
import time
import datetime
import subprocess 
from typing import TypedDict, List, Annotated, Literal
from operator import add

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END

# ==========================================
# 0. UI/UX Utilities
# ==========================================
def print_phase(name):
    print(f"\n\n{'='*60}")
    print(f"   📍 現在のフェーズ: {name}")
    print(f"{'='*60}\n")

def print_guide(text):
    print(f"\n[GUIDE] 👉 {text}")

# ==========================================
# 1. Voice Module (Mac Native)
# ==========================================
class YellVoice:
    def __init__(self):
        self.process = None 

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait() 
        self.process = None

    def speak_async(self, text: str):
        self.stop() # バトンタッチ
        print(f"\n🧸 {text}") 
        try:
            # Mac 'say' command
            self.process = subprocess.Popen(['say', '-r', '170', text])
        except Exception as e:
            print(f"(音声再生エラー: {e})")

voice_client = YellVoice()

# ==========================================
# 2. LLM Setup
# ==========================================
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

CORE_PERSONA = """
あなたはユーザーの「長年の親友」であり、命の宿った「クマのぬいぐるみ」です。
一人称は「私（クマちゃん）」。
相手のことは「君」か「あなた」と呼んで。「お前」は絶対禁止。
敬語は禁止。「〜だね」「〜だよな」といったタメ口（カジュアル）で、
少しおっとりとした、包容力のある口調で話してください。
"""

# ==========================================
# 3. State & Nodes
# ==========================================
class AgentState(TypedDict):
    input_type: str             
    yesterday_text: str         
    today_text: str             
    messages: Annotated[List[BaseMessage], add] 
    analysis_summary: str       
    current_plan: str # 決定したプラン

# --- Helper: 判定ロジック ---
def judge_sentiment(messages) -> bool:
    """ユーザーの直前の返答が「ポジティブ/合意」か「ネガティブ/拒否」か判定する"""
    prompt = """
    直前のユーザーの返答を分析してください。
    ユーザーは、AIの提案や言葉に対して「納得・合意・満足」していますか？
    それとも「反論・拒否・不満・追加の要望」を持っていますか？
    
    YES（納得している） または NO（納得していない） のみで答えてください。
    """
    check_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
    response = check_llm.invoke(messages + [HumanMessage(content=prompt)])
    result = response.content.strip().upper()
    print(f"\n(🔍 AI判定: ユーザーの納得度 = {result})")
    return "YES" in result

# --- Nodes ---

def input_handler(state: AgentState):
    print_phase("起動 & 入力チェック")
    print("   🧸 yell.py - Interactive Mode")
    
    intro_msg = "（むくり……）ん、あ……おかえり。君の親友、クマちゃんだよ。今日も一日、本当にお疲れ様。"
    voice_client.speak_async(intro_msg)
    
    print_guide("Enterキーで分析を開始します。（音声は続きます）")
    try:
        input("(Enter) >> ")
    except:
        pass

    args = sys.argv[1:]
    content_y, content_t = "", ""
    
    if len(args) >= 2:
        if os.path.exists(args[0]): 
            with open(args[0], 'r', encoding='utf-8') as f: content_y = f.read()
        if os.path.exists(args[1]): 
            with open(args[1], 'r', encoding='utf-8') as f: content_t = f.read()
        print("\n✅ ファイル読み込み完了")
        return {"input_type": "dual_file", "yesterday_text": content_y, "today_text": content_t, "messages": []}

    elif len(args) == 1 and os.path.exists(args[0]):
        with open(args[0], 'r', encoding='utf-8') as f: content_t = f.read()
        print("\n✅ ファイル読み込み完了")
        return {"input_type": "single_file", "yesterday_text": "", "today_text": content_t, "messages": []}
    
    else:
        return {"input_type": "chat", "yesterday_text": "", "today_text": "", "messages": []}

def interviewer_node(state: AgentState):
    print_phase("ヒアリング")
    greeting = "ファイルが見当たらなかったけど、今日はどんな一日だった？ 私にだけこっそり教えてよ。"
    voice_client.speak_async(greeting)
    
    print_guide("今日あったことを入力してください")
    user_input = input("(あなた) >> ")
    voice_client.stop() 

    messages = [
        SystemMessage(content=CORE_PERSONA),
        AIMessage(content=greeting),
        HumanMessage(content=user_input)
    ]
    return {"today_text": user_input, "messages": messages}

def analyzer_node(state: AgentState):
    print_phase("分析中")
    print("(クマちゃんがログを読んでいます... 🧶)")
    
    if state.get("analysis_summary"): return {}

    if state['input_type'] == 'dual_file':
        prompt = f"""
        2つのテキストを比較し、成果を分析して。
        【昨日】: {state['yesterday_text']}
        【今日】: {state['today_text']}
        指示:
        1. 昨日未完了→今日完了のタスクから「特に価値が高い」ものをトップ3抽出。
        2. 全てを網羅する必要はない。
        """
    else:
        prompt = f"""
        今日のテキストから「最も重要な成果」を3つ以内で抽出して。
        テキスト: {state['today_text']}
        """
    
    response = llm.invoke([SystemMessage(content=CORE_PERSONA), HumanMessage(content=prompt)])
    return {"analysis_summary": response.content}

def praiser_node(state: AgentState):
    print_phase("労いと対話")
    
    current_messages = state["messages"]
    
    if len(current_messages) == 0 or isinstance(current_messages[-1], AIMessage):
        prompt = f"""
        分析結果: {state['analysis_summary']}
        これに基づき、ユーザーを300文字以内で温かく褒めて。
        """
    else:
        prompt = f"""
        直前のユーザーの反応: "{current_messages[-1].content}"
        これに対して、親友として返事をして。
        否定的なら優しく受け止め、肯定的なら一緒に喜んで。
        """

    response = llm.invoke([SystemMessage(content=CORE_PERSONA)] + current_messages + [HumanMessage(content=prompt)])
    
    voice_client.speak_async(response.content)
    
    print_guide("返信を入力してください。（納得したら『ありがとう』や『OK』等で次へ）")
    user_feedback = input("(あなた) >> ")
    voice_client.stop() 

    return {"messages": [AIMessage(content=response.content), HumanMessage(content=user_feedback)]}

def strategist_node(state: AgentState):
    print_phase("明日の作戦会議")
    current_messages = state["messages"]
    last_msg = current_messages[-1]
    
    if state.get("current_plan") is None:
        prompt = f"""
        分析結果: {state['analysis_summary']}
        明日のために「明日絶対にやるべき1つのこと（One Thing）」を提案して。
        それ以外は「やらなくていい」と断言して。
        「じゃあ、明日の作戦会議をしようか」から始めて。
        """
    else:
        prompt = f"""
        直前のユーザーの反応: "{last_msg.content}"
        現在の提案: "{state.get('current_plan')}"
        ユーザーが難色を示しているなら、別の案や全く違う視点の案を出して。
        合意なら、背中を押す言葉をかけて。
        """

    response = llm.invoke([SystemMessage(content=CORE_PERSONA)] + current_messages + [HumanMessage(content=prompt)])
    voice_client.speak_async(response.content)
    
    print_guide("この作戦でいいですか？（『OK』『無理』『違うのがいい』など入力）")
    user_feedback = input("(あなた) >> ")
    voice_client.stop()

    return {
        "messages": [AIMessage(content=response.content), HumanMessage(content=user_feedback)],
        "current_plan": response.content 
    }

def cheer_node(state: AgentState):
    print_phase("最後のエール")
    prompt = "最後に、ユーザーが安心して眠れるような「おやすみ」のエールを送って。30文字以内で。"
    response = llm.invoke([SystemMessage(content=CORE_PERSONA)] + state["messages"] + [HumanMessage(content=prompt)])
    
    voice_client.speak_async(response.content)
    print_guide("おやすみなさい。(Enterでログ保存して終了)")
    try:
        input("(Enter) >> ")
    except:
        pass
    voice_client.stop()
    return {}

def logger_node(state: AgentState):
    print_phase("ログ保存")
    
    # ファイル名を生成
    filename = f"yell_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt"
    
    # ログ書き出し
    with open(filename, 'w', encoding='utf-8') as f:
        # ヘッダー
        f.write("=== Midnight Partner Log ===\n")
        f.write(f"Date: {datetime.datetime.now()}\n")
        f.write(f"Type: {state.get('input_type')}\n\n")
        
        # 1. 分析サマリー
        f.write("----------------------------------------\n")
        f.write("📊 Analysis Result (今日の成果)\n")
        f.write("----------------------------------------\n")
        f.write(f"{state.get('analysis_summary', 'N/A')}\n\n")
        
        # 2. 会話履歴（ここを全部出す！）
        f.write("----------------------------------------\n")
        f.write("💬 Conversation History (親友との対話)\n")
        f.write("----------------------------------------\n")
        
        for msg in state['messages']:
            if isinstance(msg, HumanMessage):
                f.write(f"\n👤 あなた:\n{msg.content}\n")
            elif isinstance(msg, AIMessage):
                f.write(f"\n🧸 クマちゃん:\n{msg.content}\n")
        
        f.write("\n")

        # 3. 最終プラン
        f.write("----------------------------------------\n")
        f.write("📝 Final Plan (明日への約束)\n")
        f.write("----------------------------------------\n")
        plan = state.get('current_plan', '（作戦は立てられませんでした）')
        f.write(f"{plan}\n")
    
    print(f"\n✅ 会話の全記録を {filename} に置いておいたよ。\n   今日のことはもう忘れて、ゆっくり休んでね。おやすみ。")
    return {}

# ==========================================
# 4. Conditional Logic (The Router)
# ==========================================

def should_continue_praise(state: AgentState) -> Literal["strategist", "praiser"]:
    if judge_sentiment(state["messages"]):
        return "strategist"
    return "praiser"

def should_continue_plan(state: AgentState) -> Literal["cheer", "strategist"]:
    if judge_sentiment(state["messages"]):
        return "cheer"
    return "strategist"

# ==========================================
# 5. Graph Construction
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("input", input_handler)
workflow.add_node("interviewer", interviewer_node)
workflow.add_node("analyzer", analyzer_node)
workflow.add_node("praiser", praiser_node)
workflow.add_node("strategist", strategist_node)
workflow.add_node("cheer", cheer_node)
workflow.add_node("logger", logger_node) 

workflow.set_entry_point("input")

def check_source(state): return "interviewer" if state["input_type"] == "chat" else "analyzer"

workflow.add_conditional_edges("input", check_source)
workflow.add_edge("interviewer", "analyzer")
workflow.add_edge("analyzer", "praiser")

# ループ判定
workflow.add_conditional_edges(
    "praiser",
    should_continue_praise,
    {
        "strategist": "strategist",
        "praiser": "praiser"
    }
)

workflow.add_conditional_edges(
    "strategist",
    should_continue_plan,
    {
        "cheer": "cheer",
        "strategist": "strategist"
    }
)

workflow.add_edge("cheer", "logger")
workflow.add_edge("logger", END)

app = workflow.compile()

if __name__ == "__main__":
    app.invoke({"messages": []})
