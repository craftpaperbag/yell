import sys
import os
import time
import datetime
import pyttsx3
from typing import TypedDict, List, Annotated
from operator import add

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END

# ==========================================
# 1. 音声 & 演出モジュール (Voice & UX)
# ==========================================
class YellVoice:
    def __init__(self):
        pass  # 初期化時は何もしない

    def _get_engine(self):
        """その場限りのエンジンを作成して設定する"""
        engine = pyttsx3.init()

        # 音声設定（日本語を探す）
        try:
            voices = engine.getProperty('voices')
            for voice in voices:
                if "jp" in voice.id.lower() or "japan" in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
            engine.setProperty('rate', 180)
            engine.setProperty('volume', 1.0)
        except Exception:
            pass

        return engine

    def speak_and_print(self, text: str):
        """タイプライター表示 + 音声読み上げ"""
        print(f"\n🧸 {text}")

        try:
            # 毎回新しいエンジンを作る（これが重要！）
            engine = self._get_engine()
            engine.say(text)
            engine.runAndWait()
            # 使い終わったら変数を消しておく（念のため）
            del engine
        except Exception as e:
            print(f"(音声エラー: {e})")

# グローバルな音声インスタンス
voice_client = YellVoice()

# ==========================================
# 2. Gemini (LLM) セットアップ
# ==========================================
# APIキーは環境変数 GOOGLE_API_KEY から読み込み
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# 親友エージェントの基本人格
CORE_PERSONA = """
あなたはユーザーの「長年の親友（クマのぬいぐるみ）」であり、最高の理解者です。
一人称は「私」。相手のことは「君」か「あなた」と呼んで。「お前」は絶対禁止。
敬語は禁止。「〜だね」「〜だよな」といったタメ口（カジュアル）で話してください。
温かく、包み込むような口調で。
ユーザーは仕事や勉強で疲れているので、決して「もっと頑張れ」とは言わず、
「休む勇気」や「今日の成果」を認め、肯定することを最優先してください。
"""

# ==========================================
# 3. State (状態管理)
# ==========================================
class AgentState(TypedDict):
    input_type: str             # 'single_file', 'dual_file', 'chat'
    yesterday_text: str         # 昨日のTodoなど
    today_text: str             # 今日のTodoなど
    messages: Annotated[List[BaseMessage], add] # 会話履歴
    analysis_summary: str       # 分析結果
    plan_focus: str             # 明日の指針

# ==========================================
# 4. Nodes (処理ブロック)
# ==========================================

def input_handler(state: AgentState):
    """入力ファイルの判定"""
    args = sys.argv[1:]
    
    if len(args) >= 2:
        # ファイル2つ：比較モード
        path_yesterday = args[0]
        path_today = args[1]
        
        content_y = ""
        content_t = ""
        
        if os.path.exists(path_yesterday):
            with open(path_yesterday, 'r', encoding='utf-8') as f: content_y = f.read()
        if os.path.exists(path_today):
            with open(path_today, 'r', encoding='utf-8') as f: content_t = f.read()
            
        print("\n(昨日のメモと、今日のメモを読み込んだよ...)")
        return {"input_type": "dual_file", "yesterday_text": content_y, "today_text": content_t}

    elif len(args) == 1 and os.path.exists(args[0]):
        # ファイル1つ：単独分析モード
        with open(args[0], 'r', encoding='utf-8') as f:
            content = f.read()
        print("\n(今日のメモを読み込んだよ...)")
        return {"input_type": "single_file", "yesterday_text": "", "today_text": content}
    
    else:
        # ファイルなし：チャットモード
        return {"input_type": "chat", "yesterday_text": "", "today_text": ""}

def interviewer_node(state: AgentState):
    """ファイルがない場合の聞き取り"""
    greeting = "今日もお疲れ様。……ファイルが見当たらなかったけど、今日はどんな一日だった？ コーヒーでも飲みながら教えてよ。"
    voice_client.speak_and_print(greeting)
    
    user_input = input("\nあなた >> ")
    
    messages = [
        SystemMessage(content=CORE_PERSONA),
        AIMessage(content=greeting),
        HumanMessage(content=user_input)
    ]
    
    ack_msg = "うんうん、なるほどね。話してくれてありがとう。"
    voice_client.speak_and_print(ack_msg)
    
    # チャット内容をtoday_textとして扱う
    return {"today_text": user_input, "messages": messages}

def analyzer_node(state: AgentState):
    """テキストを分析し、成果と課題を抽出"""
    print("\n(考え中... 🧶)")
    
    if state['input_type'] == 'dual_file':
        # 比較分析
        prompt = f"""
        以下の2つのテキスト（Todoリストやメモ）を比較して分析して。
        
        【昨日のメモ（予定していたこと）】:
        {state['yesterday_text']}
        
        【今日のメモ（結果や現状）】:
        {state['today_text']}
        
        指示:
        1. 「昨日あった項目」で「今日完了になっている（または消し込まれている）」ものを探し出し、それを「偉大な成果」として認識して。
        2. たとえ完了していなくても、少しでも着手した形跡があれば評価して。
        3. ユーザーの疲れ具合も推測して。
        """
    else:
        # 単独分析
        prompt = f"""
        以下のテキストから、「完了したこと（成果）」と「未完了・気がかりなこと（課題）」を分析して。
        
        テキスト:
        {state['today_text']}
        """

    response = llm.invoke([SystemMessage(content=CORE_PERSONA), HumanMessage(content=prompt)])
    return {"analysis_summary": response.content}

def praiser_node(state: AgentState):
    """分析結果に基づいて、全力で褒める"""
    prompt = f"""
    分析結果: {state['analysis_summary']}
    
    上記を踏まえて、ユーザーを労い、褒めちぎってください。
    
    重要ルール:
    1. 「できていないこと」には触れない。「できたこと」だけにフォーカスする。
    2. 特に「昨日やろうとして、今日できたこと」があれば、それを具体的に挙げて「有言実行ですごい」と褒めて。
    3. クマのぬいぐるみのような温かさで。「〜だね」「えらいぞ」と優しく。
    """
    response = llm.invoke([SystemMessage(content=CORE_PERSONA), HumanMessage(content=prompt)])
    
    voice_client.speak_and_print(response.content)
    
    input("\n(Enterで次へ) >> ")
    return {"messages": [AIMessage(content=response.content)]}

def strategist_node(state: AgentState):
    """選択と集中を提案する"""
    prompt = f"""
    分析結果: {state['analysis_summary']}
    
    明日のために、ユーザーの心を軽くする提案をして。
    
    ルール：
    1. 「明日絶対にやるべき1つのこと（One Thing）」を提案する。小さなことでいい。
    2. それ以外は「明日はやらなくていい、忘れよう」と断言して、荷物を下ろさせる。
    3. 「じゃあ、明日の作戦会議をしようか」から始めて。
    """
    response = llm.invoke([SystemMessage(content=CORE_PERSONA), HumanMessage(content=prompt)])
    
    voice_client.speak_and_print(response.content)
    input("\n(Enterで合意) >> ")
    
    return {"plan_focus": response.content, "messages": [AIMessage(content=response.content)]}

def cheer_node(state: AgentState):
    """最後のエール"""
    prompt = "最後に、ユーザーが安心して眠れるような、短く温かい「おやすみ」のエールを送って。30文字以内で、温もりを感じさせて。"
    response = llm.invoke([SystemMessage(content=CORE_PERSONA), HumanMessage(content=prompt)])
    
    voice_client.speak_and_print(response.content)
    return {"messages": [AIMessage(content=response.content)]}

def logger_node(state: AgentState):
    """ログ保存"""
    filename = f"yell_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=== Midnight Partner Log ===\n")
        f.write(f"Type: {state.get('input_type')}\n")
        f.write(f"Plan: {state.get('plan_focus')}\n")
    
    print(f"\n✅ 会話の記録を {filename} に置いておいたよ。")
    return {}

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

def check_source(state):
    return "interviewer" if state["input_type"] == "chat" else "analyzer"

workflow.add_conditional_edges("input", check_source)
workflow.add_edge("interviewer", "analyzer")
workflow.add_edge("analyzer", "praiser")
workflow.add_edge("praiser", "strategist")
workflow.add_edge("strategist", "cheer")
workflow.add_edge("cheer", "logger")
workflow.add_edge("logger", END)

app = workflow.compile()

# ==========================================
# 6. Main Execution
# ==========================================
if __name__ == "__main__":
    print("---------------------------------------")
    print("   Midnight Partner (for You) 🧸🌙      ")
    print("---------------------------------------")
    
    # 実行
    app.invoke({"messages": []})
