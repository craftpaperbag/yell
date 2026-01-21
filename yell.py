import sys
import os
import time
import datetime
import pyttsx3
import threading
from typing import TypedDict, List, Annotated
from operator import add

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END

# ==========================================
# 0. ユーティリティ (UI/UX)
# ==========================================
def print_phase(name):
    """現在のノード（フェーズ）を目立たせる"""
    print(f"\n\n{'='*60}")
    print(f"   📍 現在のフェーズ: {name}")
    print(f"{'='*60}\n")

def print_guide(text):
    """ユーザーへの入力ガイドを表示"""
    print(f"\n[GUIDE] 👉 {text}")

# ==========================================
# 1. 音声 & 演出モジュール (Voice & UX)
# ==========================================
class YellVoice:
    def __init__(self):
        self.current_engine = None
        self.speaking_thread = None 
        self.lock = threading.Lock() 

    def _speak_thread_func(self, text):
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            for voice in voices:
                if "jp" in voice.id.lower() or "japan" in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
            engine.setProperty('rate', 160) 
            engine.setProperty('volume', 1.0)
            
            self.current_engine = engine
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass
        finally:
            self.current_engine = None

    def stop(self):
        if self.current_engine:
            try:
                self.current_engine.stop()
            except:
                pass
        if self.speaking_thread and self.speaking_thread.is_alive():
            self.speaking_thread.join() 

    def speak_async(self, text: str):
        with self.lock:
            self.stop()
            time.sleep(0.3)
            print(f"\n🧸 {text}") 
            t = threading.Thread(target=self._speak_thread_func, args=(text,))
            t.daemon = True 
            self.speaking_thread = t
            t.start()

voice_client = YellVoice()

# ==========================================
# 2. Gemini (LLM) セットアップ
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
    plan_focus: str             

def input_handler(state: AgentState):
    """起動時の演出と入力判定"""
    print_phase("起動 & 入力チェック (Input Handler)")
    
    print("   🧸 yell.py - Midnight Partner Demo")
    
    intro_msg = "（むくり……）ん、あ……おかえり。君の親友、クマちゃんだよ。今日も一日、本当にお疲れ様。"
    voice_client.speak_async(intro_msg)
    
    print_guide("クマちゃんが起きました。Enterキーを押して分析を始めてください。")
    try:
        input("(Enter) >> ")
    except:
        pass
    voice_client.stop()

    args = sys.argv[1:]
    
    if len(args) >= 2:
        path_yesterday = args[0]
        path_today = args[1]
        content_y = ""
        content_t = ""
        if os.path.exists(path_yesterday):
            with open(path_yesterday, 'r', encoding='utf-8') as f: content_y = f.read()
        if os.path.exists(path_today):
            with open(path_today, 'r', encoding='utf-8') as f: content_t = f.read()
        print("\n✅ ファイル読み込み完了: 2つのファイルを比較します")
        return {"input_type": "dual_file", "yesterday_text": content_y, "today_text": content_t}

    elif len(args) == 1 and os.path.exists(args[0]):
        with open(args[0], 'r', encoding='utf-8') as f: content = f.read()
        print("\n✅ ファイル読み込み完了: 1つのファイルを分析します")
        return {"input_type": "single_file", "yesterday_text": "", "today_text": content}
    
    else:
        return {"input_type": "chat", "yesterday_text": "", "today_text": ""}

def interviewer_node(state: AgentState):
    print_phase("ヒアリング (Interviewer)")
    voice_client.stop() 
    greeting = "ファイルが見当たらなかったけど、今日はどんな一日だった？ 私にだけこっそり教えてよ。"
    voice_client.speak_async(greeting)
    
    print_guide("今日あったことを自由に入力してください（入力完了後にEnter）")
    user_input = input("(あなた) >> ")
    voice_client.stop() 

    messages = [
        SystemMessage(content=CORE_PERSONA),
        AIMessage(content=greeting),
        HumanMessage(content=user_input)
    ]
    ack_msg = "そっかそっか……。話してくれてありがとうね。"
    voice_client.speak_async(ack_msg)
    time.sleep(1.5)
    return {"today_text": user_input, "messages": messages}

def analyzer_node(state: AgentState):
    print_phase("分析中 (Analyzer)")
    voice_client.stop()
    print("(クマちゃんがログを読んでいます... 🧶)")
    
    if state['input_type'] == 'dual_file':
        prompt = f"""
        以下の2つのテキストを比較し、ユーザーの成果を分析して。
        【昨日のメモ（予定）】: {state['yesterday_text']}
        【今日のメモ（結果）】: {state['today_text']}
        指示:
        1. 昨日は未完了だったが、今日完了しているタスクの中から、「特に大変そう」「価値が高い」と思われるものを【トップ3】だけ抽出して。
        2. 全てを網羅する必要はない。
        """
    else:
        prompt = f"""
        以下のテキストから、ユーザーが今日成し遂げた「最も重要な成果」を3つ以内で抽出して。
        テキスト: {state['today_text']}
        """

    response = llm.invoke([SystemMessage(content=CORE_PERSONA), HumanMessage(content=prompt)])
    return {"analysis_summary": response.content}

def praiser_node(state: AgentState):
    print_phase("労いと称賛 (Praiser)")
    
    prompt = f"""
    分析結果: {state['analysis_summary']}
    上記を踏まえて、親友としてユーザーを褒めてください。
    【ルール】
    1. **全体で300文字以内（読み上げて1分程度）**。
    2. 分析された「トップの成果」に絞って、深く、温かく褒める。
    3. クマのぬいぐるみらしく、包容力のある言葉で。
    """
    response = llm.invoke([SystemMessage(content=CORE_PERSONA), HumanMessage(content=prompt)])
    
    voice_client.speak_async(response.content)
    
    print_guide("褒め言葉を受け取ってください。満足したらEnterキーで「明日の作戦」に進みます。")
    input("(Enter) >> ")
    voice_client.stop()

    return {"messages": [AIMessage(content=response.content)]}

def strategist_node(state: AgentState):
    print_phase("明日の作戦 (Strategist)")
    
    prompt = f"""
    分析結果: {state['analysis_summary']}
    明日のために、ユーザーの心を軽くする提案をして。
    【ルール】
    1. **150文字以内**。
    2. 「明日絶対にやるべき1つのこと（One Thing）」を提案する。
    3. それ以外は「明日はやらなくていい」と断言する。
    4. 「じゃあ、明日の作戦会議をしようか」から始めて。
    """
    response = llm.invoke([SystemMessage(content=CORE_PERSONA), HumanMessage(content=prompt)])
    
    voice_client.speak_async(response.content)
    
    print_guide("提案内容を確認してください。合意するならEnterキーを押してください。")
    input("(Enter) >> ")
    voice_client.stop()

    return {"plan_focus": response.content, "messages": [AIMessage(content=response.content)]}

def cheer_node(state: AgentState):
    print_phase("最後のエール (Cheer)")
    
    prompt = "最後に、ユーザーが安心して眠れるような、短く温かい「おやすみ」のエールを送って。30文字以内で、クマちゃんらしく。"
    response = llm.invoke([SystemMessage(content=CORE_PERSONA), HumanMessage(content=prompt)])
    
    voice_client.speak_async(response.content)
    time.sleep(1)
    
    print_guide("おやすみなさい。Enterキーを押すと終了します。")
    input("(Enter) >> ")
    voice_client.stop()
    
    return {"messages": [AIMessage(content=response.content)]}

def logger_node(state: AgentState):
    print_phase("ログ保存 (Logger)")
    filename = f"yell_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=== Midnight Partner Log ===\n")
        f.write(f"Type: {state.get('input_type')}\n")
        f.write(f"Plan: {state.get('plan_focus')}\n")
    print(f"\n✅ 会話の記録を {filename} に置いておいたよ。おやすみ。")
    return {}

# ==========================================
# Graph
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
workflow.add_edge("praiser", "strategist")
workflow.add_edge("strategist", "cheer")
workflow.add_edge("cheer", "logger")
workflow.add_edge("logger", END)
app = workflow.compile()

if __name__ == "__main__":
    app.invoke({"messages": []})
