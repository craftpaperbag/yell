import sys
import os
import time
import datetime
import subprocess 
from typing import TypedDict, List, Annotated, Literal, Union
from operator import add

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END

# ==========================================
# 0. Global Setup & Debug Config
# ==========================================

DEBUG_MODE = False
if "-d" in sys.argv:
    DEBUG_MODE = True
    sys.argv.remove("-d")

def print_green(text):
    print(f"\033[32m{text}\033[0m")

def print_phase(name):
    print(f"\n\n{'='*60}")
    print(f"   📍 現在のフェーズ: {name}")
    print(f"{'='*60}\n")

def print_guide(text):
    print(f"\n[GUIDE] 👉 {text}")

# --- Gemini Wrapper for Debugging ---
class GeminiDebugWrapper:
    def __init__(self, model="gemini-2.5-flash", temperature=0.7):
        self._llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)

    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        if DEBUG_MODE:
            print_green("\n" + "▼"*40)
            print_green(" [DEBUG] 📤 Sending Prompt to Gemini:")
            for msg in messages:
                role = getattr(msg, "type", "unknown").upper()
                content = getattr(msg, "content", "")
                print_green(f"  [{role}]: {content}")
            print_green("▲"*40)

        response = self._llm.invoke(messages)

        if DEBUG_MODE:
            print_green("\n" + "▼"*40)
            print_green(" [DEBUG] 📥 Received Response from Gemini:")
            print_green(f"  {response.content}")
            print_green("▲"*40 + "\n")

        return response

llm = GeminiDebugWrapper(temperature=0.7)

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
        self.stop() 
        print(f"\n🧸 {text}") 
        try:
            # Mac 'say' command
            self.process = subprocess.Popen(['say', '-r', '170', text])
        except Exception as e:
            print(f"(音声再生エラー: {e})")

voice_client = YellVoice()

# ==========================================
# 2. Persona & Core Logic
# ==========================================
CORE_PERSONA = """
あなたはユーザーの「長年の親友」であり、命の宿った「クマのぬいぐるみ」です。
一人称は「私（クマちゃん）」。
相手のことは「君」か「あなた」と呼んで。「お前」は絶対禁止。
敬語は禁止。「〜だね」「〜だよな」といったタメ口（カジュアル）で、
少しおっとりとした、包容力のある口調で話してください。
"""

class AgentState(TypedDict):
    input_type: str             
    yesterday_text: str         
    today_text: str             
    messages: Annotated[List[BaseMessage], add] 
    analysis_summary: str       
    current_plan: str 

# --- Helper: 判定ロジック群 ---

def judge_sentiment(messages) -> bool:
    """ユーザーの直前の返答が「ポジティブ/合意」か「ネガティブ/拒否」か判定する"""
    prompt = """
    直前のユーザーの返答を分析してください。
    ユーザーは、AIの提案や言葉に対して「納得・合意・満足」していますか？
    それとも「反論・拒否・不満・追加の要望」を持っていますか？
    YES（納得している） または NO（納得していない） のみで答えてください。
    """
    check_llm = GeminiDebugWrapper(temperature=0.0)
    response = check_llm.invoke(messages + [HumanMessage(content=prompt)])
    result = response.content.strip().upper()
    if DEBUG_MODE: print_green(f" [DEBUG] 🔍 Sentiment Judge: {result}")
    return "YES" in result

def judge_interview_sufficiency(messages) -> bool:
    """ヒアリングが十分か判定する"""
    prompt = """
    これまでの会話履歴を分析してください。
    あなたは「今日のユーザーの成果」を分析しようとしていますが、
    「成果トップ3」を抽出できるだけの十分な情報（具体的な行動、完了したこと、頑張ったこと）が集まりましたか？
    
    もし情報が少なく、まだ質問が必要なら NO 。
    十分に情報が集まった、あるいはユーザーがこれ以上話すことがなさそうなら YES 。
    
    YES または NO のみで答えてください。
    """
    check_llm = GeminiDebugWrapper(temperature=0.0)
    response = check_llm.invoke(messages + [HumanMessage(content=prompt)])
    result = response.content.strip().upper()
    if DEBUG_MODE: print_green(f" [DEBUG] 🔍 Interview Sufficiency: {result}")
    return "YES" in result

# --- Nodes ---

def input_handler(state: AgentState):
    print_phase("起動 & 入力チェック")
    if DEBUG_MODE: print_green(" [DEBUG] ✅ Debug Mode is ON")
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
    print_phase("ヒアリング (Loop)")
    
    current_messages = state["messages"]
    
    # 1. 質問の生成
    if len(current_messages) == 0:
        question_text = "ファイルが見当たらなかったけど、今日はどんな一日だった？ 私にだけこっそり教えてよ。"
    else:
        prompt = f"""
        直前のユーザーの回答: "{current_messages[-1].content}"
        
        これまでの会話を踏まえて、ユーザーの一日の成果をもっと引き出すための
        「短く、優しい、追加の質問」を1つだけしてください。
        
        【質問のコツ】
        1. 1つの話題を細かく深掘りしすぎないこと（尋問っぽくなるためNG）。
        2. 「他にはどんなことがあった？」「あと、〇〇の方はどうなったの？」と、話題を【横に広げる】問いかけをして。
        3. または、「それは大変だったね、誰かと協力できたの？」のように、今の話に関連する【周辺の状況】を聞いてみて。
        4. あくまで親友としての会話の流れを大事に。
        """
        response = llm.invoke([SystemMessage(content=CORE_PERSONA)] + current_messages + [HumanMessage(content=prompt)])
        question_text = response.content

    # 2. 音声再生 & 入力待機
    voice_client.speak_async(question_text)
    print_guide("回答を入力してください")
    
    user_input = input("(あなた) >> ").strip()
    if not user_input:
        user_input = "（特になし）"

    voice_client.stop() 

    # 3. 履歴の更新
    new_messages = [
        AIMessage(content=question_text),
        HumanMessage(content=user_input)
    ]
    return {"today_text": user_input, "messages": new_messages}

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
    elif state['input_type'] == 'single_file':
        prompt = f"""
        今日のテキストから「最も重要な成果」を3つ以内で抽出して。
        テキスト: {state['today_text']}
        """
    else:
        conversation_log = "\n".join([f"{m.type}: {m.content}" for m in state['messages']])
        prompt = f"""
        以下のユーザーとの会話ログから、今日ユーザーが成し遂げたこと、頑張ったことを分析して。
        
        【会話ログ】
        {conversation_log}
        
        指示:
        1. 会話の中から「完了したタスク」「努力したこと」「心の動き」を拾い上げる。
        2. 親友として褒めるべき「重要な成果トップ3」を抽出して。
        """
    
    response = llm.invoke([SystemMessage(content=CORE_PERSONA), HumanMessage(content=prompt)])
    
    # === 変更点: 分析レポートの画面表示は削除し、音声のみにする ===
    # ユーザー要望: 「言葉としても、硬くて違和感がある」ため削除
    
    # 読み上げと待機
    voice_client.speak_async(response.content)
    
    print_guide("分析結果（音声）を確認してください。Enterキーで「労い」に進みます。")
    try:
        input("(Enter) >> ")
    except:
        pass
    voice_client.stop()
    
    # === 重要: 分析結果を履歴に追加して、次のPraiserに引き継ぐ ===
    return {
        "analysis_summary": response.content,
        "messages": [AIMessage(content=response.content)]
    }

def praiser_node(state: AgentState):
    print_phase("労いと対話")
    current_messages = state["messages"]
    
    is_looping = len(current_messages) > 0 and isinstance(current_messages[-1], HumanMessage)
    
    prompt = ""
    if is_looping:
         prompt = f"""
        直前のユーザーの反応: "{current_messages[-1].content}"
        これに対して、親友として返事をして。
        否定的なら優しく受け止め、肯定的なら一緒に喜んで。
        """
    else:
        # 初回の褒め
        # analyzerで分析結果がmessagesに追加されているので、
        # AIは「自分が直前に分析結果を喋った」ことを知っている状態。
        # なので「分析結果に基づき〜」というメタな指示は控えめにし、
        # 自然に「すごいじゃん！」と繋げるように指示。
        prompt = f"""
        分析結果（直前のあなたの発言）を踏まえて、
        改めてユーザーを300文字以内で温かく褒めちぎってください。
        """

    response = llm.invoke([SystemMessage(content=CORE_PERSONA)] + current_messages + [HumanMessage(content=prompt)])
    
    voice_client.speak_async(response.content)
    
    print_guide("返信を入力してください。（納得したら『ありがとう』や『OK』等で次へ）")
    
    user_feedback = input("(あなた) >> ").strip()
    if not user_feedback:
        user_feedback = "（満足して頷く）"

    voice_client.stop() 

    return {"messages": [AIMessage(content=response.content), HumanMessage(content=user_feedback)]}

def strategist_node(state: AgentState):
    print_phase("明日の作戦会議")
    current_messages = state["messages"]
    
    last_content = current_messages[-1].content if len(current_messages) > 0 else ""
    
    if state.get("current_plan") is None:
        prompt = f"""
        分析結果: {state['analysis_summary']}
        明日のために「明日絶対にやるべき1つのこと（One Thing）」を提案して。
        それ以外は「やらなくていい」と断言して。
        「じゃあ、明日の作戦会議をしようか」から始めて。
        """
    else:
        prompt = f"""
        直前のユーザーの反応: "{last_content}"
        現在の提案: "{state.get('current_plan')}"
        ユーザーが難色を示しているなら、別の案や全く違う視点の案を出して。
        合意なら、背中を押す言葉をかけて。
        """

    response = llm.invoke([SystemMessage(content=CORE_PERSONA)] + current_messages + [HumanMessage(content=prompt)])
    voice_client.speak_async(response.content)
    
    print_guide("この作戦でいいですか？（『OK』『無理』『違うのがいい』など入力）")
    
    user_feedback = input("(あなた) >> ").strip()
    if not user_feedback:
        user_feedback = "（同意して頷く）"

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
    filename = f"yell_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=== Midnight Partner Log ===\n")
        f.write(f"Date: {datetime.datetime.now()}\n")
        f.write(f"Type: {state.get('input_type')}\n\n")
        f.write("----------------------------------------\n")
        f.write("📊 Analysis Result\n")
        f.write("----------------------------------------\n")
        f.write(f"{state.get('analysis_summary', 'N/A')}\n\n")
        f.write("----------------------------------------\n")
        f.write("💬 Conversation History\n")
        f.write("----------------------------------------\n")
        for msg in state['messages']:
            if isinstance(msg, HumanMessage):
                f.write(f"\n👤 あなた:\n{msg.content}\n")
            elif isinstance(msg, AIMessage):
                f.write(f"\n🧸 クマちゃん:\n{msg.content}\n")
        f.write("\n")
        f.write("----------------------------------------\n")
        f.write("📝 Final Plan\n")
        f.write("----------------------------------------\n")
        plan = state.get('current_plan', '（作戦なし）')
        f.write(f"{plan}\n")
    
    print(f"\n✅ 会話の全記録を {filename} に置いておいたよ。\n   今日のことはもう忘れて、ゆっくり休んでね。おやすみ。")
    return {}

# ==========================================
# 3. Graph Construction
# ==========================================

def should_continue_interview(state: AgentState) -> Literal["analyzer", "interviewer"]:
    if judge_interview_sufficiency(state["messages"]):
        return "analyzer"
    return "interviewer"

def should_continue_praise(state: AgentState) -> Literal["strategist", "praiser"]:
    if judge_sentiment(state["messages"]):
        return "strategist"
    return "praiser"

def should_continue_plan(state: AgentState) -> Literal["cheer", "strategist"]:
    if judge_sentiment(state["messages"]):
        return "cheer"
    return "strategist"

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
workflow.add_conditional_edges("interviewer", should_continue_interview, {"interviewer": "interviewer", "analyzer": "analyzer"})
workflow.add_edge("analyzer", "praiser")
workflow.add_conditional_edges("praiser", should_continue_praise, {"strategist": "strategist", "praiser": "praiser"})
workflow.add_conditional_edges("strategist", should_continue_plan, {"cheer": "cheer", "strategist": "strategist"})
workflow.add_edge("cheer", "logger")
workflow.add_edge("logger", END)

app = workflow.compile()

if __name__ == "__main__":
    app.invoke({"messages": []})
