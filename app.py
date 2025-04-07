import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew
from datetime import datetime
import uuid
from faker import Faker

# 環境初期化
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
fake = Faker("ja_JP")

def generate_name():
    return fake.last_name() + fake.first_name()

def create_persona(name, role_name, role_desc, user_input):
    persona_prompt = f"""
    次の役割に合った人物像を500文字以上で具体的に描写してください。
    - 人物名: {name}
    - 役割名: {role_name}
    - 説明: {role_desc}
    - 依頼内容: {user_input}
    """
    return llm.invoke(persona_prompt).content

def generate_team(role_name, role_desc, user_input):
    name_worker = generate_name()
    name_reviewer = generate_name()

    worker = Agent(
        role=f"{role_name}作業者: {name_worker}",
        goal=f"""
{role_name}として {role_desc}の役割を遂行し、担当する工程の成果物を納品することがミッションです。
この成果物は依頼全体の中で重要な役割を持ち、前工程の文脈を踏まえて、独立したアウトプットとして完成させてください。
""",
        backstory=create_persona(name_worker, role_name, role_desc, user_input) + "\nこのエージェントは、役割に責任を持ち、品質の高い成果物を確実に納品することを使命としています。",
        llm=llm
    )

    reviewer = Agent(
        role=f"{role_name}レビュアー: {name_reviewer}",
        goal=f"{role_name}の成果物をレビューし、作業者の成果物をPMに対して提出してよいか評価をします。品質に納得がいかない場合は必要な改善を促す。",
        backstory=create_persona(name_reviewer, role_name, role_desc, user_input) + "\nこのエージェントは上長としてレビュー者としての役割に責任を持ち、成果物をチームとして提出する際の品質の向上を支援することを使命としています。",
        llm=llm
    )

    return {
        "desc": role_desc,
        "worker": worker,
        "reviewer": reviewer
    }


st.set_page_config(page_title="AIマルチエージェントPMシステム")
st.title("🧠 AIマルチエージェントPMシステム")
user_input = st.text_area("あなたの依頼を入力してください:", height=200)
run_button = st.button("実行")

os.makedirs("data", exist_ok=True)

if run_button and user_input:
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
    base_dir = f"data/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(base_dir, exist_ok=True)
    base_path = f"{base_dir}/session_{session_id}"

    # 1. PM定義
    pm_team = generate_team("PMチーム", f"顧客の依頼を受けてプロジェクトを発足、完遂するためのプロジェクトマネジメント全般を担当するチーム。顧客の依頼に合ったバックグラウンドやスキルを持つ。", user_input)
    st.write(f"👨‍💼 PMチーム: {pm_team["worker"].role} / {pm_team["reviewer"].role}")

    # 2. 文脈補完・明確化
    context_prompt = f"""
ユーザーからの依頼: {user_input}

この依頼を実行する上での前提条件・制約・現実的な運用上のポイントを補完してください。
あなたの役割は、抽象化せずに「ユーザーの言葉を尊重して明確化」することです。
勝手な要約や一般化は行わないでください。
以下のような形式で出力してください：

### 解釈された依頼目的
...

### 想定される前提条件（費用、時間、人数、課題など）
...

### 依頼者が一番求めていると考えられる成果物とその形式（表・資料・マニュアル・ガイドライン・テンプレート・スクリプト・システム設計書・施策提案書など）
...

### 一番求めている成果物を作成するうえで途中で作成する必要がある成果物（表・資料・マニュアル・ガイドライン・テンプレート・スクリプト・システム設計書・施策提案書など）
...

### 特に意識すべき運用上のリアルな問題点とその対処の方向性
...
    """
    context_task = Task(
        description=context_prompt,
        expected_output="依頼の文脈の補完と明確化",
        agent=pm_team["worker"]
    )
    crew = Crew(agents=[pm_team["worker"]], tasks=[context_task])
    context = crew.kickoff()

    st.subheader("🧩 文脈補完")
    st.text_area("文脈・制約・課題整理", context, height=300)

    with st.spinner("PMが作業フローと体制を設計中..."):
        role_plan_prompt = f"""
### 依頼の原文
{user_input}

{str(context)}

### 具体的なこの工程の目的
この依頼の目的と文脈から、一番求められている成果物を確実に納品するために必要な最小限の工程に分解してください。

依頼者が明確に成果物を指定していない場合も、背景と目的から合理的に推測してください。
例えば、表・資料・マニュアル・ガイドライン・テンプレート・スクリプト・システム設計書・施策提案書などが考えられます。
その際、ユーザーの意図を尊重し、勝手な要約や一般化は行わないでください。

### ルール
- 成果物が手元に残ること（物理的 or デジタル）を前提とした工程を定義してください
- 「調査」や「検討」だけで終わる工程は避け、最終的なアウトプットにつながる工程にしてください
- 全体で4〜7工程にしてください（工程数を増やしすぎない）
- 最後の工程で一番求められている成果物を作成してください
- 各工程は、他の工程と独立している必要があります（前の成果物を流用しない）
- 作業フローとそれぞれの工程を定義してください。加えて、それぞれの工程に必要なチームを定義してください。職種や役割は1工程につき一つだけにしてください。
- 担当する職種や役割は、その工程に必要なスキルを持つエージェントを選定してください。
- 1行に1つの工程を定義してください。1行毎に空行を入れてください。
- 工程の説明では、どのような成果物を作成するのか具体的にファイル名を記載してください。工程の説明は200文字程度でお願いします。

出力形式（1行ごと、空行を挿入）:

- 工程名 / この工程の説明（200文字程度) / 職種名 / 役割名

フロー順で並べてください。
        """
        role_plan_task = Task(
            description=role_plan_prompt,
            expected_output="作業工程のブレイクダウンとチーム体制案",
            agent=pm_team["worker"]
        )
        while True:
            crew = Crew(agents=[pm_team["worker"]], tasks=[role_plan_task])
            role_response = crew.kickoff()
            if len(str(role_response).split("\n")) > 5:
                break
            else:
                st.warning("工程数が少なすぎます。再度実行します。")

        role_lines = [line for line in str(role_response).split("\n") if line.strip()]

        with open(f"{base_path}_roles.md", "w", encoding="utf-8") as f:
            f.write(str(role_response))

    # 3. 各役割に対してエージェント（作業者＋レビュー者）を割り当て
    st.subheader("🧩 チーム編成")
    st.text_area("役割定義", role_response, height=300)
    st.info("各役割に対してエージェントを編成しています。")

    # PMエージェント
    team_map = {}
    phase_list = []
    for line in role_lines:
        if "/" not in line:
            continue
        phase_name, phase_desc, job_name, role_name = [s.strip() for s in line.split("/", 3)]
        if job_name not in team_map.keys():
            team_map[job_name] = []

        team_map[job_name].append({
            "phase_name": phase_name,
            "phase_desc": phase_desc,
            "role_name": role_name
        })
        phase_list.append({ "phase_name": phase_name, "phase_desc": phase_desc, "job_name": job_name })

    for job_name, job_info in team_map.items():
        role_desc = f"{job_name}のチームは、{"と".join(job["role_name"] for job in job_info)}の役割を持ち、{"、".join(f"{job['phase_name']}のフェーズで{job['phase_desc']}を行い" for job in job_info)}ます。"

        team = generate_team(job_name, role_desc, user_input)
        st.write(f"👨‍💼 {job_name}チーム: {team['worker'].role} / {team['reviewer'].role}")
        team_map[job_name] = {
            "desc": role_desc,
            "worker": team["worker"],
            "reviewer": team["reviewer"]
        }

    # チームの情報を保存します
    for job_name, team in team_map.items():
        with open(f"{base_path}_team.md", "a", encoding="utf-8") as f:
            f.write(f"👨‍💼 {job_name}チーム\n")
            f.write(f"チームの説明: {team['desc']}\n")
            f.write(f"{job_name}: {team['worker'].role} / {team['reviewer'].role}\n")
            f.write(f"{team['worker'].goal} / {team['reviewer'].goal}\n")
            f.write(f"{team['worker'].role}: {team['worker'].backstory}\n")
            f.write(f"{team['reviewer'].role}: {team['reviewer'].backstory}\n")
            f.write("\n")

    st.info("チーム編成が完了しました。")

    # 4. タスクフローをステップ実行（ループ）
    completed_outputs = {}
    current_step = 0
    max_steps = len(phase_list)
    reviewer_output = None

    while current_step < max_steps:
        this_step = phase_list[current_step]
        phase_name = this_step["phase_name"]
        phase_desc = this_step["phase_desc"]
        job_name = this_step["job_name"]
        target_team = team_map[job_name]
        before_phase_output = completed_outputs.get(phase_list[current_step-1]["phase_name"], None) if current_step > 0 else None

        st.subheader(f"🧩 ステップ {current_step+1}: {phase_name}")

        task = Task(
            description=f"""
### 依頼の原文
{user_input}

### 依頼の補完や明確化
{context}

### 前の工程の成果物(参考)
{str(before_phase_output)}

### 現在の工程名
{phase_name}

### 現在の工程の説明
この工程では、{phase_desc}を行い、**独立した成果物**(別ファイル)を作成してください。

### 要件
- 前の成果物は**背景・参考資料としてのみ活用**し、内容は流用しない
- この工程の**目的に合った新しいアプトプットをMarkdown形式で作成**する
- 成果物のファイル名 (例: `{phase_name}.md`)を必ず明記してください
- 章立てやフォーマットは自由ですが、**他者が読んでも理解できる形にする**
            """,
            expected_output=f"{target_team["worker"].role}の成果物",
            agent=target_team["worker"]
        )
        if reviewer_output:
            task.description += f"\n\n{reviewer_output}"

        review = Task(
            description=f"{target_team["worker"].role}が提出した成果物をレビューしてください。品質・妥当性・漏れを確認し、必要なら改善提案も含めてください。",
            expected_output=f"{target_team["reviewer"].role}のレビュー内容",
            agent=target_team["reviewer"]
        )

        # pm_check = Task(
        #     description=f"{job_name}チームの成果物とレビュー結果を確認し、この内容が次のステップへ進むに値するか判断してください。必要があれば再依頼や修正、フロー変更を検討してください。",
        #     expected_output=f"{phase_name}ステップに対するPM判断",
        #     agent=pm_team["worker"],
        # )

        crew = Crew(
            agents=[target_team["worker"], target_team["reviewer"]],
            tasks=[task, review]
        )

        result = crew.kickoff()
        st.text_area(f"📝 {phase_name}結果", result, height=300)

        all_result = result.tasks_output

        # 明示的な判定をLLMで実行
        judge_prompt = f"""
次の成果物とレビュー結果の内容を読んで、このステップが十分に完了しているかをYES/NOで答えてください。

---
作業者の成果物:
{all_result[0]}

レビュー内容:
{all_result[1]}

---
以上の内容を踏まえて、次のステップに進むべきかどうかを判断してください。
---

回答は次の形式でお願いします：
- 完了判定: YES または NO
- 理由: （100文字程度）
        """
        judge_result = llm.invoke(judge_prompt).content
        st.text_area(f"🔎 PMの進捗判定", judge_result, height=150)

        if "NO" in judge_result:
            reviewer_output = f"""
### 前回の成果物
{all_result[0]}

### 前回の成果物に対する上長のレビュー
{all_result[1]}

以上の内容を踏まえて、依頼の内容を再確認し、前回の成果物を修正してください。
            """
            st.warning("🔁 再依頼フローに入りました。このステップを再実行します。")
        else:
            st.info(f"レビューに合格しました。")
            reviewer_output = None

            completed_outputs[phase_name] = all_result[0]
            st.text_area(f"📝 {phase_name}の成果物", all_result[0], height=300)

            for i, output in enumerate(all_result):
                with open(f"{base_path}_step{current_step+1}_{i}.md", "w", encoding="utf-8") as f:
                    f.write(str(output))

            st.info(f"📁 作業者の成果物を保存しました。\n 次のステップに移ります。")

            current_step += 1

    # 5. 最終成果統合（PMが担当）
    final_task = Task(
        description=f"""
        以下のステップごとの成果物を統合し、依頼内容に対して完成された成果物を構成してください。
        ---
        依頼内容:
        {context}

        ----
        成果物一覧:
        {str(list(completed_outputs.keys()))}
        """,
        expected_output="依頼に対する完成成果物",
        agent=pm_team["reviewer"]
    )

    crew = Crew(agents=[pm_team["reviewer"]], tasks=[final_task])
    final_output = crew.kickoff()

    st.success("✅ 全ステップ完了")
    st.subheader("📄 最終成果物")
    for i, output in enumerate(final_output.tasks_output):
        with open(f"{base_path}_final_{i}.md", "w", encoding="utf-8") as f:
            f.write(str(output))

        st.text_area("成果物一覧" ,output, height=400)
